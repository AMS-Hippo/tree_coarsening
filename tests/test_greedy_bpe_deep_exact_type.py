from __future__ import annotations

from copy import deepcopy
from dataclasses import fields

import networkx as nx
import pytest

from tree_coarsening import (
    CompositeType,
    EdgeBPECoarsener,
    GreedyBPECoarsener,
    NamedVertexCoarsener,
    combine,
    exact_site_count,
    validate_encoded_tree,
)
from tree_coarsening.coarseners import GreedyBPECoarsener as NamespacedGreedyBPECoarsener
from tree_coarsening.coarseners.edge_bpe_numba import numba_available
from tree_coarsening.structural import base_type

from conftest import make_tree, raw_signature


REQUIRES_NUMBA = pytest.mark.skipif(
    not numba_available(),
    reason="optional Numba backend is not installed",
)


def _star(n_children: int, *, prefix: str = "deep-star") -> nx.DiGraph:
    graph = nx.DiGraph(name=prefix)
    graph.add_node(0, label="A", time=0.0, uid=(prefix, 0), marker="root")
    for child in range(1, n_children + 1):
        graph.add_node(
            child,
            label="B",
            time=float(child),
            uid=(prefix, child),
            marker=child,
        )
        graph.add_edge(0, child)
    return graph


def _deep_type(depth: int) -> CompositeType:
    current = base_type("A")
    for rank in range(depth):
        current = CompositeType(
            model_id="deep",
            label=("deep", rank),
            parent=(-1, 0),
            components=(current, base_type("B")),
            attach=(0,),
        )
    assert isinstance(current, CompositeType)
    return current


def test_greedy_bpe_is_public_api() -> None:
    assert GreedyBPECoarsener is NamespacedGreedyBPECoarsener


def test_private_geometry_cache_preserves_frozen_public_fields() -> None:
    assert tuple(field.name for field in fields(CompositeType)) == (
        "model_id",
        "label",
        "parent",
        "components",
        "attach",
    )
    exact = _deep_type(2_000)
    equivalent = _deep_type(2_000)
    assert exact_site_count(exact) == 2_001
    assert exact == equivalent
    assert hash(exact) == hash(equivalent)
    assert repr(exact).startswith("CompositeType(")
    assert deepcopy(exact) is exact


@pytest.mark.parametrize("coarsener_class", [EdgeBPECoarsener, GreedyBPECoarsener])
def test_deep_star_full_round_trip_without_recursion_error(coarsener_class: type) -> None:
    graph = _star(600)
    kwargs = {
        "num_merges": 600 if coarsener_class is EdgeBPECoarsener else 1,
        "min_pair_count": 1,
        "backend": "python",
        "model_id": coarsener_class.__name__,
    }
    model = coarsener_class(**kwargs).fit([graph])
    encoded = model.transform(graph)
    validate_encoded_tree(encoded)
    assert encoded.number_of_nodes() == 1
    assert next(iter(encoded.nodes(data=True)))[1]["size"] == 601
    decoded = model.decode(encoded)
    assert raw_signature(decoded) == raw_signature(graph)


def test_one_greedy_seed_can_emit_multiple_ordered_edge_rules() -> None:
    graph = _star(6, prefix="greedy-multi-rule")
    model = GreedyBPECoarsener(
        num_merges=1,
        min_pair_count=2,
        backend="python",
        model_id="greedy-multi-rule",
    ).fit([graph])

    assert len(model.history_) == 6
    assert model.history_[0]["parent_label"] == "A"
    assert model.history_[0]["child_label"] == "B"
    assert [event["child_label"] for event in model.history_] == ["B"] * 6
    assert len(model.encoder_.rules) == len(model.history_)

    encoded = model.transform(graph)
    validate_encoded_tree(encoded)
    assert encoded.number_of_nodes() == 1
    assert raw_signature(model.decode(encoded)) == raw_signature(graph)


def test_greedy_transform_is_finite_when_transform_run_exceeds_fit_run() -> None:
    fit_graph = _star(3, prefix="fit-short")
    longer_graph = _star(5, prefix="transform-long")
    model = GreedyBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        backend="python",
        model_id="greedy-finite",
    ).fit([fit_graph])

    assert len(model.history_) == 3
    encoded = model.transform(longer_graph)
    validate_encoded_tree(encoded)
    assert encoded.number_of_nodes() == 3
    assert sorted(data["size"] for _node, data in encoded.nodes(data=True)) == [1, 1, 4]
    assert raw_signature(model.decode(encoded)) == raw_signature(longer_graph)


@REQUIRES_NUMBA
def test_greedy_python_numba_rule_parity_on_repeated_star() -> None:
    graph = _star(80, prefix="greedy-numba-star")
    python_model = GreedyBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        backend="python",
        model_id="parity",
    ).fit([graph])
    numba_model = GreedyBPECoarsener(
        num_merges=1,
        min_pair_count=1,
        backend="numba",
        model_id="parity",
    ).fit([graph])
    assert python_model.history_ == numba_model.history_
    assert python_model.encoder_.rules == numba_model.encoder_.rules
    assert nx.utils.graphs_equal(python_model.transform(graph), numba_model.transform(graph))


@pytest.mark.parametrize("score", ["count", "normalized", "size_weighted"])
@REQUIRES_NUMBA
def test_greedy_python_numba_rule_parity_on_branching_corpus(score: str) -> None:
    corpus = [
        make_tree(
            ["R", "A", "B", "B", "C", "A", "B", "C", "B", "D"],
            [None, 0, 1, 1, 2, 0, 5, 5, 6, 0],
            prefix=f"greedy-parity-{score}-{index}",
        )
        for index in range(3)
    ]
    kwargs = {
        "num_merges": 4,
        "min_pair_count": 1,
        "pair_score": score,
        "model_id": f"greedy-parity-{score}",
    }
    python_model = GreedyBPECoarsener(backend="python", **kwargs).fit(corpus)
    numba_model = GreedyBPECoarsener(backend="numba", **kwargs).fit(corpus)

    assert numba_model.backend_used_ == "numba"
    assert python_model.history_ == numba_model.history_
    assert python_model.encoder_.rules == numba_model.encoder_.rules
    for graph in corpus:
        encoded = python_model.transform(graph)
        validate_encoded_tree(encoded)
        assert raw_signature(python_model.decode(encoded)) == raw_signature(graph)


def test_greedy_bpe_composes_with_other_schema1_stages() -> None:
    raw = make_tree(
        ["R", "A", "B", "B", "B", "C", "A", "B", "B", "Z"],
        [None, 0, 1, 1, 1, 0, 0, 6, 6, 0],
        prefix="greedy-compose",
    )
    named = NamedVertexCoarsener(labels={"Z"}, model_id="greedy-compose-named").fit([raw])
    first = named.transform(raw)
    greedy = GreedyBPECoarsener(
        num_merges=2,
        min_pair_count=1,
        model_id="greedy-compose-bpe",
    ).fit([first])

    encoder, decoder = combine(
        (named.encoder_, greedy.encoder_),
        (named.decoder_, greedy.decoder_),
    )
    encoded = encoder.transform(raw)
    validate_encoded_tree(encoded)
    assert raw_signature(decoder.decode(encoded)) == raw_signature(raw)
