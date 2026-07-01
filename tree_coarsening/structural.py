"""Occurrence-specific exact structural types for schema 1.0.

Matching labels are deliberately opaque. All geometry required for rewiring
and decoding lives in :class:`CompositeType` occurrences.
"""

from __future__ import annotations

from collections.abc import Hashable as HashableABC, Iterator
from dataclasses import dataclass
from typing import Any, Hashable, Literal, TypeAlias

from .exceptions import ExactTypeError

MatchingLabel: TypeAlias = Hashable
BaseType: TypeAlias = tuple[Literal["base"], str]
AttachMap: TypeAlias = tuple[int, ...]


def _require_hashable(value: Any, *, what: str) -> None:
    if not isinstance(value, HashableABC):
        raise ExactTypeError(f"{what} must be hashable; got {value!r}.")
    try:
        hash(value)
    except Exception as exc:
        raise ExactTypeError(f"{what} cannot be hashed reliably: {value!r}.") from exc


def base_type(raw_label: str) -> BaseType:
    """Return the exact type of one raw node label."""

    if not isinstance(raw_label, str):
        raise ExactTypeError(f"raw base labels must be strings; got {raw_label!r}.")
    return ("base", raw_label)


def is_base_type(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == "base"
        and isinstance(value[1], str)
    )


@dataclass(frozen=True)
class CompositeType:
    """Exact immutable recipe for one contracted occurrence."""

    # Keep the five normative dataclass fields unchanged while retaining private
    # derived geometry in slots. Derived caches are allowed by schema 1.0 and
    # prevent deep exact-type chains from being re-walked recursively.
    __slots__ = (
        "model_id",
        "label",
        "parent",
        "components",
        "attach",
        "_site_count",
        "_root_count",
        "_hash",
    )

    model_id: str
    label: MatchingLabel
    parent: tuple[int, ...]
    components: tuple["ExactType", ...]
    attach: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ExactTypeError("CompositeType.model_id must be a nonempty string.")
        _require_hashable(self.label, what="CompositeType.label")
        if not isinstance(self.parent, tuple):
            raise ExactTypeError("CompositeType.parent must be a tuple.")
        if not isinstance(self.components, tuple):
            raise ExactTypeError("CompositeType.components must be a tuple.")
        if not isinstance(self.attach, tuple):
            raise ExactTypeError("CompositeType.attach must be a tuple.")

        n = len(self.parent)
        if n == 0:
            raise ExactTypeError("CompositeType requires at least one component.")
        if len(self.components) != n:
            raise ExactTypeError(
                "CompositeType.parent and CompositeType.components must have equal length."
            )
        for i, parent_i in enumerate(self.parent):
            if (
                not isinstance(parent_i, int)
                or isinstance(parent_i, bool)
                or parent_i < -1
                or parent_i >= n
                or parent_i == i
            ):
                raise ExactTypeError(f"invalid CompositeType.parent[{i}]={parent_i!r}.")
        self._check_parent_acyclic()

        for i, component in enumerate(self.components):
            if not is_exact_type(component):
                raise ExactTypeError(f"component {i} is not a valid exact type: {component!r}.")
        for i, value in enumerate(self.attach):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ExactTypeError(
                    f"CompositeType.attach[{i}] must be an integer; got {value!r}."
                )

        component_site_counts = tuple(exact_site_count(item) for item in self.components)
        component_root_counts = tuple(exact_root_count(item) for item in self.components)
        site_count = sum(component_site_counts)
        root_count = sum(
            component_root_counts[i] for i, parent_i in enumerate(self.parent) if parent_i == -1
        )

        expected = sum(
            component_root_counts[i] for i, parent_i in enumerate(self.parent) if parent_i != -1
        )
        if len(self.attach) != expected:
            raise ExactTypeError(
                f"CompositeType.attach has length {len(self.attach)}; expected {expected}."
            )

        cursor = 0
        for i, parent_i in enumerate(self.parent):
            if parent_i == -1:
                continue
            width = component_root_counts[i]
            parent_size = component_site_counts[parent_i]
            piece = self.attach[cursor : cursor + width]
            cursor += width
            bad = tuple(q for q in piece if q < 0 or q >= parent_size)
            if bad:
                raise ExactTypeError(
                    f"component {i} attaches to invalid sites {bad!r}; "
                    f"parent component {parent_i} has {parent_size} sites."
                )

        object.__setattr__(self, "_site_count", site_count)
        object.__setattr__(self, "_root_count", root_count)
        object.__setattr__(
            self,
            "_hash",
            hash((self.model_id, self.label, self.parent, self.components, self.attach)),
        )

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, CompositeType)

        pending: list[tuple[CompositeType, CompositeType]] = [(self, other)]
        while pending:
            left, right = pending.pop()
            if left is right:
                continue
            if (
                left.model_id != right.model_id
                or left.label != right.label
                or left.parent != right.parent
                or left.attach != right.attach
                or len(left.components) != len(right.components)
            ):
                return False
            for left_component, right_component in zip(
                left.components, right.components, strict=True
            ):
                if left_component is right_component:
                    continue
                if isinstance(left_component, CompositeType):
                    if not isinstance(right_component, CompositeType):
                        return False
                    pending.append((left_component, right_component))
                elif left_component != right_component:
                    return False
        return True

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        # Match the generated dataclass representation, but use an explicit
        # stack so a deep exact type does not consume the Python call stack.
        pieces: list[str] = []
        pending: list[ExactType | str] = [self]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                pieces.append(item)
                continue
            if not isinstance(item, CompositeType):
                pieces.append(repr(item))
                continue

            pieces.append(
                f"{type(item).__qualname__}("
                f"model_id={item.model_id!r}, "
                f"label={item.label!r}, "
                f"parent={item.parent!r}, "
                "components=("
            )
            pending.append(f"), attach={item.attach!r})")
            if len(item.components) == 1:
                pending.append(",")
                pending.append(item.components[0])
            else:
                for i in range(len(item.components) - 1, -1, -1):
                    if i < len(item.components) - 1:
                        pending.append(", ")
                    pending.append(item.components[i])
        return "".join(pieces)

    def __copy__(self) -> CompositeType:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> CompositeType:
        # Exact types and their labels are immutable by contract. Sharing this
        # value is therefore equivalent to recursively copying an arbitrarily
        # deep component tree, without consuming the Python call stack.
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (self.model_id, self.label, self.parent, self.components, self.attach),
        )

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def root_positions(self) -> tuple[int, ...]:
        return tuple(i for i, parent_i in enumerate(self.parent) if parent_i == -1)

    @property
    def site_count(self) -> int:
        return self._site_count

    @property
    def root_count(self) -> int:
        return self._root_count

    def attachment_slice(self, component_index: int) -> AttachMap:
        if component_index < 0 or component_index >= self.n_components:
            raise IndexError(component_index)
        if self.parent[component_index] == -1:
            return ()
        cursor = 0
        for i, parent_i in enumerate(self.parent):
            if parent_i == -1:
                continue
            width = exact_root_count(self.components[i])
            if i == component_index:
                return tuple(self.attach[cursor : cursor + width])
            cursor += width
        raise AssertionError("unreachable")

    def attachment_slices(self) -> tuple[AttachMap, ...]:
        out: list[AttachMap] = []
        cursor = 0
        for i, parent_i in enumerate(self.parent):
            if parent_i == -1:
                out.append(())
                continue
            width = exact_root_count(self.components[i])
            out.append(tuple(self.attach[cursor : cursor + width]))
            cursor += width
        return tuple(out)

    def _check_parent_acyclic(self) -> None:
        state = bytearray(len(self.parent))
        for start in range(len(self.parent)):
            if state[start] == 2:
                continue
            path: list[int] = []
            current = start
            while current != -1 and state[current] == 0:
                state[current] = 1
                path.append(current)
                current = self.parent[current]
            if current != -1 and state[current] == 1:
                raise ExactTypeError("CompositeType.parent contains a cycle.")
            for item in path:
                state[item] = 2


ExactType: TypeAlias = BaseType | CompositeType


def is_exact_type(value: Any) -> bool:
    return is_base_type(value) or isinstance(value, CompositeType)


def exact_type_label(value: ExactType) -> MatchingLabel:
    if is_base_type(value):
        return value[1]
    if isinstance(value, CompositeType):
        return value.label
    raise ExactTypeError(f"not an exact type: {value!r}.")


def exact_site_count(value: ExactType) -> int:
    if is_base_type(value):
        return 1
    if isinstance(value, CompositeType):
        return value.site_count
    raise ExactTypeError(f"not an exact type: {value!r}.")


def exact_root_count(value: ExactType) -> int:
    if is_base_type(value):
        return 1
    if isinstance(value, CompositeType):
        return value.root_count
    raise ExactTypeError(f"not an exact type: {value!r}.")


def iter_exact_types(value: ExactType) -> Iterator[ExactType]:
    stack: list[ExactType] = [value]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, CompositeType):
            stack.extend(reversed(current.components))


def iter_base_labels(value: ExactType) -> Iterator[str]:
    """Yield raw base labels in exact site order without recursive calls."""

    stack: list[ExactType] = [value]
    while stack:
        current = stack.pop()
        if is_base_type(current):
            yield current[1]
        elif isinstance(current, CompositeType):
            stack.extend(reversed(current.components))
        else:  # pragma: no cover - callers validate exact types first
            raise ExactTypeError(f"not an exact type: {current!r}.")


def exact_type_labels(value: ExactType) -> frozenset[MatchingLabel]:
    return frozenset(exact_type_label(item) for item in iter_exact_types(value))


def exact_type_model_ids(value: ExactType) -> frozenset[str]:
    return frozenset(
        item.model_id for item in iter_exact_types(value) if isinstance(item, CompositeType)
    )
