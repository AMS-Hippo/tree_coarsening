# Greedy edge BPE

`GreedyBPECoarsener` is a fit-time scheduling variant of `EdgeBPECoarsener`. It
uses the same schema-1 rule representation, encoder, structural decoder, output
labels, fitting-size accounting, validation behavior, and optional Numba
training backend as ordinary edge BPE.

For each globally selected seed pair `(A, B)`, greedy scheduling first emits the
ordinary edge-BPE rule for that pair. It then forces the repeated-child
continuation `((A, B), B)`, `(((A, B), B), B)`, and so on until the newly
created parent label has no current `B` child. Only then does fitting return to
global pair selection.

`num_merges` counts globally selected seed pairs, not the number of emitted
ordered edge rules. One seed may therefore emit multiple rules. Forced
continuations ignore `min_pair_count`; the threshold applies only to seed-pair
selection.

The method remains a finite fitted program. A transform graph with a longer
repeated-child run than any run seen during fitting may retain trailing children
instead of inventing new rules. Deep left-nested exact types are supported by the
shared iterative `CompositeType` geometry/cache implementation.
