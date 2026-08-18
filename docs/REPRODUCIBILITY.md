# Reproducibility statement

**Data.** Public sources only. The realised dataset list, together with the
reason for every candidate that could not be obtained, is written by preparation
to `data/prepared/registry.json` and summarised in `results/dataset_summary.csv`.
Preprocessing is identical for all models: categorical columns are ordinal
encoded, missing numeric values are median imputed, and scaling is applied inside
the pipelines of the models that require it.

**Determinism.** The outcome of a unit is a function of the dataset, the model,
the seed and the pinned environment. Subsample indices derive from the dataset
and the seed alone; their SHA-256 hashes are stored in every unit record and are
asserted to be identical across models by gate check F1.

**Reported numbers.** They come from a single pass executed from scratch with
resume disabled into a fresh directory, certified by the gate (`gate_report.txt`,
nine of nine checks). The manifest records the status, wall-clock cost and run
identity of all 680 units.

**Tuning.** Every tuned unit stores its budget, sampler, cross-validation
folds, complete trial history, wall-clock cost and optimism gap. The
budget curve refits the incumbent at 0, 5, 10, 20 and 30 trials from that same
history.

**Compute.** Per-unit wall-clock cost and, for GPU units, peak allocated and
reserved memory are stored in the unit records. The reported grid took 12.9 hours
on one node with a single A100-class GPU and 32 CPU cores.

**Failure handling.** Units are independent subprocesses with a per-unit timeout.
A failed unit is recorded in the manifest and does not stop the batch, but the
gate refuses to certify a pass in which any unit failed or timed out.
