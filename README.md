# Shift-aware, budget-matched evaluation of tabular foundation models

This repository contains the evaluation protocol, the runner that executes it,
the automated review gate that certifies a run, and the frozen artefacts of the
study reported in the accompanying paper. Everything needed to re-run the study
or to apply the protocol to other models and datasets is included.

Anonymised for peer review. Author, affiliation, funding and acknowledgement
information has been removed and will be restored on acceptance.

## What the protocol does

The unit of evaluation is a triple of dataset, model and seed. A single fitted
model is evaluated on both the in-distribution and the out-of-distribution test
split of its dataset, so that the shift response is measured within one unit
rather than across separate runs. Three properties are enforced by construction
and asserted afterwards by the gate.

**Frozen subsampling.** Subsample indices are derived from the dataset name and
the seed alone, never from the model. Every model therefore receives byte-identical
training and test rows, and the SHA-256 of each index set is recorded in the unit
record so the property can be checked rather than trusted.

**Matched tuning budgets.** Gradient-boosting baselines receive a disclosed
budget of 30 random-search trials with three-fold cross-validation inside the
training partition. The full trial history, the sampler, the wall-clock cost and
the optimism gap (held-out loss minus inner cross-validation loss) are logged for
every unit, and the incumbent is refitted at 0, 5, 10, 20 and 30 trials to give a
budget curve. Tabular foundation models are run with their released checkpoints
and no tuning, which is what the budget curve is compared against.

**Declared device policy.** Foundation models run on GPU; trees, the linear model
and the MLP run on CPU with pinned thread counts, because tree building at these
sample sizes is markedly slower on GPU and slower still with unpinned OpenMP. The
device of every unit is recorded and checked.

Accuracy is the primary metric. Balanced accuracy, AUROC, log-loss and the Brier
score are reported alongside it, and calibration is measured both by a 15-bin
expected calibration error and by the binning-free smooth calibration error, so
that the calibration claim does not rest on a binning choice. Significance uses a
Friedman omnibus test with Nemenyi post-hoc comparisons per condition, paired
Wilcoxon signed-rank tests per suite against the strongest boosting baseline, and
bootstrap confidence intervals over datasets.

## Layout

```
configs/grid.yaml          datasets, models, seeds, budgets, device policy
configs/gate_config.yaml   gate thresholds and the declared comparative claims
src/data_prep.py           one-off download and standardisation; writes registry.json
src/download_models.py     one-off checkpoint download, so runs work offline
src/bench_runner.py        job-based orchestrator: one subprocess per unit
src/unit_worker.py         per-unit science: subsampling, fitting, tuning, metrics
src/aggregate.py           unit records -> tidy CSVs
src/stats.py               omnibus, post-hoc, per-suite and bootstrap statistics
src/make_figures.py        the four displays
src/review_gate.py         the automated gate; non-zero exit blocks reporting
tests/test_gate.py         proves the gate separates a clean run from a dirty one
pipeline.sh                one command per stage
results/final_run/         the frozen certified run reported in the paper
results/dataset_summary.csv  the realised datasets and evaluated split sizes
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./pipeline.sh gate_test      # gate unit test, no network, seconds
./pipeline.sh smoke          # synthetic end-to-end chain, under two minutes
./pipeline.sh prep           # download every dataset and checkpoint, once
./pipeline.sh micro          # measured timing slice, prints an extrapolation
./pipeline.sh full           # the grid, resume enabled
./pipeline.sh final          # fresh directory, resume disabled, gate must pass
```

After `prep` the pipeline exports the offline flags of the model hubs and refuses
to start unless the readiness markers are present, so no stage reaches the network
mid-run. Long stages should be started inside a terminal multiplexer.

TableShift is not distributed on PyPI and is installed separately as a data
loader; the commands are given at the top of `requirements.txt`. Preparation
records every dataset it could not obtain, together with the reason, so the
realised dataset list is an auditable outcome rather than a choice made after
seeing results.

## The gate

The gate reads a completed run directory and exits non-zero if numbers that
should not be reported would reach a figure. It is configured before the final
pass, so its acceptance conditions cannot be adjusted to suit the numbers.

| Check | Asserts |
|---|---|
| A1 | the reported pass ran with resume disabled, skipped no unit, and carries a single run identity |
| B1 | every declared comparative claim has a run on datasets from a different source family |
| C1 | calibration was recorded |
| D1 | the optimism gap was recorded for every tuned configuration |
| E1 | the statistical artefacts exist |
| F1 | all models within a dataset and seed shared identical frozen subsamples |
| F2 | every unit ran on the device its model class was assigned |
| F3 | every tuned unit carries a complete tuning log at the declared budget |
| F4 | no unit failed or timed out |

Run it against the certified run shipped here:

```bash
python src/review_gate.py results/final_run --config configs/gate_config.yaml
```

The stored output of that command is `results/final_run/gate_report.txt`: nine
checks, all passing.

## Reproducing the reported numbers

`results/final_run/` is the run directory exactly as the runner produced it: one
JSON record per unit with its metrics, subsample hashes, tuning log, library
versions and wall-clock cost, plus the manifest, the run metadata, the aggregated
CSVs, the statistics and the figures. The reported grid is 17 datasets, 8 models
and 5 seeds, that is 680 units, all completed, in 12.9 hours on a single node
with one A100-class GPU and 32 CPU cores.

Aggregation and statistics can be regenerated from the unit records alone:

```bash
python src/aggregate.py results/final_run
python src/stats.py results/final_run
python src/make_figures.py results/final_run
```

Doing so reproduces the shipped CSVs to the last digit and the shipped statistics
byte for byte. Re-running the experiments themselves reproduces the numbers up to
the nondeterminism of the underlying libraries; an independent earlier pass over
the same grid gave identical values on every reported quantity.

## Data

All datasets are public. The three suites are kept disjoint by source: TableShift
tasks that are not derived from the American Community Survey, spatial-shift tasks
built from public census microdata through folktables, and a ten-dataset binary
subset of OpenML-CC18 as an i.i.d. reference. Two candidate tasks require
credentialed access and are dropped when the credentials are absent; three further
candidates were dropped for reasons recorded in `configs/grid.yaml` and in the
registry written by preparation.

## Environment

`requirements.txt` pins the versions used for the reported results. Two pins are
deliberate rather than incidental: the numpy version predates the baseline CPU
instruction requirement of the 2.x series, which some virtualised hosts do not
satisfy, and the xport version is the one that reads the source files whose
variable names begin with an underscore. Later versions of both work on current
hardware. Each unit record stores the library versions under which it ran, so the
provenance of a number does not depend on this file.

## Licence

Released under the MIT Licence; see `LICENSE`.
