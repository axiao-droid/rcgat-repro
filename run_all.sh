#!/usr/bin/env bash
# End-to-end reproduction of the RC-GAT experiments (see README.md).
#
#   bash run_all.sh              # full chain
#   SKIP_FETCH=1 bash run_all.sh # reuse data/raw_* and data/graphs/
#
# Every phase is resumable: an existing per-job JSON is never re-run, so an
# interrupted run_all.sh can simply be started again.
#
# Two switches (README sections 5.10-5.12) do not change the main chain:
#   RESULTS_ROOT=<dir>          write to a separate results root instead of
#                               results/ -- used for replication runs so the
#                               canonical numbers can never be overwritten
#   CHECKPOINT_RULE=best_val|last|first|best_val_with_init
#                               which weights get reported; recorded in every
#                               run JSON and audited by check E3
set -euo pipefail

cd "$(dirname "$0")"

# Reproducibility: single-threaded BLAS keeps the random-SVD bundle bitwise
# stable (README 5.4); the workers themselves are processes, not threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MAVEN_MIN_DATE="${MAVEN_MIN_DATE:-2021-04-19}"
WORKERS="${WORKERS:-4}"
DATASETS="${DATASETS:-npm maven}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

step () { echo "== $* =="; }

if [ "${SKIP_FETCH:-0}" != "1" ]; then
  step "fetch npm (Huawei Cloud mirror)";  python3 -u src/fetch_npm.py   2>&1 | tee "$LOGDIR/fetch_npm.log"
  step "fetch Maven Central";              python3 -u src/fetch_maven.py 2>&1 | tee "$LOGDIR/fetch_maven.log"
  step "build frozen graph snapshots"
  python3 -u src/build_dataset.py --datasets $DATASETS 2>&1 | tee "$LOGDIR/build.log"
fi

step "content floor + split diagnostics"
python3 -u src/run_experiments.py --phase floor --datasets $DATASETS 2>&1 | tee "$LOGDIR/floor.log"
python3 -u src/run_experiments.py --phase diag  --datasets $DATASETS 2>&1 | tee "$LOGDIR/diag.log"

step "tuning grid (seed 11, 4 grid points x 7 cells x 2 datasets)"
python3 -u src/run_experiments.py --phase tune --datasets $DATASETS --workers "$WORKERS" 2>&1 | tee "$LOGDIR/tune.log"

step "final multi-seed runs (10 seeds per cell)"
python3 -u src/run_experiments.py --phase final --datasets $DATASETS --workers "$WORKERS" \
  ${SHARED:+--shared} 2>&1 | tee "$LOGDIR/final.log"

step "summarize + audit + tables"
python3 -u src/summarize.py --datasets $DATASETS 2>&1 | tee "$LOGDIR/summarize.log"
python3 -u src/audit.py     --datasets $DATASETS 2>&1 | tee "$LOGDIR/audit.log"
python3 -u tools/make_tables.py --datasets $DATASETS 2>&1 | tee "$LOGDIR/tables.log"
python3 -u tools/compare_vs_paper.py 2>&1 | tee "$LOGDIR/compare_vs_paper.md"

echo
echo "done.  results/summary/summary.md   results/audit.json   results/tables/*.tex"

# ---------------------------------------------------------------------------
# Diagnostics (NOT results -- they inspect the test split and therefore leak;
# they exist to locate protocol discrepancies, see README 5.11/5.12):
#
#   OMP_NUM_THREADS=1 python3 -u tools/diag_epoch_curve.py  --dataset npm --seed 101 --epochs 100
#   OMP_NUM_THREADS=1 python3 -u tools/diag_check_trajectory.py --dataset npm --seed 101 --model gat_dir --epochs 100
#   OMP_NUM_THREADS=1 python3 -u tools/diag_alt_candidates.py --dataset npm --seed 101
#   OMP_NUM_THREADS=1 python3 -u tools/diag_floor_pool_readings.py --dataset npm
#   OMP_NUM_THREADS=1 python3 -u tools/compare_checkpoint_rules.py --dataset npm best_val=results last=results_ckpt_last
# ---------------------------------------------------------------------------
