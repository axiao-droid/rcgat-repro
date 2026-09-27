#!/usr/bin/env bash
# Full pipeline rerun after the split-rule change (SPLIT_RULE_VERSION=2):
# every stage is resumable, and the DONE marker at the end is what the polling
# loop looks for (a `kill -0` wait on a reaped-then-zombie child never returns).
set -u
cd /mnt/wisdisk-prod/agents/788f25d3d16446c8a0ae5854eb9aa856/universal_run-788f25d3d16446c8a0ae5854eb9aa856/repro
export OMP_NUM_THREADS=1
rm -f logs/DONE
log=logs/pipeline2.log
{
  echo "[chain] start $(date -u)"
  python3 -u src/run_experiments.py --phase floor --datasets npm maven || exit 1
  python3 -u src/run_experiments.py --phase diag --datasets npm maven || exit 1
  python3 -u src/run_experiments.py --phase tune --datasets npm maven --workers 4 || exit 1
  python3 -u src/run_experiments.py --phase final --datasets npm maven --workers 4 || exit 1
  python3 -u src/summarize.py --datasets npm maven || exit 1
  python3 -u src/audit.py --datasets npm maven || true
  python3 -u tools/make_tables.py --datasets npm maven || exit 1
  python3 -u tools/compare_vs_paper.py || exit 1
  echo "[chain] end $(date -u)"
} >> "$log" 2>&1
echo "$(date -u)" > logs/DONE
