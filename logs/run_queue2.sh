#!/bin/bash
set -u
cd /mnt/wisdisk-prod/agents/788f25d3d16446c8a0ae5854eb9aa856/universal_run-788f25d3d16446c8a0ae5854eb9aa856/repro
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
E4="python3 tools/ext/e4_instrument.py"

echo "=== 1. E8 learned pairwise structural ranker (NCN-style) ==="
python3 tools/ext/e8_pairwise_ncn.py --datasets maven --workers 2
python3 tools/ext/e8_pairwise_ncn.py --datasets npm --workers 2

echo "=== 2. E3c stratified floor / increment ==="
python3 tools/ext/e3c_stratified.py

echo "=== 2b. drop the seed-mismatched maven text-only cells (seeds 101-105) ==="
rm -f results_ext/e4_instrument/job_maven_seed10{1,2,3,4,5}_zero_textonly.json
rm -f results_ext/e4_instrument/ps_maven_seed10{1,2,3,4,5}_zero_textonly.npz

echo "=== 3. E4 aggregate (gate, gradients, ablations) ==="
python3 tools/ext/e4_report.py

echo "=== 4. E5 per-source inference ==="
python3 tools/ext/e5_per_source_inference.py

echo "=== 5. E7 citation control top-up (fifth final seed) ==="
python3 tools/ext/e7_citation_control.py --sources hepth --min-year 2002 --workers 2 --final-seeds 25
python3 tools/ext/e7_citation_control.py --sources hepph --min-year 2002 --workers 2 --final-seeds 25

echo "=== 6. E7 report ==="
python3 tools/ext/e7_report.py

echo "=== QUEUE2 DONE ==="
