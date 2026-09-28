#!/bin/bash
set -u
cd /mnt/wisdisk-prod/agents/788f25d3d16446c8a0ae5854eb9aa856/universal_run-788f25d3d16446c8a0ae5854eb9aa856/repro
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
E4="python3 tools/ext/e4_instrument.py"
echo "=== 1. E4 zero-init maven (catch-up seed121 + rest) ==="
$E4 --datasets maven --models gat_dir ragat_sym --seeds 121 122 123 124 125 --conditions zero --workers 2
echo "=== 2. E4 zero-init npm ==="
$E4 --datasets npm --models gat_dir ragat_sym --seeds 101 102 103 104 105 --conditions zero --workers 2
echo "=== 3. E4 non-zero-init maven ==="
$E4 --datasets maven --models gat_dir ragat_sym --seeds 121 122 123 124 125 --conditions rand --gate-sigma 0.05 --workers 2
echo "=== 4. E3b text-only ablation (no graph) ==="
$E4 --datasets npm maven --models gat_dir ragat_sym --seeds 101 102 103 104 105 --conditions zero --graph-free --workers 2
$E4 --datasets maven --models gat_dir ragat_sym --seeds 121 122 123 124 125 --conditions zero --graph-free --workers 2
echo "=== 5. E3a stronger text floor ==="
python3 tools/ext/e3_text_floor.py --test-only
echo "=== 6. E4 non-zero-init npm ==="
$E4 --datasets npm --models gat_dir ragat_sym --seeds 101 102 103 104 105 --conditions rand --gate-sigma 0.05 --workers 2
echo "=== QUEUE DONE ==="
