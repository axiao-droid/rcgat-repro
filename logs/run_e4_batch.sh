#!/bin/bash
set -u
cd /mnt/wisdisk-prod/agents/788f25d3d16446c8a0ae5854eb9aa856/universal_run-788f25d3d16446c8a0ae5854eb9aa856/repro
echo "=== A: zero-init, maven ==="
python3 tools/ext/e4_instrument.py --datasets maven --models gat_dir ragat_sym --seeds 121 122 123 124 125 --conditions zero --workers 2
echo "=== B: non-zero-init ablation, maven ==="
python3 tools/ext/e4_instrument.py --datasets maven --models gat_dir ragat_sym --seeds 121 122 123 124 125 --conditions rand --gate-sigma 0.05 --workers 2
echo "=== A: zero-init, npm ==="
python3 tools/ext/e4_instrument.py --datasets npm --models gat_dir ragat_sym --seeds 101 102 103 104 105 --conditions zero --workers 2
echo "=== B: non-zero-init ablation, npm ==="
python3 tools/ext/e4_instrument.py --datasets npm --models gat_dir ragat_sym --seeds 101 102 103 104 105 --conditions rand --gate-sigma 0.05 --workers 2
echo "=== DONE ==="
