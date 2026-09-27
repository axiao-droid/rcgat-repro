#!/usr/bin/env bash
# Reinstall the CPU-only torch stack.  download.pytorch.org was measured at
# ~0.07 MB/s from the test host; mirrors.aliyun.com serves the same wheel at
# ~21 MB/s, so the wheel is pulled directly and the pure-python dependencies
# come from the Tsinghua PyPI mirror.
set -u
cd "$(dirname "$0")/.."
WHEEL=/tmp/torch-2.9.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
URL="https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
rm -f logs/pip_torch.done

echo "=== [1/3] download wheel $(date -u) ==="
[ -s "$WHEEL" ] || curl -fL --retry 3 -o "$WHEEL" "$URL" || { echo "download failed"; echo "$(date -u) exit=download" > logs/pip_torch.done; exit 1; }
ls -l "$WHEEL"

echo "=== [2/3] install torch $(date -u) ==="
python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "$WHEEL" || { echo "$(date -u) exit=torch" > logs/pip_torch.done; exit 1; }

echo "=== [3/3] install torch_geometric $(date -u) ==="
python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "torch_geometric==2.8.0.post1" || { echo "$(date -u) exit=pyg" > logs/pip_torch.done; exit 1; }

echo "=== verify $(date -u) ==="
python3 -c "import torch, torch_geometric, sklearn, numpy; print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'sklearn', sklearn.__version__)"
echo "$(date -u) exit=$?" > logs/pip_torch.done
