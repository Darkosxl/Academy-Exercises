#!/bin/zsh
# Train every candidate sequentially; logs in logs/, metrics in results/.
cd "$(dirname "$0")/.."
for spec in "mobilenetv3_small_100 3e-4" "mobilenetv3_large_100 3e-4" "efficientnet_b0 3e-4" \
            "resnet18 3e-4" "resnet50 3e-4" "convnext_tiny 1e-4" "deit_small_patch16_224 1e-4" "vgg11_bn 1e-4"; do
  set -- ${=spec}
  echo "=== $1 (lr $2) $(date +%T)"
  .venv/bin/python -I scripts/train.py "$1" --lr "$2" > "logs/$1.log" 2>&1 || echo "FAILED $1"
  grep -E '"test_f1"|early stop' "logs/$1.log"
done
echo ALL DONE
