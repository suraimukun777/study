#!/bin/bash
# POT-SAM2 Hybrid 実験実行スクリプト

set -e

echo "=========================================="
echo "  POT-SAM2 Hybrid Experiment"
echo "=========================================="
echo ""

# Conda環境を有効化
source /root/miniconda3/bin/activate pot_sam2

# 作業ディレクトリ
cd /workspace/POT_SAM2_Hybrid

# Strategy選択
STRATEGY=${1:-points}
echo "Prompting Strategy: $STRATEGY"
echo ""

# 全パイプライン実行
python inference/generate_final_seg.py --strategy $STRATEGY

echo ""
echo "=========================================="
echo "  Experiment Completed!"
echo "=========================================="



