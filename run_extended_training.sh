#!/bin/bash
# POT-SAM2 Hybrid 追加トレーニングスクリプト
# 10エポックの追加トレーニングを実行

echo "======================================"
echo "POT-SAM2 Hybrid 追加トレーニング"
echo "======================================"
echo ""
echo "設定:"
echo "- エポック数: 10"
echo "- バッチサイズ: 1 (メモリ節約)"
echo "- 学習率: 1e-5 (Fine-tuning)"
echo "- 既存チェックポイントから再開"
echo ""

cd /workspace/POT_SAM2_Hybrid

# 既存のチェックポイントから再開
python train/train_e2e.py \
    --epochs 10 \
    --batch_size 1 \
    --lr 1e-5 \
    --data_list /workspace/POT/POT/data/trainaug_voc.txt \
    --output_dir experiments/e2e_training_extended \
    --resume experiments/e2e_training/best.pth \
    --freeze_sam2 \
    2>&1 | tee experiments/training_extended.log

echo ""
echo "======================================"
echo "トレーニング完了！"
echo "======================================"
echo ""
echo "結果:"
echo "- チェックポイント: experiments/e2e_training_extended/best.pth"
echo "- ログ: experiments/training_extended.log"
echo ""

