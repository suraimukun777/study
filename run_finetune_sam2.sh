#!/bin/bash

echo "======================================"
echo "POT-SAM2 Hybrid SAM2ファインチューニング"
echo "======================================"
echo ""
echo "設定:"
echo "- エポック数: 15"
echo "- バッチサイズ: 1 (メモリ節約)"
echo "- POT学習率: 1e-5"
echo "- SAM2学習率: 1e-6 (低学習率)"
echo "- SAM2解凍層数: 4"
echo ""

python train/train_finetune_sam2.py \
    --epochs 15 \
    --batch_size 1 \
    --lr_pot 1e-5 \
    --lr_sam2 1e-6 \
    --unfreeze_sam2_layers 4 \
    --data_list /workspace/POT/POT/data/trainaug_voc.txt \
    --output_dir experiments/finetune_sam2 \
    --resume experiments/e2e_training_extended/best.pth \
    --save_freq 5

echo ""
echo "======================================"
echo "ファインチューニング完了！"
echo "======================================"
echo ""
echo "結果:"
echo "- チェックポイント: experiments/finetune_sam2/best.pth"
echo ""
echo "次のステップ: 評価"
echo "docker exec <container> bash -c \"cd /workspace/POT_SAM2_Hybrid && python inference/run_validation.py --num_samples 50 --use_multiscale --checkpoint experiments/finetune_sam2/best.pth\""

