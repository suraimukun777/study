# POT-SAM2 Hybrid クイックスタートガイド

## 📋 前提条件

- NVIDIA GPU (12GB以上推奨)
- CUDA 11.0以上
- Python 3.8以上

---

## 🚀 セットアップ (5分)

### Step 1: 環境作成

```bash
# Conda環境作成
conda create -n pot_sam2 python=3.8
conda activate pot_sam2

# 基本パッケージインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# プロジェクトの依存関係
cd ~/POT_SAM2_Hybrid
pip install -r requirements.txt
```

### Step 2: SAM2インストール

```bash
# SAM2リポジトリをクローン
cd ~
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# SAM2モデルダウンロード
mkdir -p ~/POT_SAM2_Hybrid/pretrained
cd ~/POT_SAM2_Hybrid/pretrained
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```

### Step 3: POTコードベース準備

```bash
# POTから必要なモジュールをコピー
cd ~/POT_SAM2_Hybrid
cp -r ~/POT/POT/network/resnet50_POT.py models/
cp -r ~/POT/POT/data ./
```

### Step 4: データセット準備

```bash
# VOC2012へのシンボリックリンク
ln -s ~/POT/VOCdevkit ~/POT_SAM2_Hybrid/VOCdevkit

# CLIP-ES CAMへのリンク（あれば）
ln -s ~/POT/CLIP_ES_refined_CAM ~/POT_SAM2_Hybrid/CLIP_ES_refined_CAM
```

---

## ⚡ 動作確認 (2分)

### テスト1: POT Feature Extraction

```bash
cd ~/POT_SAM2_Hybrid
python -c "
from models.pot_encoder import POTFeatureExtractor
import torch

model = POTFeatureExtractor(num_classes=21)
print('✅ POT model loaded successfully')

# ダミー入力でテスト
x = torch.randn(1, 3, 448, 448)
out = model(x)
print(f'✅ Output shape: {out[\"features\"].shape}')
"
```

### テスト2: SAM2 Integration

```bash
python -c "
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = 'pretrained/sam2_hiera_large.pt'
model_cfg = 'sam2_hiera_l.yaml'
sam2 = build_sam2(model_cfg, sam2_checkpoint)
predictor = SAM2ImagePredictor(sam2)
print('✅ SAM2 loaded successfully')
"
```

---

## 🏃 実験実行

### Experiment 1: ベースライン（S2C再現）

```bash
cd ~/POT_SAM2_Hybrid

# S2Cの評価スクリプト実行
cd ~/S2C
python evaluation.py \
    --name 250530_s2c_S2C_train \
    --task cam \
    --dict_dir dict

# 結果を記録
# Expected: mIoU ~69-70%
```

### Experiment 2: POT-SAM2 (Strategy A)

```bash
cd ~/POT_SAM2_Hybrid

# トレーニング
python train/train_pot_stage.py \
    --config configs/pot_sam2_voc.yaml \
    --num_epochs 40

# CAM生成
python inference/generate_pot_cam.py \
    --checkpoint experiments/pot_stage/best_model.pth \
    --data_list data/val_voc.txt \
    --output_dir experiments/pot_cams

# SAM2マスク生成（Strategy A）
python inference/generate_sam2_masks.py \
    --pot_cam_dir experiments/pot_cams \
    --strategy points \
    --output_dir experiments/pot_sam2_masks_a

# 評価
python utils/evaluation.py \
    --pred_dir experiments/pot_sam2_masks_a \
    --gt_dir VOCdevkit/VOC2012/SegmentationClass \
    --num_classes 21

# Expected: mIoU ~76-78%
```

### Experiment 3: Full Pipeline (Strategy C)

```bash
# Full Hybridパイプライン
python inference/generate_final_seg.py \
    --pot_model experiments/pot_stage/best_model.pth \
    --sam2_checkpoint pretrained/sam2_hiera_large.pt \
    --strategy hybrid \
    --data_list data/val_voc.txt \
    --output_dir experiments/pot_sam2_final

# 評価
python utils/evaluation.py \
    --pred_dir experiments/pot_sam2_final \
    --gt_dir VOCdevkit/VOC2012/SegmentationClass

# Expected: mIoU ~79-82%
```

---

## 📊 結果の可視化

```bash
# TensorBoardで学習進捗確認
tensorboard --logdir experiments/logs

# 定性結果の可視化
python utils/visualize_results.py \
    --pred_dir experiments/pot_sam2_final \
    --gt_dir VOCdevkit/VOC2012/SegmentationClass \
    --img_dir VOCdevkit/VOC2012/JPEGImages \
    --output_dir experiments/visualizations
```

---

## 🐛 トラブルシューティング

### 問題: CUDA out of memory

**解決策**:
```yaml
# configs/pot_sam2_voc.yaml を編集
training:
  batch_size: 4  # 8から4に削減
```

または、SAM2-Tinyを使用:
```yaml
model:
  sam2:
    checkpoint: pretrained/sam2_hiera_tiny.pt
    config: sam2_hiera_t.yaml
```

### 問題: SAM2が見つからない

```bash
# SAM2のインストールを確認
python -c "import sam2; print(sam2.__file__)"

# 見つからない場合は再インストール
cd ~/segment-anything-2
pip install -e . --force-reinstall
```

### 問題: CLIP-ES CAMがない

CLIP-ES CAMは必須ではありません。なしで実行する場合:

```yaml
# configs/pot_sam2_voc.yaml
data:
  clip_cam_dir: null  # CAMなしで実行
```

---

## 📈 期待される性能

| Experiment | Method | mIoU | Time |
|-----------|--------|------|------|
| Exp 1 | S2C (baseline) | ~69-70% | - |
| Exp 2-A | POT-SAM2 (points) | **~76-78%** | ~30 min |
| Exp 2-B | POT-SAM2 (boxes) | **~78-80%** | ~40 min |
| Exp 3 | POT-SAM2 (hybrid) | **~79-82%** | ~60 min |

*時間はVOC val (1,449枚) の推論時間、単一RTX 3090使用*

---

## 📝 次のステップ

1. ✅ QuickStartを完了
2. ➡️ Exp 1でベースライン確認
3. ➡️ Exp 2-Aで基本動作確認
4. ➡️ ハイパーパラメータ調整
5. ➡️ Full実験（Exp 3）
6. ➡️ 論文執筆

---

## 🔗 参考リンク

- [S2C Project](~/S2C/)
- [POT Project](~/POT/)
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [詳細ドキュメント](README.md)
- [アーキテクチャ](ARCHITECTURE.md)
- [実験計画](EXPERIMENTS.md)

---

**作成日**: 2025年11月1日  
**最終更新**: 2025年11月1日  
**推定セットアップ時間**: ~10分  
**推定実験時間**: 1-2日（すべての実験）

