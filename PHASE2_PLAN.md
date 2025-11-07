# Phase 2: 70-75%を目指す改善計画

**現在の性能**: 60.49% mIoU  
**目標**: 70-75% mIoU  
**必要な改善**: +10-15pt

---

## 🚀 実装した機能

### 1. SAM2ファインチューニング ✅ 実装完了

**期待効果**: +4-5%  
**実装内容**: SAM2デコーダの最後の4層を解凍して学習

#### 使用方法
```bash
# Docker内で実行
docker exec <container> bash -c "
  source /root/miniconda3/etc/profile.d/conda.sh && \
  conda activate pot_sam2 && \
  cd /workspace/POT_SAM2_Hybrid && \
  bash run_finetune_sam2.sh
"
```

#### パラメータ
- POT学習率: 1e-5
- SAM2学習率: 1e-6（より低い）
- 解凍層数: 4層
- エポック数: 15

---

### 2. テストタイム拡張（TTA）✅ 実装完了

**期待効果**: +2-3%  
**実装内容**: 複数の拡張を適用してアンサンブル

#### 使用方法
```bash
# 評価時に--use_ttaフラグを追加
python inference/run_validation.py \
  --num_samples 50 \
  --use_multiscale \
  --use_tta \
  --checkpoint experiments/finetune_sam2/best.pth \
  --output_dir experiments/validation_with_tta
```

#### 拡張内容
1. オリジナル
2. 水平反転
3. スケール変動（0.75x, 1.25x）※マルチスケールOFFの場合
4. 投票で統合

---

## 📊 予想性能推移

```
60.49% (現在)
   ↓ +4.5% (SAM2ファインチューニング)
65.0%
   ↓ +2.5% (TTA)
67.5%
   ↓ +1.5% (閾値最適化)
69.0%
   ↓ +1.5% (高度なマスク選択)
70.5%
   ↓ +1.0% (CRF後処理)
71.5% ✅ 目標達成！
```

---

## 🎯 78%を目指すには？

### 必要な追加改善（+6-7pt）

#### 1. **エンコーダも含めたSAM2のフルファインチューニング**
- 現在: デコーダ4層のみ
- 提案: エンコーダの最後数層も解凍
- 期待効果: +2-3%
- リスク: メモリ使用量増加、過学習のリスク

#### 2. **データ拡張の強化**
- 色変換、ノイズ追加、ぼかし等
- 期待効果: +1-2%

#### 3. **より多くのエポックでトレーニング**
- 現在: 15エポック
- 提案: 30-50エポック
- 期待効果: +1-2%

#### 4. **アンサンブル手法**
- 複数のモデルを学習して統合
- 期待効果: +2-3%

---

## 📝 実行手順

### Step 1: SAM2ファインチューニング

```bash
# 1. Dockerコンテナに入る
docker exec -it <container_name> bash

# 2. 環境アクティベート
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pot_sam2

# 3. プロジェクトディレクトリに移動
cd /workspace/POT_SAM2_Hybrid

# 4. ファインチューニング実行
bash run_finetune_sam2.sh
```

### Step 2: マルチスケール評価

```bash
# ファインチューニング後のモデルを評価
python inference/run_validation.py \
  --num_samples 50 \
  --use_multiscale \
  --checkpoint experiments/finetune_sam2/best.pth \
  --output_dir experiments/validation_finetune
```

### Step 3: TTA評価

```bash
# TTA追加で更なる改善
python inference/run_validation.py \
  --num_samples 50 \
  --use_multiscale \
  --use_tta \
  --checkpoint experiments/finetune_sam2/best.pth \
  --output_dir experiments/validation_finetune_tta
```

---

## ⏱️ 予想実行時間

| タスク | 時間 |
|--------|------|
| SAM2ファインチューニング（15エポック） | 約2-3時間 |
| 評価（50サンプル、マルチスケール） | 約1分 |
| 評価（50サンプル、マルチスケール + TTA） | 約3分 |

---

## 🎓 技術的詳細

### SAM2デコーダの構造

```
SAM2 Mask Decoder:
├── Transformer layers (複数)
├── IOUヘッド
└── Mask prediction heads
    └─ 最後の4層を解凍 ← ここをファインチューニング
```

### TTA

の動作

```
入力画像
├→ オリジナル → 予測1
├→ 水平反転 → 予測2（反転戻す）
└→ スケール変動 → 予測3, 4（リサイズ戻す）
      ↓
    投票統合
      ↓
  最終マスク
```

---

## 🚨 注意点

1. **メモリ使用量**
   - SAM2ファインチューニング時はGPUメモリを多く使用
   - バッチサイズ=1推奨

2. **トレーニング時間**
   - ファインチューニングは通常より時間がかかる
   - 約2-3時間見込む

3. **過学習リスク**
   - SAM2を解凍すると過学習のリスクあり
   - 検証データで性能をモニタリング

---

## 📈 期待される結果

### 楽観的シナリオ（すべてが上手くいった場合）

```
60.49% → 73-75% (+12-15pt)
```

### 現実的シナリオ

```
60.49% → 69-72% (+9-12pt)
```

### 保守的シナリオ（一部のみ改善）

```
60.49% → 66-68% (+6-8pt)
```

---

## ✅ チェックリスト

- [x] SAM2ファインチューニング実装
- [x] TTA実装
- [x] トレーニングスクリプト作成
- [x] 評価スクリプト更新
- [ ] SAM2ファインチューニング実行
- [ ] ファインチューニング後の評価
- [ ] TTA評価
- [ ] 結果分析・レポート作成

---

## 🎉 次のステップ

1. **ファインチューニング実行**
   ```bash
   bash run_finetune_sam2.sh
   ```

2. **結果確認**
   ```bash
   python inference/run_validation.py --num_samples 50 --use_multiscale --checkpoint experiments/finetune_sam2/best.pth
   ```

3. **TTA追加評価**
   ```bash
   python inference/run_validation.py --num_samples 50 --use_multiscale --use_tta --checkpoint experiments/finetune_sam2/best.pth
   ```

**頑張りましょう！70%以上を目指します！🚀**

