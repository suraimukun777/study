# POT-SAM2 Hybrid 実験計画書

## 1. 実験の目的

S2Cの弱点分析に基づき、POTとSAM2を組み合わせた新手法が以下を達成できることを実証する：

1. ✅ **S2Cを大幅に上回る性能**（特に失敗クラスchair, bicycleなど）
2. ✅ **POTを上回る性能**（SAM2の統合により）
3. ✅ **SAM2-onlyを上回る性能**（POTガイダンスにより）
4. ✅ **State-of-the-art達成**（VOC2012ベンチマーク）

---

## 2. ベースライン比較

### 比較対象手法

| 手法 | 年 | CAM手法 | セグメンテーション | 公開mIoU (VOC val) |
|------|----|---------|--------------------|-------------------|
| S2C | 2024 | ResNet38d + SAM | SAM | ~69-70% |
| POT | 2025 | ResNet50 + OT | IRN | 77-78% |
| SAM2-only | 2024 | なし | SAM2自動生成 | 63.5% (実験値) |
| **POT-SAM2 (提案)** | 2025 | ResNet50 + OT | SAM2 | **79-82% (予想)** |

---

## 3. 実験セットアップ

### 3.1 データセット

#### PASCAL VOC 2012

- **Train set**: 1,464枚（画像レベルラベルのみ）
- **Train-aug set**: 10,582枚（SBD追加）
- **Val set**: 1,449枚（ピクセルレベルGT）
- **Test set**: 1,456枚（評価サーバー）

#### MS COCO 2014（追加実験）

- **Train set**: ~80K枚
- **Val set**: ~40K枚

### 3.2 評価指標

| 指標 | 説明 | 用途 |
|------|------|------|
| **mIoU (CAM)** | CAM品質評価 | 中間結果 |
| **mIoU (Seg)** | 最終セグメンテーション | 主評価指標 |
| **Per-class IoU** | クラス別性能 | 詳細分析 |
| **Pixel Accuracy** | 全体精度 | 補助指標 |
| **Boundary F-measure** | 境界精度 | SAM2の効果測定 |

### 3.3 実装詳細

```yaml
# config/pot_sam2_voc.yaml

model:
  pot:
    backbone: resnet50
    num_prototypes: 2  # K
    feature_dim: 2048
  
  ot:
    epsilon: 0.1
    max_iter: 100
    threshold: 1e-2
  
  sam2:
    checkpoint: sam2_hiera_large.pt
    config: sam2_hiera_l.yaml
    prompt_strategy: hybrid  # or 'points', 'boxes'

training:
  batch_size: 8
  max_epochs: 40
  learning_rate: 0.02
  weight_decay: 0.0005
  
  optimizer: SGD
  scheduler: poly
  
  # POTのみを学習（SAM2は固定）
  freeze_sam2: true
  
  # 洗練ループ
  refinement_epochs: [10, 20, 30]

data:
  train_list: voc12/train_aug.txt
  val_list: voc12/val.txt
  crop_size: 448
  scales: [0.5, 0.75, 1.0, 1.25, 1.5]
  
  # CLIP-ES事前CAM
  clip_cam_dir: CLIP_ES_refined_CAM/cams_71

evaluation:
  cam_threshold: [0.1, 0.2, 0.3, 0.4, 0.5]
  min_size: 50  # minimum object size
  crf: false  # CRF後処理（オプション）
```

---

## 4. 実験計画

### Experiment 1: ベースライン再現

**目的**: S2C、POT、SAM2-onlyの性能を同一環境で再現

**設定**:
- データセット: VOC2012 val
- 同一の評価コード使用
- 公平な比較のため同一前処理

**期待結果**:
```
S2C:        mIoU ~69-70%
POT:        mIoU ~77-78%
SAM2-only:  mIoU ~63.5%
```

**成功基準**: 公開値±1%以内で再現

---

### Experiment 2: POT-SAM2 基本性能

**目的**: 提案手法の基本性能を評価

**バリエーション**:

#### Variant 2-A: Strategy A (Prototype Centers)
```python
prompts = extract_prototype_centers(prototypes)
masks = sam2.predict(point_coords=prompts)
```

#### Variant 2-B: Strategy B (POT-CAM Boxes)
```python
prompts = extract_boxes_from_pot_cam(refined_cams)
masks = sam2.predict(box=prompts)
```

#### Variant 2-C: Strategy C (Hybrid)
```python
prompts = {
    'points': prototype_centers,
    'boxes': pot_cam_boxes,
    'automatic': True
}
masks = sam2.generate_all(prompts)
```

**評価**:
| Variant | 予想mIoU | 利点 | 欠点 |
|---------|---------|------|------|
| 2-A | 75-77% | 高速、シンプル | 粗い |
| 2-B | 76-78% | 安定 | 保守的 |
| 2-C | **79-82%** | 最高精度 | 計算コスト高 |

**成功基準**: 2-C が POT (77-78%) を上回る

---

### Experiment 3: S2C失敗クラスでの改善

**目的**: S2Cで失敗していたクラスでの大幅改善を実証

**対象クラス**:
- chair (S2C IoU: 0.316)
- bicycle (S2C IoU: 0.493)
- diningtable (S2C IoU: 0.199)
- bottle (S2C IoU: 0.604)

**実験設定**:
- これらのクラスを含む画像のみ抽出（~500枚）
- 詳細な定性分析（可視化）
- エラー分析

**期待結果**:

| クラス | S2C IoU | POT IoU | POT-SAM2 IoU | 改善 |
|--------|---------|---------|--------------|------|
| chair | 0.316 | 0.45? | **0.55-0.65** | **+74-106%** |
| bicycle | 0.493 | 0.58? | **0.65-0.75** | **+32-52%** |
| diningtable | 0.199 | 0.30? | **0.45-0.55** | **+126-176%** |
| bottle | 0.604 | 0.68? | **0.70-0.75** | **+16-24%** |

**成功基準**: 全ての失敗クラスで20%以上の改善

---

### Experiment 4: アブレーション研究

**目的**: 各コンポーネントの貢献度を定量化

#### 4-A: プロトタイプ数Kの影響

```
K=1 (single prototype, like S2C):   X% mIoU
K=2 (POT default):                   Y% mIoU
K=3:                                 Z% mIoU
K=4:                                 W% mIoU
K=5:                                 V% mIoU
```

**予想**: K=2またはK=3が最適

#### 4-B: Optimal Transportの効果

```
Without OT (simple assignment):      A% mIoU
With OT (Sinkhorn):                  B% mIoU

Improvement: (B-A) = +3-5% 予想
```

#### 4-C: CLIP-ES統合の効果

```
Without CLIP-ES pre-CAM:             C% mIoU
With CLIP-ES integration:            D% mIoU

Improvement: (D-C) = +2-3% 予想
```

#### 4-D: 洗練ループの効果

```
Single pass (no refinement):         E% mIoU
With refinement loop (2 iterations): F% mIoU

Improvement: (F-E) = +1-2% 予想
```

#### 4-E: SAM2 vs IRN

```
POT + IRN (original):                77-78% mIoU
POT + SAM2 (ours):                   79-82% mIoU

Improvement from SAM2: +2-4%
```

**Full Ablation Table**:

| Config | K | OT | CLIP-ES | Refinement | Segmentor | mIoU | Δ |
|--------|---|----|---------|-----------|-----------| -----|---|
| S2C | 1 | ❌ | ❌ | ❌ | SAM | 69-70% | baseline |
| POT | 2 | ✅ | ✅ | ❌ | IRN | 77-78% | +8-9% |
| Ours-Min | 1 | ❌ | ❌ | ❌ | SAM2 | ~72% | +2% |
| Ours-NoOT | 2 | ❌ | ✅ | ❌ | SAM2 | ~76% | +6% |
| Ours-NoRef | 2 | ✅ | ✅ | ❌ | SAM2 | ~78% | +8% |
| **Ours-Full** | **2** | ✅ | ✅ | ✅ | **SAM2** | **79-82%** | **+10-12%** |

---

### Experiment 5: 計算効率とスケーラビリティ

**目的**: 実用性の評価

#### 5-A: 推論時間

測定: VOC val 1,449枚の平均処理時間

```
S2C:         ~300ms/image (GPU)
POT:         ~150ms/image (GPU)
SAM2-only:   ~500ms/image (GPU)
POT-SAM2-A:  ~700ms/image (GPU) 予想
POT-SAM2-C:  ~1200ms/image (GPU) 予想
```

**分析**:
- Strategy A: 許容範囲内
- Strategy C: 精度重視の場合のみ

#### 5-B: メモリ使用量

```
S2C:        ~3GB VRAM
POT:        ~2GB VRAM
POT-SAM2:   ~6GB VRAM 予想
```

**対策**: Batch size調整、SAM2-Tinyの使用

#### 5-C: スケーラビリティ（MS COCO）

大規模データセットでの性能維持を確認

```
VOC (21 classes):   79-82% mIoU
COCO (80 classes):  X% mIoU (要実験)
```

---

### Experiment 6: 定性評価

**目的**: 視覚的品質の検証

#### 6-A: セグメンテーション品質

S2C、POT、POT-SAM2の出力を並べて比較

**評価基準**:
- ✅ 物体境界の正確さ（SAM2の強み）
- ✅ 複雑構造の捉え方（椅子の脚など）
- ✅ 小物体の検出（ボトルなど）
- ✅ 大きな平面の完全性（テーブル）

#### 6-B: プロトタイプ可視化

各クラスのK個のプロトタイプを可視化

**分析**:
- プロトタイプが物体の異なる部位を捉えているか？
- クラス内変動に対応できているか？

#### 6-C: Failure Case分析

POT-SAM2でも失敗するケースを特定

**予想される失敗**:
- 極端な遮蔽
- 非常に小さい物体（<20px）
- 珍しい視点・ポーズ

**対策案**:
- マルチスケール強化
- Data augmentation
- 適応的プロトタイプ数

---

## 5. 予想される実験結果サマリー

### VOC2012 Val Set

| 手法 | mIoU | chair | bicycle | diningtable | bottle | 推論速度 |
|------|------|-------|---------|-------------|--------|---------|
| S2C | 69.7% | 31.6% | 49.3% | 19.9% | 60.4% | 300ms |
| POT | 77.5% | 45% | 58% | 30% | 68% | 150ms |
| SAM2-only | 63.5% | 25% | 40% | 15% | 55% | 500ms |
| POT-SAM2-A | **76%** | **52%** | **63%** | **42%** | **72%** | **700ms** |
| POT-SAM2-B | **78%** | **58%** | **68%** | **48%** | **74%** | **900ms** |
| **POT-SAM2-C** | **80.5%** | **63%** | **72%** | **52%** | **76%** | **1200ms** |

**ハイライト**:
- 🏆 **chair**: 31.6% → **63%** (+99%改善)
- 🏆 **diningtable**: 19.9% → **52%** (+161%改善)
- 🏆 **Overall**: 69.7% → **80.5%** (+10.8pt)

### クラス別詳細（POT-SAM2-C）

| Class | S2C | POT | POT-SAM2-C | Δ from S2C |
|-------|-----|-----|------------|------------|
| aeroplane | 90.8% | 92% | **93%** | +2.2% |
| bicycle | 49.3% | 58% | **72%** | **+22.7%** |
| bird | 61.5% | 68% | **73%** | +11.5% |
| boat | 76.2% | 78% | **80%** | +3.8% |
| bottle | 60.4% | 68% | **76%** | **+15.6%** |
| bus | 86.5% | 88% | **90%** | +3.5% |
| car | 69.5% | 75% | **78%** | +8.5% |
| cat | 86.3% | 88% | **91%** | +4.7% |
| chair | 31.6% | 45% | **63%** | **+31.4%** |
| cow | 81.8% | 84% | **87%** | +5.2% |
| diningtable | 19.9% | 30% | **52%** | **+32.1%** |
| dog | 80.6% | 83% | **86%** | +5.4% |
| horse | 88.7% | 90% | **92%** | +3.3% |
| motorbike | 86.8% | 88% | **90%** | +3.2% |
| person | 69.6% | 74% | **77%** | +7.4% |
| pottedplant | 56.7% | 63% | **68%** | +11.3% |
| sheep | 67.2% | 72% | **76%** | +8.8% |
| sofa | 68.0% | 73% | **77%** | +9.0% |
| train | 92.6% | 93% | **94%** | +1.4% |
| tvmonitor | 67.3% | 72% | **75%** | +7.7% |
| **Mean** | **69.7%** | **77.5%** | **80.5%** | **+10.8%** |

---

## 6. 実験スケジュール

### Phase 1: セットアップ（Week 1-2）

- ✅ 環境構築
- ✅ POTコードベース理解
- ✅ SAM2統合テスト
- ✅ データ準備

### Phase 2: 基本実装（Week 3-4）

- ✅ POT feature extractor
- ✅ Prototype clustering
- ✅ Optimal transport
- ✅ SAM2 wrapper

### Phase 3: ベースライン実験（Week 5）

- ✅ Exp 1: ベースライン再現
- ✅ 性能確認

### Phase 4: 主要実験（Week 6-8）

- ✅ Exp 2: POT-SAM2基本性能
- ✅ Exp 3: 失敗クラス改善
- ✅ Exp 4: アブレーション

### Phase 5: 追加実験（Week 9-10）

- ✅ Exp 5: 計算効率
- ✅ Exp 6: 定性評価
- ✅ MS COCO実験（オプション）

### Phase 6: 論文執筆（Week 11-12）

- ✅ 結果整理
- ✅ 図表作成
- ✅ 論文ドラフト

**総期間**: 約3ヶ月

---

## 7. リスクと対策

### リスク1: 期待性能に届かない

**兆候**: mIoU < 78%

**原因分析**:
- POT学習が不十分
- SAM2プロンプトが不適切
- マスク選択ロジックの問題

**対策**:
- POT学習のハイパーパラメータ調整
- プロンプト戦略の見直し
- より sophisticated なマスク選択

### リスク2: 計算コストが高すぎる

**兆候**: 推論時間 > 2秒/image

**対策**:
- SAM2-Tinyの使用
- プロトタイプ数Kの削減
- Strategy Aの採用（Cの代わり）

### リスク3: 実装の複雑さ

**兆候**: バグが多発、デバッグ困難

**対策**:
- モジュール単位のユニットテスト
- 段階的統合
- 可視化ツールの充実

### リスク4: 再現性の問題

**対策**:
- Random seedの固定
- 詳細なログ記録
- Checkpoint保存

---

## 8. 成功基準

### Minimum Viable Product (MVP)

✅ VOC2012 val で mIoU > 78%  
✅ chairクラスで IoU > 0.50  
✅ S2Cより全てのクラスで改善

### Target Performance

✅ VOC2012 val で mIoU > 80%  
✅ 失敗クラス（chair, bicycle, diningtable, bottle）で20%以上改善  
✅ State-of-the-artと比肩する性能

### Stretch Goals

✅ VOC2012 val で mIoU > 82%  
✅ MS COCO でも competitive  
✅ 推論時間 < 1秒/image  
✅ トップカンファレンス（CVPR/ICCV/ECCV）採択

---

## 9. 次のアクション

1. ✅ **このドキュメントのレビューと承認**
2. ➡️ **実装開始**（`models/`から）
3. ➡️ **Exp 1 実行**（ベースライン再現）
4. ➡️ **Exp 2-A 実行**（最もシンプルなvariant）
5. ➡️ **結果に基づいて方針調整**

---

**作成日**: 2025年11月1日  
**ステータス**: 実験計画承認待ち  
**推定期間**: 3ヶ月  
**期待成果**: SOTA WSSS手法、トップカンファレンス論文

