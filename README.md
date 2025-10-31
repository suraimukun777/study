# POT-SAM2 Hybrid: 最適輸送とSAM2を融合した次世代弱教師ありセグメンテーション

[![GitHub](https://img.shields.io/badge/GitHub-suraimukun777-blue)](https://github.com/suraimukun777)
[![Status](https://img.shields.io/badge/Status-Design%20Complete-green)](https://github.com/suraimukun777/POT-SAM2-Hybrid)
[![Expected mIoU](https://img.shields.io/badge/Expected-mIoU%2080.5%25-orange)](https://github.com/suraimukun777/POT-SAM2-Hybrid)

## 概要

このプロジェクトは、**POT (Prototypical Optimal Transport)** と **SAM2** の長所を組み合わせ、S2Cの致命的な弱点を克服する革新的な手法を提案します。

### 開発の動機

S2Cの包括的分析から判明した主要な問題：
1. ❌ **CPMピークプロンプトの不正確さ** → SAM2-onlyより性能が悪い（勝率13.8%）
2. ❌ **単一プロトタイプの限界** → クラス内変動に対応できない
3. ❌ **CAM品質への過度な依存** → chair (IoU 0.316)、bicycle (IoU 0.493) で失敗
4. ❌ **クラス適応の欠如** → 全クラスで固定パラメータ

### 本手法の革新的アプローチ

✅ **POTの最適輸送** + **SAM2の強力なセグメンテーション能力** = **最強の組み合わせ**

---

## 手法の設計思想

### フェーズ1: POTベースの高品質CAM生成

**S2Cの問題を解決する要素：**

POTの**プロトタイプクラスタリング**と**最適輸送**を使用：
- ✅ **複数プロトタイプ（K=2以上）** → クラス内変動に対応
- ✅ **Similarity-aware Optimal Transport** → 重要な領域を優先
- ✅ **適応的クラスタリング** → 画像ごとに最適なプロトタイプ数を決定

**S2Cから削除するもの：**
- ❌ S2CのCPMピーク検出（不正確で有害と判明）
- ❌ 固定閾値 `th_multi = 0.5`（クラス適応なし）
- ❌ 単一ピーク戦略

**POTから活用するもの：**
- ✅ K-means クラスタリングによる複数プロトタイプ生成
- ✅ Sinkhorn最適輸送による特徴割り当て
- ✅ コサイン類似度ベースのCAM改善
- ✅ CLIP-ESの事前CAMとの統合

### フェーズ2: SAM2による精密マスク生成

**S2CのCPMプロンプトを使わない理由：**
- 実験証拠: CPM-SAM2 (IoU 0.188) << SAM2-only (IoU 0.635)
- CPMピークはSAM2の性能を**劣化させる**

**代わりに使用する戦略：**

#### **戦略A: POT-Guided SAM2 Prompts（推奨）**

POTの**プロトタイプ中心**をプロンプトとして使用：
```python
# POTで生成された各クラスのプロトタイプ中心座標
prototype_centers = extract_prototype_centers(pot_features, num_clusters=K)

# 各プロトタイプをSAM2のポイントプロンプトとして使用
for prototype_center in prototype_centers:
    sam2_masks = sam2_predictor.predict(
        point_coords=prototype_center,
        point_labels=1,  # 前景
        multimask_output=True
    )
```

**利点：**
- プロトタイプ中心は**クラスタリングで最適化**されているため、ランダムなピークより正確
- 複数プロトタイプ → **複雑な物体の複数部位**を捉えられる
- Optimal Transportで**重要度を考慮**

#### **戦略B: POT-CAM Box Prompts**

POTで生成された高品質CAMから**バウンディングボックス**を抽出：
```python
# POT CAMから信頼度の高い領域のbboxを抽出
bbox = extract_bbox_from_pot_cam(pot_cam, threshold=0.7)

# SAM2にボックスプロンプトとして渡す
sam2_masks = sam2_predictor.predict(
    box=bbox,
    multimask_output=False
)
```

**利点：**
- ポイントより**安定**（領域全体を指定）
- S2Cのピーク検出の不安定性を回避

#### **戦略C: Hybrid Prompting（最強）**

POTプロトタイプ + POT-CAM bbox + SAM2自動生成を組み合わせ：
```python
# 1. POTプロトタイプからポイントプロンプト
point_prompts = prototype_centers

# 2. POT-CAMからボックスプロンプト
box_prompts = extract_bbox_from_pot_cam(pot_cam)

# 3. SAM2で複数候補生成
masks_from_points = sam2.predict(point_coords=point_prompts)
masks_from_boxes = sam2.predict(box=box_prompts)
masks_auto = sam2.automatic_mask_generator(image)

# 4. POT-CAMとの一致度でマスクを選択・統合
best_masks = select_best_masks_by_pot_cam(
    [masks_from_points, masks_from_boxes, masks_auto],
    pot_cam
)
```

### フェーズ3: 自己洗練ループ

POTとSAM2を交互に実行して相互改善：

```
初期CAM (CLIP-ES) 
    ↓
POT改善 → プロトタイプ抽出
    ↓
SAM2マスク生成
    ↓
生成マスクでPOT特徴を再学習
    ↓
POT改善（2回目）→ より良いプロトタイプ
    ↓
SAM2最終マスク
```

---

## S2Cとの詳細比較

| 項目 | S2C | POT-SAM2 Hybrid | 改善 |
|------|-----|-----------------|------|
| **CAM生成** | ResNet38d単一プロトタイプ | POT複数プロトタイプ + OT | ✅ クラス内変動対応 |
| **ピーク検出** | 固定閾値 `th_multi=0.5` | 不使用（削除） | ✅ 不安定性を除去 |
| **プロンプト戦略** | CPMピーク（不正確） | POTプロトタイプ中心 | ✅ 最適化された位置 |
| **SAM統合** | ピークのみ | ポイント+Box+自動 | ✅ 多様なプロンプト |
| **クラス適応** | なし（全クラス同一） | あり（K適応的） | ✅ クラス特性考慮 |
| **マスク選択** | 単純な閾値 | POT-CAM類似度 | ✅ 品質評価 |
| **エンドツーエンド** | なし（2段階独立） | あり（洗練ループ） | ✅ 相互最適化 |

---

## POTとの詳細比較

| 項目 | POT | POT-SAM2 Hybrid | 改善 |
|------|-----|-----------------|------|
| **セグメンテーション** | IRN（従来手法） | SAM2（最新） | ✅ 大幅な性能向上 |
| **マスク品質** | IRNに依存 | SAM2の強力な能力 | ✅ より正確な境界 |
| **ゼロショット性能** | なし | SAM2の汎化能力 | ✅ 未知クラスにも対応可能 |
| **計算効率** | 高い | 中程度（SAM2は重い） | ⚠️ トレードオフ |

---

## アーキテクチャ詳細

### ネットワーク構成

```
入力画像
    ↓
┌─────────────────────────────────────┐
│   Stage 1: POT Feature Extraction   │
│  - ResNet50 Backbone                │
│  - Multi-scale Features             │
│  - CLIP-ES Pre-CAM Integration      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Stage 2: Prototypical Clustering  │
│  - K-means on Feature Space         │
│  - Adaptive K Selection             │
│  - Per-Class Prototypes             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Stage 3: Optimal Transport        │
│  - Sinkhorn Distance                │
│  - Similarity-aware Assignment      │
│  - Refined CAM Generation           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Stage 4: SAM2 Prompting           │
│  Strategy A: Prototype Centers      │
│  Strategy B: POT-CAM Boxes          │
│  Strategy C: Hybrid (A+B+Auto)      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Stage 5: Mask Selection           │
│  - POT-CAM Consistency Score        │
│  - Multi-Mask Voting                │
│  - Confidence Filtering             │
└─────────────────────────────────────┘
    ↓
最終セグメンテーションマスク
```

---

## 実装計画

### ディレクトリ構造

```
POT_SAM2_Hybrid/
├── README.md                      # このファイル
├── ARCHITECTURE.md                # アーキテクチャ詳細設計書
├── EXPERIMENTS.md                 # 実験計画と予想結果
├── requirements.txt               # 依存パッケージ
│
├── models/
│   ├── pot_encoder.py            # POT特徴抽出器
│   ├── prototype_module.py       # プロトタイプクラスタリング
│   ├── optimal_transport.py      # Sinkhorn OT実装
│   ├── sam2_wrapper.py           # SAM2統合ラッパー
│   └── hybrid_model.py           # 統合モデル
│
├── utils/
│   ├── cam_utils.py              # CAM処理ユーティリティ
│   ├── prompt_utils.py           # プロンプト生成
│   ├── mask_selection.py         # マスク選択ロジック
│   └── evaluation.py             # 評価関数
│
├── train/
│   ├── train_pot_stage.py        # POTステージ学習
│   ├── train_hybrid.py           # 統合学習
│   └── train_refinement.py       # 洗練ループ
│
├── inference/
│   ├── generate_pot_cam.py       # POT CAM生成
│   ├── generate_sam2_masks.py    # SAM2マスク生成
│   └── generate_final_seg.py     # 最終セグメンテーション
│
├── experiments/
│   └── (実験結果が保存される)
│
└── configs/
    ├── pot_sam2_voc.yaml         # VOC2012設定
    └── pot_sam2_coco.yaml        # MS COCO設定
```

---

## 期待される性能

### 定量的予測

#### VOC2012での予想性能

| 手法 | mIoU (CAM) | mIoU (Seg) | 改善 |
|------|-----------|-----------|------|
| S2C (元論文) | ~70% | ~74% | - |
| POT (CVPR 2025) | ~73% | **77-78%** | +3-4% |
| **POT-SAM2 Hybrid（提案）** | **75-77%** | **79-82%** | **+5-8%** |

#### クラス別改善予測

S2Cで失敗していたクラスでの改善：

| クラス | S2C IoU | 予想IoU | 改善理由 |
|--------|---------|---------|----------|
| **chair** | 0.316 | **0.55-0.65** | 複数プロトタイプで複雑構造対応 |
| **bicycle** | 0.493 | **0.65-0.75** | OTで細長い構造を正確に捉える |
| **diningtable** | 0.199 | **0.45-0.55** | SAM2で大きな平面を正確に |
| **bottle** | 0.604 | **0.70-0.75** | POT-CAMで小物体検出改善 |

**平均改善**: 現在の失敗クラスで **+20-35%の改善**を期待

### 定性的利点

1. ✅ **汎用性**: chair、bicycleなど複雑なクラスでも高精度
2. ✅ **安定性**: ピーク検出の不安定性を完全に排除
3. ✅ **適応性**: クラスごと・画像ごとに最適化
4. ✅ **拡張性**: CLIP-ESやSAM2の改良版にも対応可能
5. ✅ **解釈性**: プロトタイプを可視化できる

---

## S2Cから削除する不要な要素

### ❌ 削除リスト

1. **CPMピーク検出全体**
   - `peak_local_max` の使用
   - 固定閾値 `th_multi = 0.5`
   - 理由: 実験で有害と判明（SAM2-only以下）

2. **単一ピーク戦略**
   - 理由: 複数ピークより38%劣る

3. **SAMとの2段階独立学習**
   - 理由: エンドツーエンドの方が優れている

4. **クラス非依存のパラメータ**
   - 理由: クラス格差を生む原因

5. **ResNet38dベースのCAM**
   - 理由: POTのResNet50+OTの方が高品質

### ✅ S2Cから保持する要素

実際には、**ほとんど保持しない**（アイデアの失敗が判明したため）

唯一保持する可能性：
- Segment-Everything (SE) マップの概念（ただしPOTで代替可能）

---

## POTから活用する主要要素

### ✅ 採用リスト

1. **複数プロトタイプクラスタリング**
   - K-means による特徴クラスタリング
   - 各クラスごとにK個のプロトタイプ
   - コード: `network/resnet50_POT.py` の `get_seed_aff_x4`

2. **Sinkhorn最適輸送**
   - 特徴とプロトタイプの最適割り当て
   - コード: `Sinkhorn()` 関数

3. **Similarity-aware戦略**
   - コサイン類似度による優先順位付け
   - 重要なプロトタイプを優先的に使用

4. **CLIP-ES統合**
   - 事前CAMとの統合によるブートストラップ
   - より正確な初期化

5. **適応的閾値**
   - 画像ごとに最適な閾値を動的に決定

---

## 実装の優先順位

### Phase 1: 基本実装（1-2週間）

1. ✅ POT特徴抽出器の実装
2. ✅ プロトタイプクラスタリングモジュール
3. ✅ SAM2ラッパーの作成
4. ✅ 基本的な統合パイプライン

### Phase 2: プロンプト戦略（1週間）

1. ✅ 戦略A: Prototype Centers
2. ✅ 戦略B: POT-CAM Boxes
3. ✅ 戦略C: Hybrid Prompting

### Phase 3: 最適化と評価（2週間）

1. ✅ マスク選択ロジック
2. ✅ 洗練ループの実装
3. ✅ VOC2012での評価
4. ✅ S2C/POTとの比較実験

### Phase 4: 発展機能（オプション）

1. ⚡ MS COCOへの拡張
2. ⚡ リアルタイム推論最適化
3. ⚡ クラス適応的K選択
4. ⚡ マルチスケール統合

---

## 実験計画

### 比較実験

#### 実験1: 基本性能比較

```
S2C vs POT vs POT-SAM2 Hybrid (戦略A)
- データセット: VOC2012 val
- 評価指標: mIoU (CAM), mIoU (Seg)
- 期待: POT-SAM2 が最高性能
```

#### 実験2: プロンプト戦略の比較

```
戦略A vs 戦略B vs 戦略C vs SAM2-only
- データセット: VOC2012 val
- 評価: 各クラスごとのIoU
- 期待: 戦略C（Hybrid）が最高、SAM2-onlyを上回る
```

#### 実験3: 失敗クラスでの改善

```
chair, bicycle, diningtable, bottleでの詳細分析
- S2C vs POT-SAM2
- 質的・量的評価
- 期待: +20-35%の改善
```

#### 実験4: アブレーション研究

```
- POT vs POT+SAM2
- プロトタイプ数K=1,2,3,4,5の比較
- OTあり vs OTなし
- 洗練ループあり vs なし
```

---

## 技術的チャレンジと解決策

### チャレンジ1: 計算コスト

**問題**: SAM2は重い（POTより遅い）

**解決策**:
- バッチ処理の最適化
- SAM2の軽量版（SAM2-Tiny）を使用
- プロトタイプ数Kを適応的に調整（簡単なクラスはK=1）

### チャレンジ2: メモリ使用量

**問題**: POT + SAM2 の同時使用

**解決策**:
- ステージ別処理（POT→保存→SAM2読込）
- Mixed Precision Training
- Gradient Checkpointing

### チャレンジ3: ハイパーパラメータ調整

**問題**: 多くのパラメータ（K, OT eps, 閾値など）

**解決策**:
- POTの論文推奨値を初期値として使用
- クラス別の最適値を事前実験で決定
- AutoML的な自動調整（オプション）

---

## 予想される成果

### 学術的貢献

1. **WSSS分野の新しいベンチマーク**
   - POTとSAM2の融合による最高性能
   
2. **S2Cの失敗分析に基づく設計**
   - 大規模実験による証拠に基づく手法設計
   
3. **再利用可能なフレームワーク**
   - POTとSAMの任意の組み合わせに対応

### 実用的価値

1. **複雑なクラスでの高精度**
   - chair、furniture等で実用レベル達成
   
2. **アノテーションコスト削減**
   - より少ないラベルで高品質なセグメンテーション
   
3. **産業応用の可能性**
   - 自動運転、医療画像解析等

---

## 次のステップ

1. ✅ **このREADMEのレビュー**
2. ➡️ **アーキテクチャ詳細設計書の作成**
3. ➡️ **POTコードベースの理解と抽出**
4. ➡️ **SAM2統合インターフェースの設計**
5. ➡️ **プロトタイプ実装とテスト**
6. ➡️ **ベンチマーク実験の実施**
7. ➡️ **論文執筆**

---

## 参考文献

1. **POT**: Wang et al., "Prototypical Optimal Transport for Weakly Supervised Semantic Segmentation", CVPR 2025
2. **SAM2**: Ravi et al., "Segment Anything 2", Meta AI, 2024
3. **S2C**: Kweon & Yoon, "From SAM to CAMs", CVPR 2024
4. **CLIP-ES**: Lin et al., "CLIP-ES for Weakly Supervised Semantic Segmentation"

---

**作成日**: 2025年11月1日  
**作成者**: S2C弱点分析に基づく次世代手法設計  
**ステータス**: 設計フェーズ（実装準備中）

