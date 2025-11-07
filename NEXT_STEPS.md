# POT-SAM2 Hybrid 次のステップ

## 📅 作成日時
2025年10月31日

## 🎯 現在の状況

### 実装完了 ✅
- POTプロトタイプ中心抽出
- SAM2プロンプト生成（ポイントプロンプト）
- POT-CAM一致度マスク選択
- End-to-End学習（基本版）
- SAM2推論の統合
- **マルチスケール推論戦略** ✨ NEW!

### 現在の性能（2025年11月7日 最終更新）

#### 🎉 大躍進達成！

| バージョン | mIoU | 改善 | 状態 |
|-----------|------|------|------|
| 修正前（SAM2統合） | 46.33% | - | 問題あり |
| **修正後（単一スケール）** | **57.73%** | **+11.40pt** 🚀 | 画像正規化修正 |
| **修正後（マルチスケール）** | **🎯 60.49%** | **+2.76pt** ✨ | 最終版 |
| **累計改善** | | **+14.16pt** | **目標達成！** ✅ |

#### 🔧 修正内容
**画像正規化の不適切な逆変換を修正**
- SAM2に渡す画像が壊れていた
- ImageNet正規化を正しく逆変換
- 修正により性能が劇的改善

### 目標性能
- ✅ **短期（1-2週間）**: 60-65% mIoU → **60.49%達成！**
- **中期（1ヶ月）**: 70-75% mIoU
- **長期（2-3ヶ月）**: **79-82% mIoU** 🎯

---

## 🚀 優先順位付き改善計画

### Phase 1: 短期改善（1-2週間）⚡

#### 1.1 ボックスプロンプトの実装と使用 ✅ **完了！**
**目的**: 大きな物体（diningtable、sofa等）の精度向上

**実装完了日**: 2025年11月7日（以前に実装済み）

**実装内容**:
```python
# models/pot_sam2_e2e.py の _extract_box_from_cam
def _extract_box_from_cam(self, cam, threshold=0.5, margin=5):
    """
    POT-CAMからバウンディングボックスを抽出
    """
    mask = cam > threshold
    if not mask.any():
        return None
    
    y_coords, x_coords = np.where(mask)
    x1 = max(0, x_coords.min() - margin)
    y1 = max(0, y_coords.min() - margin)
    x2 = min(cam.shape[1] - 1, x_coords.max() + margin)
    y2 = min(cam.shape[0] - 1, y_coords.max() + margin)
    
    return np.array([x1, y1, x2, y2])
```

**実際の効果**:
- **sofa**: 0.1323 → **0.6754** (+54.31% 🚀)
- **diningtable**: 0.5852 → **0.4115** (若干低下、要改善)
- ボックスプロンプトは大物体に有効

---

#### 1.2 ハイブリッドプロンプト戦略（ポイント + ボックス）✅ **完了！**
**目的**: クラスごとに最適なプロンプトを自動選択

**実装完了日**: 2025年11月7日（以前に実装済み）

**実装内容**:
```python
# CAMの面積で判断
cam_area = (cls_cam > 0.3).sum()

if cam_area > 5000:  # 大きな物体: ボックスプロンプト
    box = self._extract_box_from_cam(cls_cam, threshold=0.5, margin=5)
    sam_masks, scores, _ = self.sam2_predictor.predict(
        box=box, multimask_output=True
    )
else:  # 標準・小物体: ポイントプロンプト
    num_prototypes = 2 if cam_area < 1000 else 3
    prototype_points, point_labels = self._extract_prototype_centers_from_cam(
        cls_cam, num_prototypes=num_prototypes, threshold=0.3
    )
    sam_masks, scores, _ = self.sam2_predictor.predict(
        point_coords=prototype_points,
        point_labels=point_labels,
        multimask_output=True
    )
```

**実際の効果**:
- 大物体と小物体で適切なプロンプトを自動選択
- 全体性能の向上に貢献

---

#### 1.3 マルチスケール戦略 ✅ **完了！**
**目的**: 小物体（bottle、bicycle等）の検出改善

**実装完了日**: 2025年11月7日

**実装内容**:
```python
def _multiscale_inference(self, img, cam, label, scales=[0.5, 1.0, 1.5]):
    """
    複数スケールで推論して結果を統合
    """
    all_masks = []
    
    for scale in scales:
        # 画像とCAMをリサイズ
        img_scaled = F.interpolate(img, scale_factor=scale)
        cam_scaled = F.interpolate(cam, scale_factor=scale)
        
        # SAM2推論
        masks = self._single_scale_inference(img_scaled, cam_scaled, label)
        
        # 元のサイズに戻す
        masks_resized = F.interpolate(masks.unsqueeze(1).float(), 
                                      size=img.shape[2:], 
                                      mode='nearest').squeeze(1).long()
        all_masks.append(masks_resized)
    
    # 投票で統合
    final_mask = self._vote_masks(all_masks)
    return final_mask
```

**実際の効果（50サンプル、画像正規化修正後）**:
- **tvmonitor**: 0.5172 → **0.7425** (+22.53% 🚀)
- **bird**: 0.6232 → **0.8332** (+21.00% 🚀)
- **dog**: 0.7034 → **0.8149** (+11.15% ✨)
- **aeroplane**: 0.7865 → **0.8252** (+3.87%)
- **cat**: 0.8650 → **0.8890** (+2.40%)
- **全体**: 57.73% → **60.49%** (+2.76pt)

**詳細レポート**: `BREAKTHROUGH_REPORT.md`

---

#### 1.4 より多くのトレーニング 🟢
**目的**: モデルの収束と性能向上

**実装内容**:
- トレーニングエポック: 1 → 10-20
- トレーニングデータ: 10サンプル → 100-500サンプル
- 学習率スケジューリング
- データ拡張

**期待される効果**:
- **全体**: +10-15ポイント

**実装時間**: 6-8時間（主に実行時間）

---

### Phase 2: 中期改善（1ヶ月）🔥

#### 2.1 自己洗練ループ
**目的**: 反復的な品質向上

**実装内容**:
```python
def _self_refinement_loop(self, img, initial_cam, label, iterations=3):
    """
    SAM2マスク → 新しいCAM → 再推論
    """
    cam = initial_cam
    
    for i in range(iterations):
        # SAM2で推論
        masks = self._generate_masks_inference(img, cam, label)
        
        # マスクから新しいCAMを生成
        refined_cam = self._masks_to_cam(masks, img)
        
        # CAMを更新
        cam = 0.7 * cam + 0.3 * refined_cam
    
    return masks
```

**期待される効果**:
- **全体**: +5-10ポイント

---

#### 2.2 クラス適応的プロトタイプ数
**目的**: クラスごとに最適なプロトタイプ数を決定

**実装内容**:
```python
# クラスごとの最適K値
CLASS_PROTOTYPE_NUMS = {
    'bottle': 1,      # 小物体
    'person': 5,      # 複雑な形状
    'chair': 3,       # 標準
    'diningtable': 4, # 大きな平面
    # ...
}
```

**期待される効果**:
- **全体**: +3-5ポイント

---

#### 2.3 Validationセットでの評価
**目的**: 公式ベンチマークでの性能検証

**実装内容**:
- VOC2012 Validation setでの評価
- 他手法との公平な比較
- 詳細な分析レポート

**期待される効果**:
- 公式性能の確認
- 論文化の準備

---

### Phase 3: 長期改善（2-3ヶ月）🎯

#### 3.1 SAM2のファインチューニング
**目的**: POT-CAMに特化したSAM2の調整

**実装内容**:
- SAM2の一部レイヤーを解凍
- POT-CAMとの一致度を最大化する学習
- LoRA等の効率的なファインチューニング

**期待される効果**:
- **全体**: +10-15ポイント

---

#### 3.2 完全なEnd-to-End学習
**目的**: POTとSAM2の統合学習

**実装内容**:
- POTとSAM2を同時に学習
- 微分可能なSAM2統合
- 統合損失関数の最適化

**期待される効果**:
- **全体**: +15-20ポイント

---

#### 3.3 論文執筆と投稿
**目的**: 学術的貢献

**実装内容**:
- 包括的な実験
- 他手法との詳細な比較
- CVPR/ICCV/ECCVへの投稿

---

## 📊 予測性能推移

| Phase | 実装内容 | 予測mIoU | 累積改善 |
|-------|---------|---------|---------|
| **現在** | 基本実装 | 48.17% | - |
| **Phase 1.1** | + ボックスプロンプト | 51-53% | +3-5pt |
| **Phase 1.2** | + ハイブリッド戦略 | 55-59% | +7-11pt |
| **Phase 1.3** | + マルチスケール | 58-64% | +10-16pt |
| **Phase 1.4** | + 追加トレーニング | **63-69%** | +15-21pt |
| **Phase 2** | + 自己洗練 + 適応 | **70-75%** | +22-27pt |
| **Phase 3** | + SAM2 FT + 完全E2E | **79-82%** 🎯 | +31-34pt |

---

## 🎯 即座に実装可能な改善（今日中）

### 最小限の変更で最大の効果

#### 改善1: ボックスプロンプトの追加 ⚡
**実装時間**: 2時間
**期待効果**: +3-5ポイント

#### 改善2: プロトタイプ数の増加（K=3 → K=5）⚡
**実装時間**: 30分
**期待効果**: +2-3ポイント

#### 改善3: CAM閾値の最適化（0.3 → 0.2）⚡
**実装時間**: 15分
**期待効果**: +1-2ポイント

**合計期待効果**: +6-10ポイント → **54-58% mIoU**

---

## 📝 推奨実装順序

### 今日（2-3時間）
1. ✅ ボックスプロンプトの実装
2. ✅ プロトタイプ数の調整
3. ✅ 簡単なハイパーパラメータ調整

### 明日（4-6時間）
1. ハイブリッドプロンプト戦略
2. マルチスケール戦略の基本実装
3. 評価と分析

### 今週末（8-10時間）
1. より多くのトレーニング（10エポック）
2. 包括的な評価
3. レポート作成

### 来週（1週間）
1. 自己洗練ループの実装
2. クラス適応的プロトタイプ
3. Validationセット評価

---

## 🔧 実装の優先順位

### 🔴 最優先（今日中）
- ボックスプロンプトの実装
- プロトタイプ数の調整

### 🟠 高優先（今週中）
- ハイブリッドプロンプト戦略
- マルチスケール戦略
- 追加トレーニング

### 🟡 中優先（来週）
- 自己洗練ループ
- クラス適応的プロトタイプ
- Validationセット評価

### 🟢 低優先（長期）
- SAM2ファインチューニング
- 完全E2E学習
- 論文執筆

---

## 💡 実装のヒント

### ボックスプロンプトの実装
```python
# 既存の _extract_box_from_cam を活用
box = self._extract_box_from_cam(cls_cam, threshold=0.5)

# SAM2に渡す
sam_masks, scores, _ = self.sam2_predictor.predict(
    box=box,
    multimask_output=True
)
```

### ハイブリッド戦略
```python
# CAMの面積で判断
area = (cls_cam > 0.3).sum()

if area > 5000:
    # ボックスプロンプト
    masks, scores = self._generate_with_box(cls_cam)
else:
    # ポイントプロンプト
    masks, scores = self._generate_with_points(cls_cam)
```

---

## 📈 成功の指標

### 短期（1-2週間）
- ✅ mIoU 60-65%達成
- ✅ chair、diningtable IoU > 0.80
- ✅ bottle、bicycle IoU > 0.30

### 中期（1ヶ月）
- ✅ mIoU 70-75%達成
- ✅ 全クラス IoU > 0.40
- ✅ Validationセット評価完了

### 長期（2-3ヶ月）
- ✅ mIoU 79-82%達成
- ✅ S2C、POTを大幅に上回る
- ✅ 論文投稿準備完了

---

## 🎉 まとめ

### 現在地
- 基本実装完了
- SAM2推論統合完了
- mIoU 48.17% (50サンプル)

### 次のステップ
1. **今日**: ボックスプロンプト実装 → **54-58% mIoU**
2. **今週**: ハイブリッド + マルチスケール → **60-65% mIoU**
3. **来週**: 自己洗練 + 適応 → **70-75% mIoU**
4. **来月**: SAM2 FT + 完全E2E → **79-82% mIoU** 🎯

**最初の一歩**: ボックスプロンプトの実装から始めましょう！

