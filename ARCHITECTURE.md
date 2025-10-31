# POT-SAM2 Hybrid アーキテクチャ詳細設計書

## 1. システム全体構成

```
┌──────────────────────────────────────────────────────────────┐
│                     Input: Image + Labels                     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              Module 1: POT Feature Extraction                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ResNet50 Backbone (POTベース)                          │  │
│  │ - Stage 1-4: Multi-scale features                      │  │
│  │ - Side connections: 特徴拡張                           │  │
│  │ - CLIP-ES CAM統合                                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│         Module 2: Prototypical Clustering (POT Core)          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ For each class c:                                      │  │
│  │   1. Extract class features F_c                        │  │
│  │   2. K-means → K prototypes {p_c1, ..., p_cK}        │  │
│  │   3. Compute cosine similarity                         │  │
│  │   4. Generate proto-CAM for each prototype            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│         Module 3: Optimal Transport (Sinkhorn)                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Feature-to-Prototype Assignment:                       │  │
│  │   - Distance: 1 - cosine_similarity                    │  │
│  │   - Constraint: ∑T = 1 (行), ∑T = 1 (列)             │  │
│  │   - Sinkhorn iteration (max_iter=100)                  │  │
│  │   - Output: Transport matrix T [M×N×C]                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│            Module 4: Refined CAM Generation                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ POT-CAM = Mean(Proto-CAMs) weighted by T              │  │
│  │ Normalization: POT-CAM / max(POT-CAM)                 │  │
│  │ Multi-scale fusion                                     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│          Module 5: SAM2 Prompt Generation (3 Strategies)      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Strategy A: Prototype Centers                          │  │
│  │   → K points per class from prototype positions       │  │
│  │                                                        │  │
│  │ Strategy B: POT-CAM Bounding Boxes                    │  │
│  │   → Extract bbox from high-confidence regions          │  │
│  │                                                        │  │
│  │ Strategy C: Hybrid (A + B + SAM2 Auto)                │  │
│  │   → Combine all prompting strategies                   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              Module 6: SAM2 Mask Generation                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ For each prompt type:                                  │  │
│  │   sam2_predictor.set_image(image)                      │  │
│  │   masks = sam2_predictor.predict(prompts)              │  │
│  │   → Multiple mask candidates per class                 │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│           Module 7: Mask Selection & Fusion                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ For each class c:                                      │  │
│  │   1. Compute consistency: IoU(mask, POT-CAM)          │  │
│  │   2. Compute confidence: SAM2 scores                   │  │
│  │   3. Select best mask per class                        │  │
│  │   4. Fuse overlapping masks (NMS)                      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                   Output: Final Masks                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. モジュール詳細設計

### Module 1: POT Feature Extraction

**入力**: 
- Image: `[B, 3, H, W]`
- Labels: `[B, C]` (image-level)
- CLIP-ES CAM (オプション): `[B, C, H', W']`

**出力**:
- Features: `[B, 2048, H/16, W/16]` (stage4)
- Multi-scale features: List of `[B, C_i, H_i, W_i]`

**実装の核心**:

```python
class POTFeatureExtractor(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.resnet50 = resnet50(pretrained=True, 
                                 strides=(2,2,2,1))
        
        # POT特有のside connections
        self.side1 = nn.Conv2d(256+3, 128, 1)
        self.side2 = nn.Conv2d(512+3, 128, 1)
        self.side3 = nn.Conv2d(1024+3, 256, 1)
        self.side4 = nn.Conv2d(2048+3, 256, 1)
        
        self.classifier = nn.Conv2d(2048, num_classes-1, 1)
        
    def forward(self, img, clip_es_cam=None):
        # Multi-scale feature extraction
        x0 = self.stage0(img)
        x1 = self.stage1(x0)  # [B, 256, H/4, W/4]
        x2 = self.stage2(x1)  # [B, 512, H/8, W/8]
        x3 = self.stage3(x2)  # [B, 1024, H/16, W/16]
        x4 = self.stage4(x3)  # [B, 2048, H/16, W/16]
        
        # CLIP-ES CAMとの統合（あれば）
        if clip_es_cam is not None:
            x4 = self.integrate_clip_cam(x4, clip_es_cam)
        
        # Classification branch
        class_logits = self.classifier(x4)
        
        return {
            'features': x4,
            'multi_scale': [x1, x2, x3, x4],
            'class_logits': class_logits
        }
```

---

### Module 2: Prototypical Clustering

**アルゴリズム**:

```
For each class c with label=1:
  1. Extract features: F_c = features[class_mask==c]
  2. Normalize: F_c = F_c / ||F_c||
  3. K-means clustering:
      cluster_ids, centers = kmeans(F_c, K, metric='cosine')
  4. Store: prototypes[c] = centers [K, D]
```

**重要なパラメータ**:
- `K`: プロトタイプ数（デフォルト=2、クラス適応可能）
- `metric`: 'cosine'（角度距離）
- `device`: GPU加速

**実装**:

```python
class PrototypicalClusteringModule(nn.Module):
    def __init__(self, num_clusters=2, feature_dim=2048):
        super().__init__()
        self.K = num_clusters
        self.feature_dim = feature_dim
        
    def forward(self, features, labels, initial_cam):
        """
        Args:
            features: [B, D, H, W]
            labels: [B, C]
            initial_cam: [B, C, H, W] (from classifier)
        
        Returns:
            prototypes: [B, C, K, D]
            proto_cams: [B, C, K, H, W]
        """
        B, D, H, W = features.shape
        C = labels.shape[1]
        
        prototypes = torch.zeros(B, C, self.K, D).cuda()
        proto_cams = torch.zeros(B, C, self.K, H, W).cuda()
        
        for b in range(B):
            for c in range(C):
                if labels[b, c] == 0:
                    continue
                
                # Class seedsの生成
                class_mask = (initial_cam[b, c] > 0.5).float()
                class_features = features[b] * class_mask.unsqueeze(0)
                
                # Flatten and filter
                flat_features = class_features.view(D, -1).t()  # [HW, D]
                valid_mask = class_mask.view(-1) > 0
                valid_features = flat_features[valid_mask]  # [N_valid, D]
                
                if valid_features.size(0) < self.K:
                    # 不十分な特徴数の場合はスキップ
                    continue
                
                # K-means clustering
                cluster_ids, cluster_centers = kmeans(
                    X=valid_features,
                    num_clusters=self.K,
                    distance='cosine',
                    device='cuda'
                )
                
                prototypes[b, c] = cluster_centers
                
                # 各プロトタイプのCAM生成
                for k in range(self.K):
                    # Cosine similarity map
                    proto = cluster_centers[k].unsqueeze(0)  # [1, D]
                    feat_2d = features[b].view(D, -1).t()  # [HW, D]
                    
                    sim = F.cosine_similarity(
                        feat_2d.unsqueeze(1), 
                        proto.unsqueeze(0), 
                        dim=2
                    )  # [HW, 1]
                    
                    proto_cams[b, c, k] = sim.view(H, W)
        
        return prototypes, proto_cams
```

**クラス適応的K選択**（オプション）:

```python
def adaptive_K_selection(self, class_features, max_K=5):
    """
    Silhouette scoreを使ってクラスごとに最適なKを選択
    """
    best_K = 2
    best_score = -1
    
    for K in range(2, max_K+1):
        cluster_ids, _ = kmeans(class_features, K, 'cosine')
        score = silhouette_score(
            class_features.cpu().numpy(), 
            cluster_ids.cpu().numpy()
        )
        if score > best_score:
            best_score = score
            best_K = K
    
    return best_K
```

---

### Module 3: Optimal Transport (Sinkhorn)

**数学的定式化**:

```
min_{T} <T, C>  (Cost: C = 1 - cosine_similarity)
s.t.  T1 = μ    (行制約: 各特徴が割り当てられる)
      T^T1 = ν  (列制約: 各プロトタイプの割り当て)
      T ≥ 0

Sinkhorn iteration:
  K = exp(-C/ε)
  repeat:
    r = μ / (K * c)
    c = ν / (K^T * r)
  until convergence
  
  T = diag(r) * K * diag(c)
```

**実装**:

```python
class OptimalTransportModule(nn.Module):
    def __init__(self, epsilon=0.1, max_iter=100, threshold=1e-2):
        super().__init__()
        self.eps = epsilon
        self.max_iter = max_iter
        self.thresh = threshold
        
    def sinkhorn(self, cost_matrix, mu, nu):
        """
        Sinkhorn algorithm for optimal transport
        
        Args:
            cost_matrix: [N, M] distance matrix
            mu: [N] source distribution
            nu: [M] target distribution
        
        Returns:
            T: [N, M] transport plan
        """
        K = torch.exp(-cost_matrix / self.eps)
        
        r = torch.ones_like(mu)
        c = torch.ones_like(nu)
        
        for i in range(self.max_iter):
            r_prev = r
            r = mu / (K @ c + 1e-8)
            c = nu / (K.t() @ r + 1e-8)
            
            err = (r - r_prev).abs().mean()
            if err < self.thresh:
                break
        
        T = torch.diag(r) @ K @ torch.diag(c)
        return T
    
    def forward(self, features, prototypes, proto_cams):
        """
        Compute optimal transport between features and prototypes
        
        Args:
            features: [B, D, H, W]
            prototypes: [B, C, K, D]
            proto_cams: [B, C, K, H, W]
        
        Returns:
            transport_maps: [B, C, K, H, W]
            refined_cams: [B, C, H, W]
        """
        B, D, H, W = features.shape
        C = prototypes.shape[1]
        K = prototypes.shape[2]
        
        transport_maps = torch.zeros(B, C, K, H, W).cuda()
        refined_cams = torch.zeros(B, C, H, W).cuda()
        
        for b in range(B):
            # Flatten features
            feat_flat = features[b].view(D, -1).t()  # [HW, D]
            feat_norm = feat_flat / (feat_flat.norm(dim=1, keepdim=True) + 1e-5)
            
            for c in range(C):
                # Prototypes for this class
                proto_c = prototypes[b, c]  # [K, D]
                proto_norm = proto_c / (proto_c.norm(dim=1, keepdim=True) + 1e-5)
                
                # Cost matrix: 1 - cosine similarity
                sim_matrix = feat_norm @ proto_norm.t()  # [HW, K]
                cost_matrix = 1.0 - sim_matrix
                
                # Distributions
                mu = torch.ones(H*W).cuda() / (H*W)  # uniform
                nu = torch.ones(K).cuda() / K  # uniform
                
                # Compute optimal transport
                T = self.sinkhorn(cost_matrix, mu, nu)  # [HW, K]
                
                # Reshape transport map
                T_reshaped = T.view(H, W, K).permute(2, 0, 1)  # [K, H, W]
                transport_maps[b, c] = T_reshaped
                
                # Refined CAM: weighted sum of proto_cams
                refined_cam = (proto_cams[b, c] * T_reshaped).sum(dim=0)
                refined_cams[b, c] = refined_cam
        
        # Normalize
        refined_cams = refined_cams / (refined_cams.amax(dim=(2,3), keepdim=True) + 1e-5)
        
        return transport_maps, refined_cams
```

**Similarity-aware改良**:

POT論文では、重要なプロトタイプを優先的に割り当てる戦略を使用：

```python
# 重要度スコアの計算
importance = proto_cams.mean(dim=(3,4))  # [B, C, K]

# 重要度に基づいてnuを調整
nu_weighted = importance[b, c]
nu_weighted = nu_weighted / nu_weighted.sum()

# OTにweighted distributionを使用
T = self.sinkhorn(cost_matrix, mu, nu_weighted)
```

---

### Module 4: Refined CAM Generation

**多段階洗練**:

```python
def refine_pot_cam(self, proto_cams, transport_maps, multi_scale_features):
    """
    Multi-scale fusion and refinement
    """
    # 1. Transport-weighted aggregation
    base_cam = (proto_cams * transport_maps).sum(dim=2)  # [B, C, H, W]
    
    # 2. Multi-scale consistency
    refined_cams = []
    for scale_idx, feat_scale in enumerate(multi_scale_features):
        # Up/downscale base_cam to match feature scale
        cam_scaled = F.interpolate(base_cam, size=feat_scale.shape[2:], mode='bilinear')
        
        # Refine with feature similarity
        refined = self.refine_with_features(cam_scaled, feat_scale)
        refined_cams.append(refined)
    
    # 3. Fusion
    final_cam = self.fuse_multi_scale(refined_cams)
    
    return final_cam
```

---

### Module 5: SAM2 Prompt Generation

**Strategy A: Prototype Centers**

```python
class PrototypeCenterPrompts:
    def extract_prompts(self, prototypes, refined_cams, labels):
        """
        プロトタイプの空間的位置をプロンプトとして抽出
        
        Args:
            prototypes: [B, C, K, D]
            refined_cams: [B, C, H, W]
            labels: [B, C]
        
        Returns:
            point_prompts: List of [N_points, 2] (x, y)
            point_labels: List of [N_points] (1=fg, 0=bg)
            class_ids: List of class indices
        """
        prompts_per_image = []
        
        for b in range(len(labels)):
            prompts = []
            prompt_labels = []
            prompt_classes = []
            
            for c in range(labels.shape[1]):
                if labels[b, c] == 0:
                    continue
                
                # 各プロトタイプのCAMから最大値位置を取得
                for k in range(prototypes.shape[2]):
                    cam = refined_cams[b, c]  # [H, W]
                    
                    # 最大値の位置
                    max_idx = cam.argmax()
                    y = max_idx // cam.shape[1]
                    x = max_idx % cam.shape[1]
                    
                    prompts.append([x.item(), y.item()])
                    prompt_labels.append(1)  # foreground
                    prompt_classes.append(c)
            
            prompts_per_image.append({
                'points': np.array(prompts),
                'labels': np.array(prompt_labels),
                'classes': np.array(prompt_classes)
            })
        
        return prompts_per_image
```

**Strategy B: POT-CAM Bounding Boxes**

```python
class POTCAMBoxPrompts:
    def extract_prompts(self, refined_cams, labels, threshold=0.7):
        """
        POT-CAMから高信頼度領域のbboxを抽出
        """
        prompts_per_image = []
        
        for b in range(len(labels)):
            boxes = []
            box_classes = []
            
            for c in range(labels.shape[1]):
                if labels[b, c] == 0:
                    continue
                
                cam = refined_cams[b, c]  # [H, W]
                
                # 閾値処理
                binary_mask = (cam > threshold).cpu().numpy().astype(np.uint8)
                
                if binary_mask.sum() == 0:
                    continue
                
                # Bounding boxを抽出
                y_indices, x_indices = np.where(binary_mask)
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                
                boxes.append([x_min, y_min, x_max, y_max])
                box_classes.append(c)
            
            prompts_per_image.append({
                'boxes': np.array(boxes),
                'classes': np.array(box_classes)
            })
        
        return prompts_per_image
```

**Strategy C: Hybrid Prompting**

```python
class HybridPromptGenerator:
    def __init__(self):
        self.strategy_a = PrototypeCenterPrompts()
        self.strategy_b = POTCAMBoxPrompts()
        
    def generate(self, prototypes, refined_cams, labels, image):
        """
        3つのソースからプロンプトを生成：
        1. Prototype centers (Strategy A)
        2. POT-CAM boxes (Strategy B)
        3. SAM2 automatic masks
        """
        # Strategy A
        point_prompts = self.strategy_a.extract_prompts(
            prototypes, refined_cams, labels
        )
        
        # Strategy B
        box_prompts = self.strategy_b.extract_prompts(
            refined_cams, labels
        )
        
        # Strategy C: SAM2 automatic (プロンプトなし)
        # これは後でSAM2 moduleで実行
        
        return {
            'point_prompts': point_prompts,
            'box_prompts': box_prompts,
            'use_automatic': True  # SAM2自動生成も使用
        }
```

---

### Module 6: SAM2 Mask Generation

```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2MaskGenerator:
    def __init__(self, sam2_checkpoint, model_cfg):
        self.sam2 = build_sam2(model_cfg, sam2_checkpoint)
        self.predictor = SAM2ImagePredictor(self.sam2)
        
    def generate_masks(self, image, prompts, refined_cams):
        """
        Args:
            image: [H, W, 3] numpy array
            prompts: dict with 'point_prompts', 'box_prompts'
            refined_cams: [C, H, W] for consistency check
        
        Returns:
            masks: dict {class_id: [H, W] binary mask}
        """
        # SAM2に画像を設定
        self.predictor.set_image(image)
        
        all_masks = {}
        
        # Point prompts (Strategy A)
        if 'point_prompts' in prompts:
            for prompt in prompts['point_prompts']:
                masks_from_points = self.predictor.predict(
                    point_coords=prompt['points'],
                    point_labels=prompt['labels'],
                    multimask_output=True
                )
                all_masks.setdefault('points', []).append(masks_from_points)
        
        # Box prompts (Strategy B)
        if 'box_prompts' in prompts:
            for prompt in prompts['box_prompts']:
                masks_from_boxes = self.predictor.predict(
                    box=prompt['boxes'],
                    multimask_output=False
                )
                all_masks.setdefault('boxes', []).append(masks_from_boxes)
        
        # Automatic masks (Strategy C part)
        if prompts.get('use_automatic', False):
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            mask_generator = SAM2AutomaticMaskGenerator(self.sam2)
            auto_masks = mask_generator.generate(image)
            all_masks['automatic'] = auto_masks
        
        return all_masks
```

---

### Module 7: Mask Selection & Fusion

**マスク選択基準**:

1. **POT-CAM Consistency**: IoU(mask, POT-CAM)
2. **SAM2 Confidence**: SAM2が出力する信頼度スコア
3. **Size Constraint**: 適切なサイズ範囲内

```python
class MaskSelectionModule:
    def select_best_masks(self, all_masks, refined_cams, labels):
        """
        各クラスごとに最良のマスクを選択
        
        Scoring function:
          score = α * consistency + β * confidence + γ * size_penalty
          
        where:
          consistency = IoU(mask, POT-CAM)
          confidence = SAM2 score
          size_penalty = gaussian(size, mean=optimal_size)
        """
        final_masks = {}
        
        for class_id in range(labels.shape[0]):
            if labels[class_id] == 0:
                continue
            
            pot_cam = refined_cams[class_id]  # [H, W]
            candidates = self.gather_candidates(all_masks, class_id)
            
            best_score = -1
            best_mask = None
            
            for candidate in candidates:
                mask = candidate['mask']
                sam2_conf = candidate.get('confidence', 0.5)
                
                # Consistency score
                consistency = self.compute_consistency(mask, pot_cam)
                
                # Size penalty
                size_penalty = self.compute_size_penalty(mask)
                
                # Total score
                score = (
                    0.5 * consistency + 
                    0.3 * sam2_conf + 
                    0.2 * size_penalty
                )
                
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask is not None:
                final_masks[class_id] = best_mask
        
        return final_masks
    
    def compute_consistency(self, mask, pot_cam, threshold=0.5):
        """IoU between binary mask and thresholded POT-CAM"""
        pot_cam_binary = (pot_cam > threshold).float()
        mask_float = torch.from_numpy(mask).float().cuda()
        
        intersection = (mask_float * pot_cam_binary).sum()
        union = ((mask_float + pot_cam_binary) > 0).sum()
        
        iou = intersection / (union + 1e-5)
        return iou.item()
    
    def fuse_overlapping_masks(self, masks):
        """NMSで重複マスクを統合"""
        # クラス間でIoU > 0.3のマスクをマージ
        # 優先度: POT-CAM consistencyが高い方を残す
        pass
```

---

## 3. Loss Functions

### Training Loss (POT Stage)

```python
class POTTrainingLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # classification loss
        self.beta = beta    # OT consistency loss
        self.gamma = gamma  # prototype diversity loss
        
    def forward(self, outputs, labels):
        # 1. Classification loss
        cls_loss = F.multilabel_soft_margin_loss(
            outputs['class_logits'],
            labels
        )
        
        # 2. OT consistency loss (POT論文)
        ot_loss = self.compute_ot_consistency(
            outputs['refined_cams'],
            outputs['proto_cams'],
            outputs['transport_maps']
        )
        
        # 3. Prototype diversity loss
        div_loss = self.compute_prototype_diversity(
            outputs['prototypes']
        )
        
        total_loss = (
            self.alpha * cls_loss +
            self.beta * ot_loss +
            self.gamma * div_loss
        )
        
        return total_loss, {
            'cls': cls_loss,
            'ot': ot_loss,
            'div': div_loss
        }
```

---

## 4. Training Pipeline

```python
def train_pot_sam2_hybrid(config):
    # 1. Initialize models
    pot_model = POTFeatureExtractor(num_classes=21)
    clustering_module = PrototypicalClusteringModule(num_clusters=2)
    ot_module = OptimalTransportModule()
    sam2_generator = SAM2MaskGenerator(config.sam2_ckpt, config.sam2_cfg)
    
    # 2. Training loop
    for epoch in range(config.max_epochs):
        for batch in train_loader:
            # Stage 1: POT training
            pot_outputs = pot_model(batch['image'], batch['clip_cam'])
            prototypes, proto_cams = clustering_module(
                pot_outputs['features'],
                batch['labels'],
                pot_outputs['class_logits']
            )
            transport_maps, refined_cams = ot_module(
                pot_outputs['features'],
                prototypes,
                proto_cams
            )
            
            # Compute loss and backward
            loss = pot_loss_fn(refined_cams, batch['labels'])
            loss.backward()
            optimizer.step()
        
        # Stage 2: Generate SAM2 masks (evaluation phase)
        if epoch % config.sam2_interval == 0:
            with torch.no_grad():
                for val_batch in val_loader:
                    # Generate prompts
                    prompts = generate_prompts(prototypes, refined_cams)
                    
                    # SAM2 inference
                    masks = sam2_generator.generate_masks(
                        val_batch['image'],
                        prompts,
                        refined_cams
                    )
                    
                    # Evaluate
                    miou = evaluate_masks(masks, val_batch['gt_masks'])
                    print(f"Epoch {epoch}, mIoU: {miou:.3f}")
```

---

## 5. 推論パイプライン

```python
def inference(image, labels, models):
    """
    Single image inference
    
    Args:
        image: [H, W, 3] numpy array
        labels: [C] binary labels
        models: dict of all modules
    
    Returns:
        final_masks: dict {class_id: binary mask}
        visualization: annotated image
    """
    # 1. POT feature extraction
    with torch.no_grad():
        pot_out = models['pot'](
            torch.from_numpy(image).cuda(),
            clip_cam=None
        )
        
        # 2. Clustering
        prototypes, proto_cams = models['clustering'](
            pot_out['features'],
            labels,
            pot_out['class_logits']
        )
        
        # 3. Optimal Transport
        transport_maps, refined_cams = models['ot'](
            pot_out['features'],
            prototypes,
            proto_cams
        )
        
        # 4. Generate prompts
        prompts = models['prompt_gen'].generate(
            prototypes, refined_cams, labels, image
        )
        
        # 5. SAM2 mask generation
        all_masks = models['sam2'].generate_masks(
            image, prompts, refined_cams
        )
        
        # 6. Mask selection
        final_masks = models['selector'].select_best_masks(
            all_masks, refined_cams, labels
        )
    
    return final_masks
```

---

## 6. 計算量とメモリ

### 理論的計算量

| Module | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| POT Feature | O(HWD) | O(BD × HW) |
| Clustering | O(NKD × iter) | O(CKD) |
| OT (Sinkhorn) | O(MN × iter) | O(CMN) |
| SAM2 | O(SAM2の複雑度) | O(SAM2のメモリ) |

**推定推論時間**（単一画像、GPU）:
- POT Feature: ~50ms
- Clustering: ~20ms
- OT: ~30ms
- SAM2: ~500ms（複数プロンプト）
- **合計**: ~600ms/image

**メモリ使用量**:
- POT: ~2GB
- SAM2: ~4GB
- **合計**: ~6GB VRAM

---

**次のステップ**: `EXPERIMENTS.md`で詳細な実験計画を記述

