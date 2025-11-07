"""
POT-SAM2 End-to-End統合モデル
POTとSAM2を統合して学習可能なモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace')

from network.resnet50_POT import CAM as POTNet
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class POTSAM2EndToEnd(nn.Module):
    """
    POT + SAM2のEnd-to-End統合モデル
    
    フロー：
    1. POTで高品質CAMを生成
    2. CAMからプロンプトを抽出
    3. SAM2でマスクを生成
    4. 損失を計算してバックプロパゲーション
    """
    
    def __init__(self, sam2_checkpoint, sam2_config='sam2_hiera_l.yaml', 
                 freeze_sam2=True, num_classes=21):
        super().__init__()
        
        self.num_classes = num_classes
        
        # POTモデル
        self.pot = POTNet(num_cls=21)  # VOC has 20 foreground classes + 1 background
        
        # SAM2モデル
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device='cuda')
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # SAM2を固定するかどうか
        if freeze_sam2:
            for param in self.sam2_model.parameters():
                param.requires_grad = False
        
        # プロンプト生成用パラメータ
        self.cam_threshold = nn.Parameter(torch.tensor(0.5))
        
        # マルチスケール推論フラグ（デフォルトはFalse）
        self.use_multiscale = False
        
    def forward(self, img, label, clip_cam, keys, return_loss=True):
        """
        Forward pass
        
        Args:
            img: 入力画像 (B, 3, H, W)
            label: クラスラベル (B, C, 1, 1)
            clip_cam: CLIP-ES CAM (C, H, W) or (B, C, H, W)
            keys: クラスキー
            return_loss: 損失を計算するか
        
        Returns:
            outputs: 出力辞書
        """
        batch_size = img.size(0)
        
        # 1. POTでCAMとプロトタイプ情報を生成
        # POTモデルに渡す
        # forward(x, label, cams_clip, keys, packs=None)
        # clip_camはnumpy配列またはTensor（POTモデル内で処理される）
        # 戻り値: (cam_class, cam_add) のタプル
        cam_class, cam_add = self.pot(img, label, clip_cam, keys, None)
        
        # バッチ次元を追加（POTは[0]で取り出しているため）
        cam_class = cam_class.unsqueeze(0)  # (1, C, H', W')
        cam_add = cam_add.unsqueeze(0)      # (1, C, H', W')
        
        # POTの内部特徴を取得（プロトタイプ抽出用）
        # 注: POTモデルの内部状態にアクセスする必要がある
        # 現在は簡易版として、CAMから直接プロトタイプ位置を推定
        self.last_pot_features = {
            'cam_class': cam_class,
            'cam_add': cam_add
        }
        
        # CAMを結合して正規化
        cam_combined = cam_class + cam_add
        cam_normalized = cam_combined / (F.adaptive_max_pool2d(cam_combined, (1, 1)) + 1e-5)
        
        # 元の画像サイズにリサイズ
        cam_resized = F.interpolate(
            cam_normalized,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 2. CAMからプロンプトを抽出してSAM2でマスクを生成
        if return_loss:
            # トレーニング時
            sam2_masks = self._generate_masks_differentiable(
                img, cam_resized, label
            )
        else:
            # 推論時（マルチスケールフラグを渡す）
            sam2_masks = self._generate_masks_inference(
                img, cam_resized, label, use_multiscale=self.use_multiscale
            )
        
        outputs = {
            'cam': cam_resized,
            'cam_class': cam_class,
            'cam_add': cam_add,
            'masks': sam2_masks
        }
        
        return outputs
    
    def _generate_masks_differentiable(self, img, cam, label):
        """
        微分可能なマスク生成（トレーニング用）
        
        注意: SAM2は現状では完全には微分可能ではないため、
        CAMベースの疑似マスクを使用するか、
        SAM2の出力をターゲットとして学習する必要がある
        """
        # 簡易実装: CAMを閾値処理してマスクとして使用
        threshold = torch.sigmoid(self.cam_threshold)
        masks = (cam > threshold).float()
        
        return masks
    
    def _generate_masks_inference(self, img, cam, label, use_multiscale=True):
        """
        推論時のマスク生成（SAM2を使用）
        
        Args:
            img: 入力画像 (B, 3, H, W)
            cam: POT-CAM (B, C, H, W)
            label: クラスラベル
            use_multiscale: マルチスケール推論を使用するか
        """
        if use_multiscale:
            return self._multiscale_inference(img, cam, label)
        else:
            return self._single_scale_inference(img, cam, label)
    
    def _single_scale_inference(self, img, cam, label):
        """
        単一スケールでの推論（従来版）
        """
        # imgはマルチスケールの場合、複数のスケールが含まれる可能性がある
        # 最初のスケールのみを使用
        if img.size(0) > cam.size(0):
            img = img[:cam.size(0)]
        
        batch_size = cam.size(0)  # camのバッチサイズを基準にする
        _, _, h, w = img.shape
        
        final_masks = []
        
        for b in range(batch_size):
            # 画像を設定（正規化を逆変換）
            img_np = img[b].cpu().numpy()  # (3, H, W)
            
            # ImageNet正規化の逆変換
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_np = img_np * std + mean  # 正規化を元に戻す
            img_np = np.clip(img_np, 0, 1)  # [0, 1]にクリップ
            
            # (3, H, W) -> (H, W, 3) そして [0, 255]に変換
            img_np = img_np.transpose(1, 2, 0)
            img_np = (img_np * 255).astype('uint8')
            
            self.sam2_predictor.set_image(img_np)
            
            # クラスごとにマスクを生成
            mask = torch.zeros((h, w), device=img.device, dtype=torch.long)
            
            # labelの次元を確認
            # labelは(21, 1, 1)の形式（バッチ次元なし）
            if label.dim() == 3:
                valid_classes = torch.where(label[:, 0, 0] > 0)[0]
            elif label.dim() == 4:
                valid_classes = torch.where(label[b, :, 0, 0] > 0)[0]
            elif label.dim() == 2:
                valid_classes = torch.where(label[:, 0] > 0)[0]
            else:
                valid_classes = torch.where(label[:] > 0)[0]
            
            for cls_idx in valid_classes:
                if cls_idx == 0:  # background skip
                    continue
                
                cls_cam = cam[b, cls_idx].cpu().numpy()
                
                # ハイブリッドプロンプト戦略: CAMの面積で判断
                cam_area = (cls_cam > 0.3).sum()
                
                if cam_area > 5000:  # 大きな物体: ボックスプロンプト
                    # 戦略B: POT-CAM Box Prompts
                    box = self._extract_box_from_cam(cls_cam, threshold=0.5, margin=5)
                    
                    if box is not None:
                        with torch.no_grad():
                            sam_masks, scores, _ = self.sam2_predictor.predict(
                                box=box,
                                multimask_output=True
                            )
                    else:
                        continue
                        
                else:  # 標準・小物体: ポイントプロンプト
                    # 戦略A: POTプロトタイプ中心からポイントプロンプト
                    num_prototypes = 2 if cam_area < 1000 else 3
                    
                    prototype_points, point_labels = self._extract_prototype_centers_from_cam(
                        cls_cam, num_prototypes=num_prototypes, threshold=0.3
                    )
                    
                    if len(prototype_points) == 0:
                        # フォールバック: 従来の点抽出
                        prototype_points, point_labels = self._extract_points_from_cam(cls_cam)
                    
                    if len(prototype_points) == 0:
                        continue
                    
                    # SAM2で予測
                    with torch.no_grad():
                        sam_masks, scores, _ = self.sam2_predictor.predict(
                            point_coords=prototype_points,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                
                # POT-CAMとの一致度で最良のマスクを選択
                best_mask = self._select_best_mask_by_cam(
                    sam_masks, scores, cls_cam
                )
                
                if best_mask is not None:
                    mask[torch.from_numpy(best_mask).to(img.device) > 0] = cls_idx
            
            final_masks.append(mask)
        
        return torch.stack(final_masks)
    
    def _extract_prototype_centers_from_cam(self, cam, num_prototypes=3, threshold=0.3):
        """
        POT-CAMからプロトタイプ中心を抽出（K-meansベース）
        README.md戦略A: POTプロトタイプ中心をプロンプトとして使用
        """
        import numpy as np
        
        mask = cam > threshold
        
        if not mask.any():
            # フォールバック: 最大値の位置
            y, x = np.unravel_index(cam.argmax(), cam.shape)
            return np.array([[x, y]]), np.array([1])
        
        y_coords, x_coords = np.where(mask)
        cam_values = cam[y_coords, x_coords]
        
        if len(y_coords) < num_prototypes:
            points = np.stack([x_coords, y_coords], axis=1)
            point_labels = np.ones(len(points), dtype=np.int32)
            return points, point_labels
        
        # CAM値でソートして均等に分割（簡易K-means）
        sorted_indices = np.argsort(cam_values)[::-1]
        chunk_size = max(1, len(sorted_indices) // num_prototypes)
        
        centers = []
        for i in range(num_prototypes):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(sorted_indices))
            if start_idx >= len(sorted_indices):
                break
            
            chunk_indices = sorted_indices[start_idx:end_idx]
            chunk_x = x_coords[chunk_indices]
            chunk_y = y_coords[chunk_indices]
            chunk_weights = cam_values[chunk_indices]
            
            # 重み付き重心
            center_x = np.average(chunk_x, weights=chunk_weights)
            center_y = np.average(chunk_y, weights=chunk_weights)
            centers.append([int(center_x), int(center_y)])
        
        centers = np.array(centers)
        point_labels = np.ones(len(centers), dtype=np.int32)
        
        return centers, point_labels
    
    def _extract_box_from_cam(self, cam, threshold=0.5, margin=5):
        """
        POT-CAMからバウンディングボックスを抽出
        README.md戦略B: POT-CAM Box Prompts
        """
        import numpy as np
        
        mask = cam > threshold
        
        if not mask.any():
            return None
        
        y_coords, x_coords = np.where(mask)
        
        x1 = max(0, x_coords.min() - margin)
        y1 = max(0, y_coords.min() - margin)
        x2 = min(cam.shape[1] - 1, x_coords.max() + margin)
        y2 = min(cam.shape[0] - 1, y_coords.max() + margin)
        
        return np.array([x1, y1, x2, y2])
    
    def _select_best_mask_by_cam(self, masks, scores, cam, iou_threshold=0.5):
        """
        POT-CAMとの一致度で最良のマスクを選択
        README.md: POT-CAM Consistency Score
        """
        import numpy as np
        
        if len(masks) == 0:
            return None
        
        best_score = -1
        best_mask = None
        
        # CAMを二値化
        cam_binary = (cam > 0.5).astype(np.float32)
        
        for mask, sam_score in zip(masks, scores):
            mask_binary = (mask > 0).astype(np.float32)
            
            # IoU計算
            intersection = (mask_binary * cam_binary).sum()
            union = ((mask_binary + cam_binary) > 0).sum()
            
            if union == 0:
                continue
            
            iou = intersection / union
            
            # SAM2のスコアとIoUを組み合わせ
            combined_score = 0.5 * sam_score + 0.5 * iou
            
            if combined_score > best_score:
                best_score = combined_score
                best_mask = mask
        
        return best_mask
    
    def _extract_points_from_cam(self, cam, num_points=5, threshold=0.5):
        """CAMから点プロンプトを抽出（従来版）"""
        import cv2
        import numpy as np
        
        mask = (cam > threshold).astype(np.uint8)
        
        if mask.sum() == 0:
            y, x = np.unravel_index(cam.argmax(), cam.shape)
            return np.array([[x, y]]), np.array([1])
        
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        points = []
        labels = []
        
        for _ in range(num_points):
            if dist_transform.max() == 0:
                break
            
            y, x = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
            points.append([x, y])
            labels.append(1)
            
            cv2.circle(dist_transform, (x, y), 20, 0, -1)
        
        if len(points) == 0:
            y, x = np.unravel_index(cam.argmax(), cam.shape)
            points.append([x, y])
            labels.append(1)
        
        return np.array(points), np.array(labels)
    
    def _multiscale_inference(self, img, cam, label, scales=[0.5, 1.0, 1.5]):
        """
        マルチスケール推論（小物体の検出改善）
        
        Args:
            img: 入力画像 (B, 3, H, W)
            cam: POT-CAM (B, C, H, W)
            label: クラスラベル
            scales: 使用するスケールのリスト
        
        Returns:
            final_mask: 統合されたマスク (B, H, W)
        """
        batch_size = cam.size(0) if cam.size(0) <= img.size(0) else img.size(0)
        _, _, h, w = img[:batch_size].shape
        
        all_masks = []
        
        for scale in scales:
            # 画像とCAMをリサイズ
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            
            img_scaled = F.interpolate(
                img[:batch_size],
                size=(scaled_h, scaled_w),
                mode='bilinear',
                align_corners=False
            )
            
            cam_scaled = F.interpolate(
                cam,
                size=(scaled_h, scaled_w),
                mode='bilinear',
                align_corners=False
            )
            
            # 単一スケールで推論
            masks_scaled = self._single_scale_inference(img_scaled, cam_scaled, label)
            
            # 元のサイズに戻す
            if masks_scaled.dim() == 2:
                masks_scaled = masks_scaled.unsqueeze(0)
            
            masks_resized = F.interpolate(
                masks_scaled.unsqueeze(1).float(),
                size=(h, w),
                mode='nearest'
            ).squeeze(1).long()
            
            all_masks.append(masks_resized)
        
        # 投票で統合
        final_masks = self._vote_masks(all_masks)
        
        return final_masks
    
    def _vote_masks(self, all_masks):
        """
        複数スケールのマスクを投票で統合
        
        Args:
            all_masks: マスクのリスト [(B, H, W), ...]
        
        Returns:
            final_mask: 統合されたマスク (B, H, W)
        """
        # all_masksをスタック: (num_scales, B, H, W)
        masks_stack = torch.stack(all_masks, dim=0)
        
        batch_size = masks_stack.size(1)
        h, w = masks_stack.size(2), masks_stack.size(3)
        num_classes = self.num_classes
        
        final_masks = []
        
        for b in range(batch_size):
            # 各ピクセルでクラスごとの投票数を計算
            votes = torch.zeros((num_classes, h, w), device=masks_stack.device)
            
            for scale_idx in range(len(all_masks)):
                mask = masks_stack[scale_idx, b]
                for cls in range(num_classes):
                    votes[cls] += (mask == cls).float()
            
            # 最多投票のクラスを選択
            final_mask = votes.argmax(dim=0)
            final_masks.append(final_mask)
        
        return torch.stack(final_masks)


class POTSAM2Loss(nn.Module):
    """
    POT-SAM2のEnd-to-End損失関数
    
    3つの損失を組み合わせ：
    1. CAM損失（クラス分類）
    2. マスク損失（セグメンテーション）
    3. 一貫性損失（CAMとマスクの一貫性）
    """
    
    def __init__(self, lambda_cam=1.0, lambda_mask=1.0, lambda_consist=0.5):
        super().__init__()
        self.lambda_cam = lambda_cam
        self.lambda_mask = lambda_mask
        self.lambda_consist = lambda_consist
    
    def forward(self, outputs, label, gt_mask=None):
        """
        損失を計算
        
        Args:
            outputs: モデル出力
            label: クラスラベル (B, C, 1, 1)
            gt_mask: Ground truth マスク (optional, B, H, W)
        
        Returns:
            total_loss: 合計損失
            loss_dict: 個別損失の辞書
        """
        cam = outputs['cam']
        masks = outputs['masks']
        
        losses = {}
        
        # 1. CAM損失（multi-label分類）
        cam_pooled = F.adaptive_avg_pool2d(cam, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 21)
        
        # labelにbackgroundを追加して形状を合わせる
        if label.dim() == 1:
            label = label.unsqueeze(0)  # (1, C)
        label_with_bg = F.pad(label, (1, 0), 'constant', 1.0)  # (B, C+1)
        
        cam_loss = F.binary_cross_entropy_with_logits(
            cam_pooled, label_with_bg.float()
        )
        losses['cam_loss'] = cam_loss * self.lambda_cam
        
        # 2. マスク損失（GTマスクがある場合）
        if gt_mask is not None:
            mask_loss = F.cross_entropy(
                masks.unsqueeze(1).float(),
                gt_mask.long(),
                ignore_index=255
            )
            losses['mask_loss'] = mask_loss * self.lambda_mask
        
        # 3. 一貫性損失（CAMとマスクの一貫性）
        # トレーニング時はCAM損失のみを使用し、一貫性損失をスキップ
        # （推論時とトレーニング時でmasksの形状が異なるため）
        try:
            # マスクから各クラスの存在を確認
            if masks.dim() == 2:
                # (H, W) -> (1, H, W)
                masks_for_loss = masks.unsqueeze(0)
            else:
                masks_for_loss = masks
            
            # バッチサイズを取得
            if masks_for_loss.dim() == 3:
                batch_size = masks_for_loss.size(0)
                mask_presence = []
                for c in range(cam.size(1)):
                    # (B, H, W) -> (B,)
                    presence = (masks_for_loss == c).float().view(batch_size, -1).sum(dim=1) > 0
                    mask_presence.append(presence)
                mask_presence = torch.stack(mask_presence, dim=1).float()  # (B, C)
            else:
                # 4D tensor の場合
                batch_size = masks_for_loss.size(0)
                mask_presence = []
                for c in range(cam.size(1)):
                    presence = (masks_for_loss == c).float().view(batch_size, -1).sum(dim=1) > 0
                    mask_presence.append(presence)
                mask_presence = torch.stack(mask_presence, dim=1).float()  # (B, C)
            
            # CAMから予測されるクラス存在
            cam_presence = (cam_pooled > 0).float()  # (B, C)
            
            # サイズを確認して一致させる
            if mask_presence.shape == cam_presence.shape:
                consistency_loss = F.binary_cross_entropy(
                    cam_presence, mask_presence
                )
                losses['consistency_loss'] = consistency_loss * self.lambda_consist
            else:
                # サイズが一致しない場合は一貫性損失をスキップ
                losses['consistency_loss'] = torch.tensor(0.0).to(cam.device)
        except Exception as e:
            # エラーが発生した場合は一貫性損失をスキップ
            losses['consistency_loss'] = torch.tensor(0.0).to(cam.device)
        
        # 合計損失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return total_loss, losses

