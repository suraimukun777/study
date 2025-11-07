"""プロンプト生成ユーティリティ"""
import numpy as np
import cv2

def extract_points_from_cam(cam, num_points=5, threshold=0.5):
    """CAMから点プロンプトを抽出"""
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

def extract_box_from_cam(cam, threshold=0.5):
    """CAMからボックスプロンプトを抽出"""
    mask = (cam > threshold).astype(np.uint8)
    
    if mask.sum() == 0:
        h, w = cam.shape
        return np.array([0, 0, w, h])
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        h, w = cam.shape
        return np.array([0, 0, w, h])
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return np.array([x, y, x + w, y + h])



