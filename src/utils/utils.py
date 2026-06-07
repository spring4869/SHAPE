import numpy as np
import torch
from math import atan2, sin, cos, pi

def euclidean_distance_np(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def augment_and_reencode_trajectory(
    raw_coords,                # 一条轨迹的原始坐标, shape (N, 2)
    scaler_ri,                 # 用于归一化R_i特征的scaler
    scaler_latlng,             # 用于归一化S/E经纬度的scaler
    scaler_len,                # 用于归一化总距离的scaler
    max_len,                   # 最终序列的目标长度
    dropout_rate=0.3,         # 节点丢弃率
    noise_std=20.0,             # 坐标高斯噪声的标准差 (单位：米)
    is_augment=True
):
    """
    接收一条轨迹的原始坐标，返回一套完整的、增强后的、可直接输入模型的序列。
    """
    
    # --- 1. 数据增强 (在真实坐标上操作) ---
    if np.random.rand() < 0.5:
        # 策略A: 节点随机丢弃 (不丢弃首尾)
        if len(raw_coords) > 2:
            num_to_drop = int((len(raw_coords) - 2) * dropout_rate)
            if num_to_drop > 0:
                drop_indices = np.random.choice(range(1, len(raw_coords) - 1), num_to_drop, replace=False)
                coords_aug = np.delete(raw_coords, drop_indices, axis=0)
            else:
                coords_aug = raw_coords
        else:
            coords_aug = raw_coords
    else:
        # 策略B: 高斯噪声
        noise = np.random.normal(loc=0.0, scale=noise_std, size=raw_coords.shape)
        coords_aug = raw_coords + noise

    # --- 2. 重编码 (从增强后的坐标计算新特征) ---
    new_features = []
    total_distance_aug = 0.0 
    
    # 如果增强后只剩一个点或没有点，就没有R_i
    if len(coords_aug) > 1:
        lats_aug = coords_aug[:, 0]
        lngs_aug = coords_aug[:, 1]
        cumulative_dist = 0.0
        prev_bearing = None # 用于计算方向变化率
        
        for i in range(1, len(coords_aug)):
            lat1, lng1 = lats_aug[i-1], lngs_aug[i-1]
            lat2, lng2 = lats_aug[i], lngs_aug[i]

            dlat = lat2 - lat1
            dlng = lng2 - lng1
            bearing = atan2(dlng, dlat)
            segment_dist = euclidean_distance_np(lng1, lat1, lng2, lat2)
            cumulative_dist += segment_dist

            # ===== 新增特征计算 =====
            if prev_bearing is not None:
                # 计算方向变化率，并处理角度环绕问题 (例如从 359° -> 1°)
                bearing_change = bearing - prev_bearing
                if bearing_change > pi: bearing_change -= 2 * pi
                if bearing_change < -pi: bearing_change += 2 * pi
            else:
                # 第一个路段没有“变化”
                bearing_change = 0.0
            
            # 计算曲率的简单近似：方向变化 / 距离。反映单位距离内的弯曲程度
            # 避免除以零的错误
            curvature = abs(bearing_change) / (segment_dist + 1e-9)
            
            # 更新上一个方位角
            prev_bearing = bearing
            # =========================
            
            new_features.append([
                dlat, dlng,
                sin(bearing), cos(bearing),
                cumulative_dist,
                bearing_change,
                curvature
            ])
        total_distance_aug = cumulative_dist # 增强后的总距离
    
    new_features_np = np.array(new_features)

    # --- 3. 归一化 ---
    # 归一化 R_i 特征
    if len(new_features_np) > 0:
        norm_features = scaler_ri.transform(new_features_np)
    else:
        norm_features = np.empty((0, 7))
        
    # 重新生成并归一化 S 和 E token
    s_coords_aug = coords_aug[0:1, :]
    e_coords_aug = coords_aug[-1:, :]
    s_norm = scaler_latlng.transform(s_coords_aug)[0]
    e_norm = scaler_latlng.transform(e_coords_aug)[0]
    
    # 归一化 distance token
    d_norm = scaler_len.transform(np.array([[total_distance_aug]]))[0, 0]

    # --- 4. 组装与填充 (确保结构与原模型一致) ---
    FEATURE_DIM = 7
    # CLS = np.zeros(FEATURE_DIM)
    SEP = np.ones(FEATURE_DIM)
    PAD = np.zeros(FEATURE_DIM)

    # 构造S, E, 和 Distance token，用0填充到FEATURE_DIM维
    s_token = np.pad(s_norm, (0, FEATURE_DIM - len(s_norm)))
    e_token = np.pad(e_norm, (0, FEATURE_DIM - len(e_norm)))
    distance_token = np.pad([d_norm], (0, FEATURE_DIM - 1)) 

    # 拼接成一个完整的序列
    seq_list = [
        # CLS.reshape(1, -1),
        s_token.reshape(1, -1),
        norm_features,
        e_token.reshape(1, -1),
        distance_token.reshape(1, -1), 
        SEP.reshape(1, -1)
    ]
    
    # 过滤掉空的 norm_features (当轨迹只有一个点时)
    seq = np.vstack([item for item in seq_list if item.shape[0] > 0])
    
    # 填充，最终长度为 max_len
    current_len = len(seq)
    if current_len < max_len:
        pad_len = max_len - current_len
        seq = np.vstack([seq, np.tile(PAD, (pad_len, 1))])
        mask = np.concatenate([np.ones(current_len), np.zeros(pad_len)])
    else:
        # 长度超过：截断到max_len
        seq = seq[:max_len]
        mask = np.ones(max_len)  # 全部有效（无填充）
        
    return torch.from_numpy(seq).float(), torch.from_numpy(mask).bool()