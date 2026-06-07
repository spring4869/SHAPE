import numpy as np
import torch
from math import atan2, sin, cos, pi

def euclidean_distance_np(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def augment_and_reencode_trajectory(
    raw_coords,                
    scaler_ri,                
    scaler_latlng,             
    scaler_len,                
    max_len,                   
    dropout_rate=0.3,         
    noise_std=20.0,             
    is_augment=True
):
    if np.random.rand() < 0.5:
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
        noise = np.random.normal(loc=0.0, scale=noise_std, size=raw_coords.shape)
        coords_aug = raw_coords + noise

    new_features = []
    total_distance_aug = 0.0 
    
    if len(coords_aug) > 1:
        lats_aug = coords_aug[:, 0]
        lngs_aug = coords_aug[:, 1]
        cumulative_dist = 0.0
        prev_bearing = None 
        
        for i in range(1, len(coords_aug)):
            lat1, lng1 = lats_aug[i-1], lngs_aug[i-1]
            lat2, lng2 = lats_aug[i], lngs_aug[i]

            dlat = lat2 - lat1
            dlng = lng2 - lng1
            bearing = atan2(dlng, dlat)
            segment_dist = euclidean_distance_np(lng1, lat1, lng2, lat2)
            cumulative_dist += segment_dist

            if prev_bearing is not None:
                bearing_change = bearing - prev_bearing
                if bearing_change > pi: bearing_change -= 2 * pi
                if bearing_change < -pi: bearing_change += 2 * pi
            else:
                bearing_change = 0.0
            
            curvature = abs(bearing_change) / (segment_dist + 1e-9)
            
            prev_bearing = bearing
            
            new_features.append([
                dlat, dlng,
                sin(bearing), cos(bearing),
                cumulative_dist,
                bearing_change,
                curvature
            ])
        total_distance_aug = cumulative_dist 
    
    new_features_np = np.array(new_features)

    if len(new_features_np) > 0:
        norm_features = scaler_ri.transform(new_features_np)
    else:
        norm_features = np.empty((0, 7))
        
    s_coords_aug = coords_aug[0:1, :]
    e_coords_aug = coords_aug[-1:, :]
    s_norm = scaler_latlng.transform(s_coords_aug)[0]
    e_norm = scaler_latlng.transform(e_coords_aug)[0]
    
    d_norm = scaler_len.transform(np.array([[total_distance_aug]]))[0, 0]

    FEATURE_DIM = 7
    SEP = np.ones(FEATURE_DIM)
    PAD = np.zeros(FEATURE_DIM)

    s_token = np.pad(s_norm, (0, FEATURE_DIM - len(s_norm)))
    e_token = np.pad(e_norm, (0, FEATURE_DIM - len(e_norm)))
    distance_token = np.pad([d_norm], (0, FEATURE_DIM - 1)) 

    seq_list = [
        s_token.reshape(1, -1),
        norm_features,
        e_token.reshape(1, -1),
        distance_token.reshape(1, -1), 
        SEP.reshape(1, -1)
    ]
    
    seq = np.vstack([item for item in seq_list if item.shape[0] > 0])
    
    current_len = len(seq)
    if current_len < max_len:
        pad_len = max_len - current_len
        seq = np.vstack([seq, np.tile(PAD, (pad_len, 1))])
        mask = np.concatenate([np.ones(current_len), np.zeros(pad_len)])
    else:
        seq = seq[:max_len]
        mask = np.ones(max_len) 
        
    return torch.from_numpy(seq).float(), torch.from_numpy(mask).bool()