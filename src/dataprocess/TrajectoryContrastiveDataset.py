from torch.utils.data import Dataset
import torch
import numpy as np
import pickle 
from src.utils.utils import *

class TrajectoryContrastiveDataset(Dataset):
    def __init__(self, npz_path, scalers_path, partition='train', max_len=55, aug_params=None):
        all_data = np.load(npz_path, allow_pickle=True)
        self.anchors = all_data[f'X_{partition}']
        self.raw_trajs = all_data[f'traj_{partition}_raw']
        self.masks = all_data[f'mask_{partition}']

        self.scalers = {}
        if scalers_path is not None:
            for key, path in scalers_path.items():
                with open(path, 'rb') as f:
                    self.scalers[key] = pickle.load(f)
        
        self.partition = partition
        self.max_len = max_len

        if self.partition == 'train':
            self.aug_params = aug_params if aug_params is not None else {}
        else:
            self.aug_params = None


    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        anchor_seq = torch.from_numpy(self.anchors[idx]).float()
        anchor_mask = torch.from_numpy(self.masks[idx]).bool()
        raw_coords = self.raw_trajs[idx]

        if self.partition == 'train':
            positive_seq, positive_mask = augment_and_reencode_trajectory(
                raw_coords=raw_coords,
                scaler_ri=self.scalers['ri'],
                scaler_latlng=self.scalers['latlng'],
                scaler_len=self.scalers['len'],
                max_len=self.max_len,
                dropout_rate=self.aug_params.get('dropout_rate', 0.15),
                noise_std=self.aug_params.get('noise_std', 10.0),
                is_augment=True
            )
        else:
            positive_seq = anchor_seq.clone()
            positive_mask = anchor_mask.clone()

        return {
            "anchor_seq": anchor_seq,
            "anchor_mask": anchor_mask,
            "positive_seq": positive_seq,
            "positive_mask": positive_mask,
        }