from torch.utils.data import Dataset
import torch
import numpy as np
import pickle 
from src.utils.utils import *

class TrajectoryContrastiveDataset(Dataset):
    def __init__(self, npz_path, scalers_path, partition='train', max_len=55, aug_params=None):
        """
        构造函数：加载数据和scalers。
        
        Args:
            npz_path (str): 处理好的 .npz 数据文件的路径。
            scalers_path (dict): 一个包含所有scaler文件路径的字典。
            partition (str): 'train', 'val', 或 'test'。
            max_len (int): 序列的最大长度。
            aug_params (dict): 数据增强的参数，如 dropout_rate, noise_std。
        """
        all_data = np.load(npz_path, allow_pickle=True)
        self.anchors = all_data[f'X_{partition}']
        self.raw_trajs = all_data[f'traj_{partition}_raw']
        self.masks = all_data[f'mask_{partition}']
        # self.time_encodings = all_data[f'time_encoding_{partition}']
        # self.etas = all_data[f'eta_{partition}']

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
        """返回数据集的样本总数。"""
        return len(self.anchors)
    
    def __getitem__(self, idx):
        """
        获取、处理并返回第 idx 个数据样本。
        """
        # 1. 获取锚点和原始数据
        anchor_seq = torch.from_numpy(self.anchors[idx]).float()
        anchor_mask = torch.from_numpy(self.masks[idx]).bool()
        # time_encoding = torch.from_numpy(self.time_encodings[idx]).float()
        # eta = torch.tensor(self.etas[idx], dtype=torch.float32)
        raw_coords = self.raw_trajs[idx]

        # 2. 根据分区生成正样本
        if self.partition == 'train':
            # 动态生成正样本
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
            # 验证/测试模式：复制锚点
            positive_seq = anchor_seq.clone()
            positive_mask = anchor_mask.clone()

        # 3. 返回字典
        return {
            "anchor_seq": anchor_seq,
            "anchor_mask": anchor_mask,
            "positive_seq": positive_seq,
            "positive_mask": positive_mask,
            # "time_encoding": time_encoding,
            # "eta": eta
        }