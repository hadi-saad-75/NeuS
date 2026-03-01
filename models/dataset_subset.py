# models/dataset_subset.py

import torch
import numpy as np
from models.dataset import Dataset


class SubsampledDataset(Dataset):
    def __init__(self, conf, view_fraction=1.0):
        super().__init__(conf)

        assert 0 < view_fraction <= 1.0

        total_views = self.n_images
        keep_views = int(total_views * view_fraction)

        # Deterministic subsampling
        indices = np.linspace(0, total_views - 1, keep_views, dtype=int)

        self.images = self.images[indices]
        self.masks = self.masks[indices]
        self.intrinsics_all = self.intrinsics_all[indices]
        self.intrinsics_all_inv = self.intrinsics_all_inv[indices]
        self.pose_all = self.pose_all[indices]
        self.world_mats_np = [self.world_mats_np[i] for i in indices]
        self.scale_mats_np = [self.scale_mats_np[i] for i in indices]

        self.n_images = keep_views

        print(f"Using {keep_views}/{total_views} views ({view_fraction*100:.1f}%)")
