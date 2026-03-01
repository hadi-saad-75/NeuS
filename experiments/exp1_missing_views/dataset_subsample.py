import torch
from models.dataset import Dataset


class SubsampleDataset(Dataset):
    """
    Dataset wrapper that subsamples views deterministically.
    view_ratio ∈ (0,1], e.g. 1.0, 0.5, 0.25
    """

    def __init__(self, conf, view_ratio=1.0):
        super().__init__(conf)

        assert 0 < view_ratio <= 1.0
        self.view_ratio = view_ratio

        n_keep = int(self.n_images * view_ratio)

        # deterministic subsampling
        indices = torch.arange(self.n_images)
        step = int(1 / view_ratio) if view_ratio < 1.0 else 1
        indices = indices[::step][:n_keep]

        self.images = self.images[indices]
        self.masks = self.masks[indices]
        self.intrinsics_all = self.intrinsics_all[indices]
        self.intrinsics_all_inv = self.intrinsics_all_inv[indices]
        self.pose_all = self.pose_all[indices]
        indices_list = indices.tolist()
        self.world_mats_np = [self.world_mats_np[i] for i in indices_list]
        self.scale_mats_np = [self.scale_mats_np[i] for i in indices_list]

        self.n_images = len(indices)

        print(f"[Experiment 1] Using {self.n_images} views ({view_ratio*100:.0f}%)")
