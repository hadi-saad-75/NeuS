import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# ------------------------------------------------------------
# Borrowed from IDR
# ------------------------------------------------------------
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K, R, t = out[0], out[1], out[2]

    K = K / K[2, 2]

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


# ============================================================
# Dataset
# ============================================================
class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print("Load data: Begin")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf

        self.data_dir = conf.get_string("data_dir")
        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_outside_sphere = conf.get_bool("camera_outside_sphere", default=True)
        self.scale_mat_scale = conf.get_float("scale_mat_scale", default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        self.images_lis = sorted(glob(os.path.join(self.data_dir, "image/*.png")))
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, "mask/*.png")))

        self.n_images = len(self.images_lis)

        self.images_np = np.stack([cv.imread(im) for im in self.images_lis]) / 256.0
        self.masks_np = np.stack([cv.imread(im) for im in self.masks_lis]) / 256.0

        self.world_mats_np = [
            camera_dict[f"world_mat_{idx}"].astype(np.float32)
            for idx in range(self.n_images)
        ]

        self.scale_mats_np = [
            camera_dict[f"scale_mat_{idx}"].astype(np.float32)
            for idx in range(self.n_images)
        ]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = (world_mat @ scale_mat)[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # --------------------------------------------------------
        # Tensors
        # --------------------------------------------------------
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.pose_all = torch.stack(self.pose_all).to(self.device)

        self.focal = self.intrinsics_all[0][0, 0]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # --------------------------------------------------------
        # Bounding box for mesh extraction
        # --------------------------------------------------------
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])

        object_scale_mat = np.load(
            os.path.join(self.data_dir, self.object_cameras_name)
        )["scale_mat_0"]

        object_bbox_min = (
            np.linalg.inv(self.scale_mats_np[0])
            @ object_scale_mat
            @ object_bbox_min[:, None]
        )
        object_bbox_max = (
            np.linalg.inv(self.scale_mats_np[0])
            @ object_scale_mat
            @ object_bbox_max[:, None]
        )

        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print("Load data: End")

    # ============================================================
    # Generate full image rays
    # ============================================================
    def gen_rays_at(self, img_idx, resolution_level=1):
        device = self.device
        l = resolution_level

        tx = torch.linspace(0, self.W - 1, self.W // l, device=device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=device)

        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)],
            dim=-1,
        )

        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, None, :3, :3],
            p[:, :, :, None],
        ).squeeze()

        rays_v = p / torch.linalg.norm(p, dim=-1, keepdim=True)
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, None, :3, :3],
            rays_v[:, :, :, None],
        ).squeeze()

        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    # ============================================================
    # Generate random training rays
    # ============================================================
    def gen_random_rays_at(self, img_idx, batch_size):
        device = self.device

        pixels_x = torch.randint(0, self.W, (batch_size,), device=device)
        pixels_y = torch.randint(0, self.H, (batch_size,), device=device)

        # Image tensors live on CPU
        color = self.images[img_idx][(pixels_y.cpu(), pixels_x.cpu())]
        mask = self.masks[img_idx][(pixels_y.cpu(), pixels_x.cpu())]

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)],
            dim=-1,
        ).float()

        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3],
            p[:, :, None],
        ).squeeze()

        rays_v = p / torch.linalg.norm(p, dim=-1, keepdim=True)

        rays_v = torch.matmul(
            self.pose_all[img_idx, None, :3, :3],
            rays_v[:, :, None],
        ).squeeze()

        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)

        # Final tensor returned on CUDA
        return torch.cat(
            [rays_o, rays_v, color.to(device), mask[:, :1].to(device)],
            dim=-1,
        )

    # ============================================================
    # Interpolated camera rays
    # ============================================================
    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        device = self.device
        l = resolution_level

        tx = torch.linspace(0, self.W - 1, self.W // l, device=device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=device)

        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)],
            dim=-1,
        )

        p = torch.matmul(
            self.intrinsics_all_inv[0, None, None, :3, :3],
            p[:, :, :, None],
        ).squeeze()

        rays_v = p / torch.linalg.norm(p, dim=-1, keepdim=True)

        pose_0 = np.linalg.inv(self.pose_all[idx_0].cpu().numpy())
        pose_1 = np.linalg.inv(self.pose_all[idx_1].cpu().numpy())

        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]

        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        slerp = Slerp([0, 1], rots)
        rot = slerp(ratio)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)

        rot = torch.from_numpy(pose[:3, :3]).to(device)
        trans = torch.from_numpy(pose[:3, 3]).to(device)

        rays_v = torch.matmul(
            rot[None, None, :3, :3],
            rays_v[:, :, :, None],
        ).squeeze()

        rays_o = trans[None, None, :3].expand(rays_v.shape)

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    # ============================================================
    # Near/Far computation
    # ============================================================
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    # ============================================================
    # Load image
    # ============================================================
    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return cv.resize(
            img,
            (self.W // resolution_level, self.H // resolution_level),
        ).clip(0, 255)
