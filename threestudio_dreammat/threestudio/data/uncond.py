import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from threestudio.models.renderers.raytracing_renderer import RayTracer

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
import subprocess,os
import pickle
import cv2
import numpy as np
def fovy_to_focal(fovy, sensor_height):
    return sensor_height / (2 * math.tan(fovy / 2))
@dataclass
class RandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    fix_view_num: int = 128
    fix_env_num: int = 5
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    use_fix_views: bool = True
    blender_generate: bool = False

class RandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb = (
            torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

            

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi * 2 - math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        env_id: Int[Tensor,"B"]=(torch.rand(self.batch_size)*self.cfg.fix_env_num).floor().int()
        #env_id[0]=3
        return {
            "env_id":env_id,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            #'k':Ks,
            "c2w": c2w,
            "w2c": w2c,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
        }

class FixCameraIterableDataset(IterableDataset, Updateable):

    def render_oneview_gt(self, view_id):

        elevation_deg: Float[Tensor, "B"] = self.elevation_degs[view_id]
        elevation: Float[Tensor, "B"] = elevation_deg * math.pi / 180

        
        azimuth_deg: Float[Tensor, "B"] = self.azimuth_degs[view_id]
        azimuth: Float[Tensor, "B"] = azimuth_deg * math.pi / 180
        

        camera_distances: Float[Tensor, "B"] = self.fix_camera_distances[view_id]

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb = self.camera_perturbs[view_id,...]
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb = self.center_perturbs[view_id,...]
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb = self.up_perturbs[view_id,...]
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = self.fovy_degs[view_id]
        fovy = fovy_deg * math.pi / 180

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        proj_mtx = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)
        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        ray_tracer = RayTracer(self.mesh.v_pos, self.mesh.t_pos_idx)
        inters, normals, depth = ray_tracer.trace(rays_o.reshape(-1,3), rays_d.reshape(-1,3))
        normals = F.normalize(normals, dim=-1)
        miss_mask = depth >= 10
        hit_mask = ~miss_mask

        

        def xfm_vectors(vectors, matrix):
            out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
            return out
        
        def saveimg(img,path):
            from PIL import Image
            img=img.detach().reshape(self.height,self.width,3).cpu().numpy()
            img=Image.fromarray((img*255).astype(np.uint8))
            img.save(path)
        

        normal_view  = xfm_vectors(normals[hit_mask].view(view_id.shape[0], normals[hit_mask].shape[0], normals[hit_mask].shape[1]), w2c.to(normals.device)).view(*normals[hit_mask].shape)
        normal_view = F.normalize(normal_view)
        normal_controlnet=0.5*(normal_view+1)
        normal_controlnet[..., 0]=1.0-normal_controlnet[..., 0] # Flip the sign on the x-axis to match bae system
        normals[hit_mask]=normal_controlnet

        min_val=0.3
        depth_inv = 1. / (depth + 1e-6)
        depth_max = depth_inv[hit_mask].max()
        depth_min = depth_inv[hit_mask].min()
        depth[hit_mask] = (1 - min_val) *(depth_inv[hit_mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val
        depth[~hit_mask]=0.0

        hit_mask=hit_mask.reshape((self.height,self.width,1))
        hit_mask=hit_mask.repeat(1,1,3).float()
        depth=depth.reshape((self.height,self.width,1))
        depth=depth.repeat(1,1,3)
        normals=normals.reshape((self.height,self.width,3))
        #saveimg(hit_mask,self.temp_image_save_dir+'/gt/mask'+str(view_id[0])+'.png')
        #saveimg(depth,self.temp_image_save_dir+'/gt/depth'+str(view_id[0])+'.png')
        saveimg(normals,self.temp_image_save_dir+'/gt/normal'+str(view_id[0])+'.png')

        return normals, depth, hit_mask

    def render_fixview_imgs(self):
        envmap_dir = "load/lights/envmap"

        if self.cfg.blender_generate:

            elevation_deg = self.elevation_degs
            elevation = elevation_deg * math.pi / 180

            
            azimuth_deg = self.azimuth_degs
            azimuth = azimuth_deg * math.pi / 180
            

            camera_distances = self.fix_camera_distances

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )
            camera_perturb = self.camera_perturbs
            camera_positions = camera_positions + camera_perturb
            # # sample center perturbations from a normal distribution with mean 0 and std center_perturb
            center_perturb = self.center_perturbs
            center = torch.zeros_like(camera_positions)
            center = center + center_perturb
            up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1).expand(128, -1)
            lookat = F.normalize(center - camera_positions, dim=-1)
            right = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4 = torch.cat(
                [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            c2w = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0
            fovy_deg = self.fovy_degs
            fovy = fovy_deg * math.pi / 180
            focal_length = 0.5 * self.height / torch.tan(0.5 * fovy)

            
            data = {}
            data['v_pos']     = self.mesh.v_pos.cpu().numpy()
            #data['v_uv']      = self.mesh.v_uv.cpu().numpy()
            data['t_pos_idx'] = self.mesh.t_pos_idx.cpu().numpy()
            # --- Add v_color to pickle data if it exists --- 
            if hasattr(self.mesh, 'v_color') and self.mesh.v_color is not None:
                 data['v_color'] = self.mesh.v_color.cpu().numpy()
                 threestudio.info("Added v_color to pickle data.") # Add info message
            else:
                 # Handle the case where v_color is missing - crucial for blender script
                 threestudio.warn("v_color not found in mesh, cannot add to pickle. Blender script might fail if it requires v_color.")
                 # Option 1: Add a placeholder (e.g., array of ones) - might hide issues
                 # num_vertices = data['v_pos'].shape[0]
                 # data['v_color'] = np.ones((num_vertices, 4), dtype=np.float32) 
                 # Option 2: Let it be missing - blender script needs robust checking (already added)
                 pass
            # --- End of adding v_color ---

            data['width']     = self.width
            data['height']    = self.height
            data['focal_length'] = focal_length.cpu().numpy()
            data['c2w']       = c2w.cpu().numpy()

            # 원래 GLB 파일로 세그멘테이션 메시를 처리하던 방식 제거
            # 세그멘테이션 레이블을 원본 메시에 직접 적용
            # 세그멘테이션 레이블 데이터 로드
            try:
                import numpy as np
                
                # 세그멘테이션 레이블 파일 경로
                seg_labels_path = "load/shapes/seg/ruined_house.npy"
                if os.path.exists(seg_labels_path):
                    seg_labels = np.load(seg_labels_path)
                    
                    # face 수가 일치하는지 확인
                    if len(seg_labels) == len(data['t_pos_idx']):
                        threestudio.info(f"Loaded segmentation labels: {seg_labels.shape}, unique labels: {np.unique(seg_labels)}")
                        
                        # 색상 맵 정의 (레이블 -> RGB 색상)
                        # 고유한 색상을 위해 각 레이블별 고정 색상 사용
                        num_labels = int(np.max(seg_labels)) + 1
                        
                        # 고정된 색상 매핑 정의
                        color_map = np.array([
                            [4/255.0, 200/255.0, 3/255.0], # 0 : tree
                            [0.0, 0.0, 0.0], # 1 : house
                            [4/255.0, 200/255.0, 3/255.0], # 2 : tree
                            [4/255.0, 200/255.0, 3/255.0], # 3 : tree
                            [0.0, 0.0, 0.0], # 4 : house
                            [0.0, 0.0, 0.0], # 5 : house
                            [0.0, 0.0, 0.0], # 6 : house
                            [0.0, 0.0, 0.0], # 7 : house
                            [0.0, 0.0, 0.0], # 8 : house
                            [0.0, 0.0, 0.0], # 9 : house
                            [4/255.0, 200/255.0, 3/255.0], # 10 : tree
                            [0.0, 0.0, 0.0], # 11 : house
                            [0.0, 0.0, 0.0], # 12 : house
                            [0.0, 0.0, 0.0], # 13 : house
                            [0.0, 0.0, 0.0], # 14 : house
                            [0.0, 0.0, 0.0], # 15 : house
                            [0.0, 0.0, 0.0], # 16 : house
                        ], dtype=np.float32)
                        
                        # 레이블 수가 color_map보다 많으면 추가 색상 생성
                        if num_labels > len(color_map):
                            additional_colors = np.random.random((num_labels - len(color_map), 3))
                            color_map = np.vstack([color_map, additional_colors])
                        
                        # house에 해당하는 레이블 목록
                        house_labels = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
                        
                        # house가 아닌 face만 필터링
                        non_house_mask = np.array([label not in house_labels for label in seg_labels])
                        threestudio.info(f"Keeping {non_house_mask.sum()} out of {len(seg_labels)} faces (removing house)")
                        
                        # house가 아닌 face만 남긴 새로운 t_pos_idx 생성
                        filtered_t_pos_idx = data['t_pos_idx'][non_house_mask]
                        
                        # 업데이트된 face 인덱스로 데이터 교체
                        data['t_pos_idx'] = filtered_t_pos_idx
                        
                        # 파일터링된 face에 대한 색상만 계산
                        face_colors = color_map[seg_labels[non_house_mask]]
                        
                        # vertex colors 계산 (face colors에서 평균 계산)
                        vertex_colors = np.zeros((len(data['v_pos']), 3), dtype=np.float32)
                        vertex_count = np.zeros(len(data['v_pos']), dtype=np.int32)
                        
                        # 각 face의 색상을 해당 face의 vertex에 할당
                        for i, face in enumerate(filtered_t_pos_idx):
                            for vertex_idx in face:
                                vertex_colors[vertex_idx] += face_colors[i]
                                vertex_count[vertex_idx] += 1
                        
                        # 평균 계산
                        for i in range(len(vertex_colors)):
                            if vertex_count[i] > 0:
                                vertex_colors[i] /= vertex_count[i]
                        
                        # 알파 채널 추가
                        vertex_colors_with_alpha = np.ones((len(data['v_pos']), 4), dtype=np.float32)
                        vertex_colors_with_alpha[:, :3] = vertex_colors
                        
                        # data에 vertex color 추가
                        data['v_color'] = vertex_colors_with_alpha
                    else:
                        threestudio.warn(f"Segmentation labels count ({len(seg_labels)}) doesn't match face count ({len(data['t_pos_idx'])})")
                else:
                    threestudio.warn(f"Segmentation labels file not found: {seg_labels_path}")
            except Exception as e:
                threestudio.warn(f"Failed to apply segmentation colors to original mesh: {str(e)}")
            
            os.makedirs('temp', exist_ok=True)
            pkl_path = 'temp/render_fixview_temp.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            
            cmd = f'blender -b -P ./threestudio/data/blender_script_fixview.py -- --param_dir {pkl_path} --env_dir {envmap_dir} --output_dir {self.temp_image_save_dir} --num_images {self.cfg.fix_view_num}'
            print(cmd)
            print("pre-rendering light conditions...please wait for about 15min")
            # Execute the command and capture output
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # Check return code
            if process.returncode != 0:
                print(f"Blender script failed with return code {process.returncode}")
                # Optionally raise an exception or handle the error
                # raise RuntimeError(f"Blender script failed. Check logs above.")
            else:
                print("rendering done")

        def loadrgb(imgpath,dim):
            import numpy as np  # numpy를 함수 내에서 임포트
            img = cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32) / 255.0
            return img

        def loaddepth(imgpath,dim):
            import numpy as np  # numpy를 함수 내에서 임포트
            depth = cv2.imread(imgpath, cv2.IMREAD_ANYDEPTH)/1000
            # Check if image loading failed
            if depth is None:
                print(f"Error: Could not load depth image at {imgpath}")
                # Return a zero array or handle appropriately
                return np.zeros((dim[1], dim[0], 1), dtype=np.float32)
            depth = cv2.resize(depth, dim, interpolation = cv2.INTER_NEAREST)
            object_mask = depth>0
                
            if object_mask.sum()<=0:
                print(f"Warning: No object mask found in depth image {imgpath}")
                return depth[...,None]

            min_val=0.3
        
            depth_inv = 1. / (depth + 1e-6)
            
            depth_max = depth_inv[object_mask].max()
            depth_min = depth_inv[object_mask].min()
                        
            depth[object_mask] = (1 - min_val) *(depth_inv[object_mask] - depth_min) / (depth_max - depth_min + 1e-6) + min_val
            return depth[...,None]
        
        self.depths = torch.zeros((128,self.height,self.width,1))
        self.normals = torch.ones((128,self.height,self.width,3))
        self.lightmaps = torch.zeros((128,5,self.height,self.width,18))
        self.segs = torch.ones((128,self.height,self.width,3)) # 세그멘테이션 텐서 초기화

        dim = (self.width, self.height)
        for view_idx in range(self.cfg.fix_view_num):
            depth_path = self.temp_image_save_dir+"/depth/"+f"{view_idx:03d}.png"
            normal_path = self.temp_image_save_dir+"/normal/"+f"{view_idx:03d}.png"
            seg_path = self.temp_image_save_dir+"/seg/"+f"{view_idx:03d}.png"
            
            self.depths[view_idx] = torch.from_numpy(loaddepth(depth_path, dim))
            self.normals[view_idx] = torch.from_numpy(loadrgb(normal_path, dim))
            
            seg_img = loadrgb(seg_path, dim)
            if view_idx == 0:  # 첫 번째 뷰에 대해서만 로깅
                threestudio.info(f"Successfully loaded seg image: shape={seg_img.shape}, range=[{seg_img.min()}, {seg_img.max()}]")
                import numpy as np  # numpy 명시적 임포트 추가
                unique_colors = np.unique(seg_img.reshape(-1, 3), axis=0)
            self.segs[view_idx] = torch.from_numpy(seg_img)
                
            for env_idx in range(1,6):
                light_path_m0r0 =    self.temp_image_save_dir+"/light/"+f"{view_idx:03d}_m0.0r0.0_env"+str(env_idx)+".png"
                light_path_m0rhalf = self.temp_image_save_dir+"/light/"+f"{view_idx:03d}_m0.0r0.5_env"+str(env_idx)+".png"
                light_path_m0r1 =    self.temp_image_save_dir+"/light/"+f"{view_idx:03d}_m0.0r1.0_env"+str(env_idx)+".png"
                light_path_m1r0 =    self.temp_image_save_dir+"/light/"+f"{view_idx:03d}_m1.0r0.0_env"+str(env_idx)+".png"
                light_path_m1rhalf = self.temp_image_save_dir+"/light/"+f"{view_idx:03d}_m1.0r0.5_env"+str(env_idx)+".png"
                light_path_m1r1 =    self.temp_image_save_dir+"/light/"+f"{view_idx:03d}_m1.0r1.0_env"+str(env_idx)+".png"
                light_m0r0=loadrgb(light_path_m0r0,dim)
                light_m0rhalf=loadrgb(light_path_m0rhalf,dim)
                light_m0r1=loadrgb(light_path_m0r1,dim)
                light_m1r0=loadrgb(light_path_m1r0,dim)
                light_m1rhalf=loadrgb(light_path_m1rhalf,dim)
                light_m1r1=loadrgb(light_path_m1r1,dim)
                self.lightmaps[view_idx, env_idx-1] = torch.from_numpy(np.concatenate([
                    light_m0r0, light_m0rhalf, light_m0r1, light_m1r0, light_m1rhalf, light_m1r1], axis=-1))

    def set_fix_elevs(self) -> None:
        elevation_degs1 = (
                torch.rand(int(self.cfg.fix_view_num/2))
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
        elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
        elevation = torch.asin(
                2
                * (
                    torch.rand(int(self.cfg.fix_view_num/2))
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
        elevation_degs2 = elevation / math.pi * 180.0
        self.elevation_degs=torch.cat((elevation_degs1,elevation_degs2))
    
    def set_fix_azims(self) ->None:
        self.azimuth_degs = (
                torch.rand(self.cfg.fix_view_num) + torch.arange(self.cfg.fix_view_num)
            ) / self.cfg.fix_view_num * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]

    def set_fix_camera_distance(self) -> None:
        self.fix_camera_distances= (
            torch.rand(self.cfg.fix_view_num)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )
    
    def set_fix_camera_perturb(self) -> None:
        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        self.camera_perturbs = (
            torch.rand(self.cfg.fix_view_num, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
    
    def set_fix_center_perturb(self) ->None:
        self.center_perturbs = (
            torch.randn(self.cfg.fix_view_num, 3) * self.cfg.center_perturb
        )
    
    def set_fix_up_perturb(self) -> None:
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        self.up_perturbs = (
            torch.randn(self.cfg.fix_view_num, 3) * self.cfg.up_perturb
        )

    def set_fix_fovy(self) -> None:
        self.fovy_degs = (
            torch.rand(self.cfg.fix_view_num) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )

    def __init__(self, cfg: Any, mesh,prerender_dir) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.mesh = mesh
        self.temp_image_save_dir=prerender_dir
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

        self.set_fix_elevs()
        self.set_fix_azims()
        self.set_fix_camera_distance()
        self.set_fix_camera_perturb()
        self.set_fix_center_perturb()
        self.set_fix_up_perturb()
        self.set_fix_fovy()
        
        os.makedirs(self.temp_image_save_dir+"/gt", exist_ok=True)
        for i in range(self.cfg.fix_view_num):
            gt_view_id=torch.ones((1))*i 
            self.render_oneview_gt(gt_view_id.long()) 

        self.render_fixview_imgs()

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        #self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        view_id=(torch.rand(self.batch_size)*self.cfg.fix_view_num).floor().long()
        
        elevation_deg: Float[Tensor, "B"] = self.elevation_degs[view_id]
        elevation: Float[Tensor, "B"] = elevation_deg * math.pi / 180

        
        azimuth_deg: Float[Tensor, "B"] = self.azimuth_degs[view_id]
        azimuth: Float[Tensor, "B"] = azimuth_deg * math.pi / 180
        

        camera_distances: Float[Tensor, "B"] = self.fix_camera_distances[view_id]

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb = self.camera_perturbs[view_id,...]
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb = self.center_perturbs[view_id,...]
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb = self.up_perturbs[view_id,...]
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = self.fovy_degs[view_id]
        fovy = fovy_deg * math.pi / 180

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)

        #env_id: Int[Tensor,"B"]=(torch.rand(self.batch_size)*self.cfg.fix_env_num).floor().long()
        env_id: Int[Tensor,"B"]=(torch.rand(self.batch_size)*self.cfg.fix_env_num).floor().long()
        #env_id[0]=3
        cur_depth = self.depths[view_id,...]
        cur_normal = self.normals[view_id,...]
        cur_light = self.lightmaps[view_id,env_id,...]
        cur_seg = self.segs[view_id,...]  # Segmentation Map

        condition_map = torch.cat((cur_depth,cur_normal,cur_light),-1) # (H, W, 22)

        return {
            "view_id":view_id,
            "env_id": env_id,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            #'k':Ks,
            "c2w": c2w,
            "w2c": w2c,
            "light_positions": None,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "condition_map":condition_map,
            "cond_seg":cur_seg
        }

class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180

        light_positions = camera_positions

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx, w2c= get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.w2c = w2c
        #self.Ks=Ks
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "env_id": 4,
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "w2c": self.w2c[index],
            "camera_positions": self.camera_positions[index],
            #'k':self.Ks[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("random-camera-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, mesh, prerender_dir,cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)
        self.mesh=mesh
        self.prerender_dir=prerender_dir
        os.makedirs(self.prerender_dir, exist_ok=True)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            if self.cfg.use_fix_views:
                self.train_dataset = FixCameraIterableDataset(self.cfg,self.mesh,self.prerender_dir)
            else:
                self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )