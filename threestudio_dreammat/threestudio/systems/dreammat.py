from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.ops import (
            get_mvp_matrix,
            get_projection_matrix,
            get_ray_directions,
            get_rays,
        )
import math
import random
from threestudio.utils.ops import get_activation
@threestudio.register("dreammat-system")
class DreamMat(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture:bool=True
        latent_steps: int = 1000
        save_train_image: bool = True
        save_train_image_iter: int = 1
        init_step: int = 0
        init_width:int=512
        init_height:int=512

        custom_prompt:str="Black trees"

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        prompt_utils = self.prompt_processor()
        out = self(batch)
        guidance_inp = out["comp_rgb"]  # 원본 렌더링 이미지 (HWC 형식)
        batch['cond_normal']=out.get('comp_normal', None)
        batch['cond_depth']=out.get('comp_depth', None)

        guidance_out = self.guidance(
            guidance_inp, prompt_utils, **batch, rgb_as_latents=False, 
        )

        loss = 0.0
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        if self.cfg.save_train_image:
            if self.true_global_step % self.cfg.save_train_image_iter == 0:
                # Generate image with Segmentation ControlNet
                custom_prompt = self.cfg.custom_prompt
                
                seg_controlnet_output = self.guidance.generate_with_segmentation_controlnet(
                    prompt_utils,
                    guidance_inp,
                    batch['cond_seg'],
                    batch['elevation'],
                    batch['azimuth'],
                    batch['camera_distances'],
                    custom_prompt=custom_prompt,
                    blend_alpha=0.85,  # Higher blend factor to make the generated trees more visible
                )
                
                seg_img = seg_controlnet_output.permute(0, 2, 3, 1)
                
                # Image row : train
                train_images_row = [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],  # 원본 렌더링 이미지
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["specular_light"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["diffuse_light"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["comp_depth"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": out["albedo"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["metalness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "grayscale",
                        "img": out["roughness"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": seg_img[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                
                # Image row : condition
                train_conditions_row = [
                    {
                        "type": "grayscale",
                        "img": batch["condition_map"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 1:4],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 4:7],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 7:10],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 10:13],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 13:16],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 16:19],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["condition_map"][0, :, :, 19:22],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": batch["cond_seg"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                
                self.save_image_grid(f"train/it{self.true_global_step}.png",
                    imgs=[train_images_row,train_conditions_row],
                    name="train_step",
                    step=self.true_global_step,
                )

        return {"loss": loss}

    def validation_step(self, batch):
        out = self(batch)
        srgb=out["comp_rgb"][0].detach()
        self.save_image_grid(
            f"validate/it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": srgb,
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["specular_light"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["diffuse_light"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["specular_color"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["diffuse_color"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["albedo"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "grayscale",
                    "img": out["metalness"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "grayscale",
                    "img": out["roughness"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch):
        # Get prompt utilities for test time
        prompt_utils = self.prompt_processor()
        
        out = self(batch)
        srgb = out["comp_rgb"][0].detach()
        
        # Generate image with Segmentation ControlNet
        custom_prompt = self.cfg.custom_prompt
        seg_controlnet_output = self.guidance.generate_with_segmentation_controlnet(
            prompt_utils,
            srgb.unsqueeze(0),  # Need batch dimension
            batch['cond_seg'],
            batch['elevation'],
            batch['azimuth'],
            batch['camera_distances'],
            custom_prompt=custom_prompt,
            blend_alpha=0.85,  # Higher blend factor to make the generated trees more visible
        )
        
        # Convert to image format
        seg_img = seg_controlnet_output.permute(0, 2, 3, 1)[0]
        
        # Make sure the seg_render directory exists
        os.makedirs(f"it{self.true_global_step}-test/seg_render", exist_ok=True)
        os.makedirs(f"it{self.true_global_step}-test/seg_mask", exist_ok=True)
        
        # 세그멘테이션 마스크도 저장
        seg_mask = batch['cond_seg'][0]
        self.save_img(seg_mask, f"it{self.true_global_step}-test/seg_mask/{batch['index'][0]}.png")
        
        # Save the segmentation-processed image separately
        self.save_img(seg_img, f"it{self.true_global_step}-test/seg_render/{batch['index'][0]}.png")
        
        self.save_image_grid(
            f"it{self.true_global_step}-test/view/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": srgb,
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": seg_mask,
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                    {
                        "type": "rgb",
                        "img": seg_img,
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["albedo"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
                {
                    "type": "grayscale",
                    "img": out["metalness"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "grayscale",
                    "img": out["roughness"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )
        
        mask = out["opacity"][0].detach()
        albedo = out["albedo"][0].detach()
        roughness = out["roughness"][0].detach().repeat(1,1,3)
        metallic = out["metalness"][0].detach().repeat(1,1,3)
        self.save_img(torch.cat((albedo,mask),2),f"it{self.true_global_step}-test/albedo/{batch['index'][0]}.png")
        self.save_img(torch.cat((roughness,mask),2),f"it{self.true_global_step}-test/roughness/{batch['index'][0]}.png")
        self.save_img(torch.cat((metallic,mask),2),f"it{self.true_global_step}-test/metallic/{batch['index'][0]}.png")
        self.save_img(torch.cat((srgb,mask),2),f"it{self.true_global_step}-test/render/{batch['index'][0]}.png")
        # Save the segmentation result with mask
        self.save_img(torch.cat((seg_img,mask),2),f"it{self.true_global_step}-test/seg_render/{batch['index'][0]}.png")

    def on_test_epoch_end(self):
        viewpath="it"+str(self.true_global_step)+"-test/view"
        self.save_gif(viewpath,fps=30)