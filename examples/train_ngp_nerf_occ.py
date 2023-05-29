"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import pathlib
import time
import warnings

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPRadianceField

from examples.utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    # default=str(pathlib.Path.cwd() / "data/360_v2"),
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    # choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--use_depth",
    action="store_true",
    help="Add depth_loss"
)
parser.add_argument(
    "--use_rgb_reg",
    action="store_true",
    help="Use rgb regularization loss"
)
parser.add_argument(
    "--depth_loss_weight",
    default=0.01,
    help="Depth loss weight",
    type=float
)
parser.add_argument(
    "--save_folder",
    help="Folder to save eval images",
    type=str
)
parser.add_argument(
    "--num_iters",
    "-it",
    type=int,
    default=5000,
    help="Maximum num of training iterations"
)
parser.add_argument(
    "--grid",
    type=str,
    help="path to grid if so"
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="learning rate"
)
parser.add_argument(
    "--bg_color",
    type=str,
    choices=["green", "black", "white", "random"],
    default="white",
    help="background color"
)
parser.add_argument(
    "--iter_depth_div",
    type=int,
    default=-1
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = args.num_iters
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = 0.0
    # scene parameters
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 4
    # render parameters
    render_step_size = 1e-3
    alpha_thre = 1e-2
    cone_angle = 0.004

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = args.num_iters
    init_batch_size = 1024
    target_sample_batch_size = 1 << 18
    weight_decay = (
        1e-5 if args.scene.startswith(("materials", "ficus", "drums")) else 1e-6
    )
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": f"{args.bg_color}"}
    test_dataset_kwargs = {} #{"color_bkgd_aug": f"{args.bg_color}"}
    # model parameters
    grid_resolution = 128
    grid_nlvl = 1
    # render parameters
    render_step_size = 5e-3
    alpha_thre = 0.0
    cone_angle = 0.0

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=estimator.aabbs[-1]).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(), lr=args.lr, eps=1e-15, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        # torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[
        #         max_steps // 2,
        #         max_steps * 3 // 4,
        #         max_steps * 9 // 10,
        #     ],
        #     gamma=0.33,
        # ),
    ]
)
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]
    depth_gt = data["depth"]
    depth_coeff = data["depth_coeff"]

    from_min, from_max, to_min, to_max = 0, 8, 1, 0  # params from blender
    depth_gt = (depth_gt - to_max) * (from_max - from_min) / (to_min - to_max) + from_min
    depth_gt[depth_gt > 0] = 8 - depth_gt[depth_gt > 0]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    if args.grid is None:
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1 if args.grid is not None else 1e-2,
            warmup_steps=0,
        )

    # render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    if args.iter_depth_div == -1:
        rgb_loss = F.smooth_l1_loss(rgb, pixels, reduction="none")
        loss = rgb_loss.mean()

        if args.use_depth:
            depth_loss = F.smooth_l1_loss(
                depth * depth_coeff,
                depth_gt,
                reduction='none'
            )
            if args.use_rgb_reg:
                rgb_regularization = rgb_loss.sum(dim=1, keepdim=True) / 3
                depth_loss = depth_loss * rgb_regularization

            loss += depth_loss.mean() * args.depth_loss_weight
    else:
        if step > max_steps // args.iter_depth_div:
            loss = F.smooth_l1_loss(rgb, pixels)
        else:
            loss = F.smooth_l1_loss(depth * depth_coeff, depth_gt) * args.depth_loss_weight

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 1000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        estimator.eval()

        if args.save_folder is not None:
            os.makedirs(args.save_folder, exist_ok=True)
        psnrs = []
        rmses = []
        lpips = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                depth_gt = data["depth"]
                depth_coeff = data["depth_coeff"]

                from_min, from_max, to_min, to_max = 0, 8, 1, 0  # params from blender
                depth_gt = (depth_gt - to_max) * (from_max - from_min) / (to_min - to_max) + from_min
                depth_gt[depth_gt > 0] = 8 - depth_gt[depth_gt > 0]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                rmse_depth = torch.sqrt(
                    F.mse_loss((depth * depth_coeff)[depth_mask], depth_gt[depth_mask])
                )
                psnrs.append(psnr.item())
                rmses.append(rmse_depth.item())
                # lpips.append(lpips_fn(rgb, pixels).item())
                if args.save_folder is not None:
                    imageio.imwrite(
                        os.path.join(args.save_folder, f"rgb_test_{i}.png"),
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    # imageio.imwrite(
                    #     os.path.join(args.save_folder, f"rgb_error_{i}.png"),
                    #     (
                    #         (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                    #     ).astype(np.uint8),
                    # )
        psnr_avg = sum(psnrs) / len(psnrs)
        rmse_avg = sum(rmses) / len(rmses)
        # lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg} | ",
              f"rmse_depth={rmse_avg} | "
              # f"lpips_avg={lpips_avg}"
              )
        print(f"{psnr_avg:.4f},{rmse_avg:.4f},{elapsed_time:.2f}")
