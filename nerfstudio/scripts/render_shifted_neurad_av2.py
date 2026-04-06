# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
Shifted render for Argoverse 2 with NeuRAD checkpoints.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
import struct
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import mediapy as media
import numpy as np
import plotly.graph_objs as go
import torch
import tyro
import viser.transforms as tf
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_interpolated_spiral_camera_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.datasets.base_dataset import Dataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.poses import inverse
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
import pyarrow as pa
import pyarrow.feather as feather
try:
    from av2.utils.io import read_ego_SE3_sensor
    has_av2 = True
except ImportError:
    has_av2 = False
    CONSOLE.print("[yellow]Warning: av2 library not available, cannot load calibration. Points will be in wrong coordinate frame.[/yellow]")


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""
    render_nearest_camera: bool = False
    """Whether to render the nearest training camera to the rendered camera."""
    check_occlusions: bool = False
    """If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other"""


@dataclass
class ShiftedDatasetRender(BaseRender):
    """Render all images in the dataset."""

    pose_source: Literal["train", "val", "test", "train+test", "train+val"] = "val"
    """Split to render."""
    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    config_output_dir: Optional[Path] = None
    """Override the config output dir. Used to load the model."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb", "gt-rgb", "depth"])
    strict_load: bool = True
    """Whether to strictly load the config."""
    load_ignore_keys: Optional[List[str]] = field(
        default_factory=lambda: []
    )  # e.g. ["model.camera_optimizer.pose_adjustment", "_model.camera_optimizer.pose_adjustment"]
    """Keys to ignore when loading the config."""

    render_height: Optional[int] = None
    """Height to render the images at."""
    render_width: Optional[int] = None
    """Width to render the images at."""
    output_height: Optional[int] = None
    """Height to crop the output images at."""
    output_width: Optional[int] = None
    """Width to crop the output images at."""

    shift: Tuple[float, float, float] = (0, 0, 0)
    """Shift to apply to the camera pose."""

    render_point_clouds: bool = False
    """Whether to render point clouds."""

    original_data_root: Optional[Path] = Path("/workspace/datasets/self-driving/argoverse2")
    """Root of original Argoverse2 dataset (sensor/<split>/<log_id>/...)."""
    target_data_root: Optional[Path] = Path("data/argoverse2")
    """Root of target dataset to write shifted outputs in AV2 layout."""

    def main(self):
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.config_output_dir is not None:
                config.output_dir = self.config_output_dir
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
            strict_load=self.strict_load,
            ignore_keys=self.load_ignore_keys,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))

        self.output_path.mkdir(exist_ok=True, parents=True)
        metrics_out = dict()
        for split in self.pose_source.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
                lidar_dataset = datamanager.train_lidar_dataset
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                lidar_dataset = datamanager.eval_lidar_dataset
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            
            # Get the dataparser transform (w2m - world to mean transform)
            w2m = dataparser_outputs.dataparser_transform
            
            # Compute shift vector in world coordinates from shift in mean coordinates
            # w2m transforms from world to mean, so we need m2w (mean to world) = inverse(w2m)
            m2w = inverse(w2m)
            shift_mean = torch.tensor(self.shift, dtype=torch.float32, device=w2m.device)
            # Transform translation vector using rotation part only (for pure translation vectors)
            shift_world = (m2w[:3, :3] @ shift_mean).cpu().numpy()
            
            CONSOLE.print(f"[green]Shift in mean coordinates: {self.shift}[/green]")
            CONSOLE.print(f"[green]Shift in world coordinates: [{shift_world[0]:.6f}, {shift_world[1]:.6f}, {shift_world[2]:.6f}][/green]")
            
            # Store shift in world coordinates to scene folder root for use in missing points AV2
            log_id = str(getattr(data_manager_config.dataparser, "sequence"))
            target_root = self.target_data_root or Path("data/argoverse2")
            scene_root = target_root / "sensor" / split / log_id
            scene_root.mkdir(parents=True, exist_ok=True)
            shift_file = scene_root / "shift_world.json"
            shift_data = {
                "shift_mean": list(self.shift),
                "shift_world": [float(shift_world[0]), float(shift_world[1]), float(shift_world[2])],
                "log_id": log_id,
                "split": split
            }
            with open(shift_file, "w", encoding="UTF-8") as f:
                json.dump(shift_data, f, indent=4)
            CONSOLE.print(f"[green]Saved shift_world to: {shift_file}[/green]")

            dataset.cameras.height = (
                torch.full_like(dataset.cameras.height, self.render_height)
                if self.render_height is not None
                else dataset.cameras.height
            )
            dataset.cameras.width = (
                torch.full_like(dataset.cameras.width, self.render_width)
                if self.render_width is not None
                else dataset.cameras.width
            )
            
            c2w = dataset.cameras.camera_to_worlds.clone()
            c2w[..., :3, 3] += torch.tensor(self.shift, dtype=torch.float32)
            dataset.cameras.camera_to_worlds = c2w

            dataloader = FixedIndicesEvalDataloader(
                dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            lidar_dataloader = FixedIndicesEvalDataloader(
                dataset=lidar_dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )

            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    # Get the original filename path
                    original_image_path = Path(dataparser_outputs.image_filenames[camera_idx])
                    image_name = original_image_path.with_suffix("").relative_to(images_root)
                    
                    # Parse image_name to extract camera name and filename
                    # image_name format: {sequence_id}/sensors/cameras/{camera_name}/{timestamp}
                    image_name_parts = image_name.parts
                    # Find the index of "cameras" to locate camera name
                    if "cameras" in image_name_parts:
                        cameras_idx = image_name_parts.index("cameras")
                        if cameras_idx + 1 < len(image_name_parts):
                            camera_name = image_name_parts[cameras_idx + 1]
                            filename = image_name_parts[-1]  # timestamp without extension
                        else:
                            CONSOLE.print(f"[yellow]Warning: Could not parse camera name from {image_name}[/yellow]")
                            continue
                    else:
                        # Fallback: try to get from original path structure
                        original_parts = original_image_path.parts
                        if "cameras" in original_parts:
                            cameras_idx = original_parts.index("cameras")
                            if cameras_idx + 1 < len(original_parts):
                                camera_name = original_parts[cameras_idx + 1]
                                filename = original_image_path.stem
                            else:
                                CONSOLE.print(f"[yellow]Warning: Could not parse camera name from {original_image_path}[/yellow]")
                                continue
                        else:
                            CONSOLE.print(f"[yellow]Warning: Could not find cameras in path {image_name}[/yellow]")
                            continue

                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(camera)

                    if self.output_height is not None:
                        dataset.cameras.height[batch["image_idx"]] = torch.full_like(
                            dataset.cameras.height[0:1], self.output_height
                        )
                        batch["image"] = batch["image"][..., : self.output_height, :, :]
                        outputs["rgb"] = outputs["rgb"][..., : self.output_height, :, :]

                    if self.output_width is not None:
                        dataset.cameras.width[batch["image_idx"]] = torch.full_like(
                            dataset.cameras.width[0:1], self.output_width
                        )
                        batch["image"] = batch["image"][..., : self.output_width, :]
                        outputs["rgb"] = outputs["rgb"][..., : self.output_width, :]

                    # Predicted RGB (model output)
                    output_image = outputs["rgb"]

                    # Map to color space / numpy
                    output_image = (
                        colormaps.apply_colormap(
                            image=output_image,
                            colormap_options=self.colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )

                    # Save to file
                    height = (
                        min(output_image.shape[0], self.output_height)
                        if self.output_height
                        else output_image.shape[0]
                    )
                    width = (
                        min(output_image.shape[1], self.output_width)
                        if self.output_width
                        else output_image.shape[1]
                    )
                    output_image = output_image[:height, :width]

                    # Save in AV2 format: target_root/sensor/{split}/{log_id}/sensors/cameras/{camera_name}/{filename}.jpg
                    output_path = scene_root / "sensors" / "cameras" / camera_name / f"{filename}.jpg"
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    media.write_image(
                        output_path,
                        output_image,
                        fmt="jpeg",
                        quality=self.jpeg_quality,
                    )

                    # GT camera image (dataset JPEG) under gt/<camera_name>/ — same stem as rgb
                    if "image" in batch and batch["image"] is not None:
                        gt_t = batch["image"]
                        if gt_t.dim() == 3 and gt_t.shape[0] == 3:
                            gt_t = gt_t.permute(1, 2, 0)
                        gt_np = gt_t.clamp(0.0, 1.0).float().detach().cpu().numpy()
                        gt_np = gt_np[:height, :width]
                        gt_path = scene_root / "gt" / camera_name / f"{filename}.jpg"
                        gt_path.parent.mkdir(parents=True, exist_ok=True)
                        media.write_image(
                            gt_path,
                            gt_np,
                            fmt="jpeg",
                            quality=self.jpeg_quality,
                        )

            if self.render_point_clouds:
                with Progress(
                    TextColumn(f":movie_camera: Rendering lidars for split {split} :movie_camera:"),
                    BarColumn(),
                    TaskProgressColumn(
                        text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                        show_speed=True,
                    ),
                    ItersPerSecColumn(suffix="fps"),
                    TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                    TimeElapsedColumn(),
                ) as progress:
                    vis_flag = True
                    with torch.no_grad():
                        output_path = self.output_path / split / "lidar"
                        output_path.mkdir(exist_ok=True, parents=True)

                        vis_output_path = self.output_path / split / "lidar_vis"
                        vis_output_path.mkdir(exist_ok=True, parents=True)

                        vis_gt_output_path = self.output_path / split / "lidar_vis_gt"
                        vis_gt_output_path.mkdir(exist_ok=True, parents=True)

                        # Buffer points per timestamp to combine up/down into one AV2 sweep
                        sweep_buffer: Dict[float, Dict[str, Dict[str, torch.Tensor]]] = {}
                        # Track actual file paths/timestamps from dataloader
                        sweep_file_paths: Dict[float, Path] = {}
                        # sensor id to name mapping (same lidar_dataset as lidar_dataloader)
                        sensor_idx_to_name = None
                        if getattr(lidar_dataset, "metadata", None) is not None:
                            sensor_idx_to_name = lidar_dataset.metadata.get("sensor_idx_to_name", None)

                        # Filenames must come from the same split as lidar_dataloader
                        lidar_filenames = None
                        if hasattr(lidar_dataset, "filenames"):
                            lidar_filenames = lidar_dataset.filenames
                        elif getattr(lidar_dataset, "metadata", None):
                            lidar_filenames = lidar_dataset.metadata.get("pc_filenames", None)
                        
                        if lidar_filenames is None or len(lidar_filenames) == 0:
                            CONSOLE.print(f"[yellow]Warning: Could not get lidar filenames from dataset. Skipping lidar rendering for split {split}.[/yellow]")
                            continue

                        from nerfstudio.data.utils.lidar_elevation_mappings import (
                            ARGOVERSE2_VELODYNE_VLP32C_ELEVATION_MAPPING as EL_MAP,
                        )

                        for lidar_idx, (lidar, batch) in enumerate(
                            progress.track(lidar_dataloader, total=len(lidar_dataloader))
                        ):
                            device = lidar.lidar_to_worlds[:, :3, 3].device
                            lidar.lidar_to_worlds[:, :3, 3] += torch.tensor(self.shift, dtype=torch.float32).to(device)

                            lidar_out, _ = pipeline.model.get_outputs_for_lidar(lidar, batch=batch)
                            xyz = lidar_out["points"]
                            intensity = lidar_out["intensity"]
                            time_ch = lidar_out.get("time")

                            if "ray_drop_prob" in lidar_out:
                                rd_mask = (lidar_out["ray_drop_prob"] < 0.5).squeeze(-1)
                                xyz = xyz[rd_mask]
                                intensity = intensity[rd_mask]
                                if time_ch is not None:
                                    time_ch = time_ch[rd_mask]
                            elif "ray_drop_logits" in lidar_out:
                                rd_mask = lidar_out["ray_drop_logits"].sigmoid().squeeze(-1) <= 0.5
                                xyz = xyz[rd_mask]
                                intensity = intensity[rd_mask]
                                if time_ch is not None:
                                    time_ch = time_ch[rd_mask]


                            intensity_uint8 = (intensity.flatten().clamp(0, 1) * 255.0).round().to(torch.uint8)
                            if time_ch is not None:

                                sweep_time_s = lidar.times[0].to(device=time_ch.device, dtype=time_ch.dtype)
                                rel_s = time_ch.flatten() - sweep_time_s
                                offset_ns = (rel_s * 1e9).to(torch.int64)
                            else:
                                offset_ns = torch.zeros(xyz.shape[0], dtype=torch.int64, device=xyz.device)

                            dirn = xyz / (xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8))
                            elev_deg = torch.rad2deg(torch.asin(dirn[..., 2].clamp(-1.0, 1.0)))
                            channel_elev = torch.tensor(
                                [EL_MAP[k] for k in sorted(EL_MAP.keys())], dtype=torch.float32, device=xyz.device
                            )
                            nearest = torch.argmin(torch.abs(elev_deg[:, None] - channel_elev[None, :]), dim=1).to(
                                torch.int32
                            )

                            num_files = len(lidar_filenames)
                            lidar_name = None
                            if lidar.metadata and "sensor_idxs" in lidar.metadata:
                                sid = int(lidar.metadata["sensor_idxs"][0].item())
                                if sensor_idx_to_name is not None and sid in sensor_idx_to_name:
                                    lidar_name = sensor_idx_to_name[sid]
                                else:
                                    lidar_name = "lidar_up" if sid == 0 else "lidar_down"
                            if lidar_name is None and num_files > 0:
                                lidar_name = "lidar_up" if lidar_idx < num_files else "lidar_down"
                            if lidar_name is None:
                                CONSOLE.print(
                                    f"[yellow]Warning: could not resolve lidar_up/lidar_down for idx {lidar_idx}. Skipping.[/yellow]"
                                )
                                continue

                            if lidar_name == "lidar_down":
                                laser_number = nearest + 32
                            else:
                                laser_number = nearest

                            sweep_time = float(lidar.times[0].item())

                            # Register sweep + GT feather path once per timestamp (before buffering up/down).
                            if sweep_time not in sweep_buffer:
                                file_idx = lidar_idx % num_files
                                if not (0 <= file_idx < num_files):
                                    CONSOLE.print(
                                        f"[yellow]Warning: Could not map lidar_idx {lidar_idx} to file index. Skipping.[/yellow]"
                                    )
                                    continue
                                sweep_buffer[sweep_time] = {}
                                sweep_file_paths[sweep_time] = Path(lidar_filenames[file_idx])

                            sweep_buffer[sweep_time][lidar_name] = {
                                "xyz": xyz.detach().cpu(),
                                "intensity": intensity_uint8.detach().cpu(),
                                "offset_ns": offset_ns.detach().cpu(),
                                "laser_number": laser_number.detach().cpu()}

                            # If both up and down present for this timestamp, write combined AV2 feather
                            if "lidar_up" in sweep_buffer[sweep_time] and "lidar_down" in sweep_buffer[sweep_time]:
                                up = sweep_buffer[sweep_time]["lidar_up"]
                                down = sweep_buffer[sweep_time]["lidar_down"]
                                
                                # Build IO paths using exact filename from dataset - no fallbacks
                                orig_pc_path = sweep_file_paths.get(sweep_time)
                                if orig_pc_path is None:
                                    CONSOLE.print(f"[red]Error: No file path stored for sweep_time {sweep_time}. Skipping.[/red]")
                                    del sweep_buffer[sweep_time]
                                    if sweep_time in sweep_file_paths:
                                        del sweep_file_paths[sweep_time]
                                    continue
                                
                                if not orig_pc_path.exists():
                                    CONSOLE.print(f"[red]Error: Original feather file does not exist: {orig_pc_path}. Skipping.[/red]")
                                    del sweep_buffer[sweep_time]
                                    if sweep_time in sweep_file_paths:
                                        del sweep_file_paths[sweep_time]
                                    continue
                                
                                # Read original feather to get offset_ns values for computing time_to_center_adjustment
                                orig_table = feather.read_table(str(orig_pc_path))
                                orig_offset_ns = np.array(orig_table['offset_ns'].to_pylist())
                                orig_laser_number = np.array(orig_table['laser_number'].to_pylist())
                                
                                # Separate up and down lidar points (up: 0-31, down: 32-63)
                                up_mask = orig_laser_number < 32
                                down_mask = orig_laser_number >= 32
                                original_time_full_up = orig_offset_ns[up_mask]
                                original_time_full_down = orig_offset_ns[down_mask]
                                
                                # Compute time_to_center_adjustment from original feather (same way splatad computes it)
                                if len(original_time_full_up) > 0:
                                    max_offset_up = original_time_full_up.max()
                                    min_offset_up = original_time_full_up.min()
                                    time_to_center_adjustment_up = (max_offset_up + min_offset_up) / 2.0
                                else:
                                    time_to_center_adjustment_up = 0.0
                                
                                if len(original_time_full_down) > 0:
                                    max_offset_down = original_time_full_down.max()
                                    min_offset_down = original_time_full_down.min()
                                    time_to_center_adjustment_down = (max_offset_down + min_offset_down) / 2.0
                                else:
                                    time_to_center_adjustment_down = 0.0
                                
                                # Convert adjustment from nanoseconds to nanoseconds (already in ns, just cast)
                                up_adj_ns = np.int64(time_to_center_adjustment_up)
                                down_adj_ns = np.int64(time_to_center_adjustment_down)
                                
                                # Load calibration to transform SENSOR -> EGO coordinates
                                if has_av2:
                                    log_dir = orig_pc_path.parent.parent.parent
                                    sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
                                    ego_SE3_up_lidar = sensor_name_to_pose["up_lidar"]
                                    ego_SE3_down_lidar = sensor_name_to_pose["down_lidar"]
                                    
                                    # Transform points from SENSOR to EGO frame
                                    # ego_SE3_up_lidar is sensor->ego transform
                                    up_xyz_ego = transform_points(
                                        up["xyz"], 
                                        torch.from_numpy(ego_SE3_up_lidar.transform_matrix).float()
                                    )
                                    down_xyz_ego = transform_points(
                                        down["xyz"],
                                        torch.from_numpy(ego_SE3_down_lidar.transform_matrix).float()
                                    )
                                else:
                                    CONSOLE.print("[red]Error: Cannot transform coordinates without av2 library. Skipping.[/red]")
                                    del sweep_buffer[sweep_time]
                                    if sweep_time in sweep_file_paths:
                                        del sweep_file_paths[sweep_time]
                                    continue
                                
                                # Combine arrays in EGO coordinates
                                xyz_cat = torch.cat([up_xyz_ego, down_xyz_ego], dim=0).numpy()
                                intensity_cat = torch.cat([up["intensity"], down["intensity"]], dim=0).numpy().astype(np.uint8)
                                
                                # Add adjustment to offset_ns (offset_ns is already in nanoseconds)
                                offset_ns_up = up["offset_ns"].numpy() + up_adj_ns
                                offset_ns_down = down["offset_ns"].numpy() + down_adj_ns
                                offset_ns_raw = np.concatenate([offset_ns_up, offset_ns_down])
                                
                                laser_num_cat = torch.cat([up["laser_number"], down["laser_number"]], dim=0).numpy()

                                # Convert to correct data types to match original feather format
                                xyz_cat = xyz_cat.astype(np.float16)  # halffloat to match original
                                laser_num_cat = laser_num_cat.astype(np.uint8)  # uint8 to match original

                                # Build target path using exact timestamp from original filename
                                ts_ns = int(orig_pc_path.stem)
                                log_id = str(getattr(data_manager_config.dataparser, "sequence"))
                                # Use sensors/lidar structure to match AV2 format
                                tgt_pc_path = (self.target_data_root / "sensor" / split / log_id / "sensors" / "lidar" / f"{ts_ns}.feather")
                                tgt_pc_path.parent.mkdir(parents=True, exist_ok=True)

                                # Clone original feather and replace key columns
                                table = feather.read_table(str(orig_pc_path))
                                offset_ns_cat = offset_ns_raw.astype(np.int32)
                                # Read original point cloud from x, y, z columns
                                gt_x = np.array(table['x'].to_pylist())
                                gt_y = np.array(table['y'].to_pylist())
                                gt_z = np.array(table['z'].to_pylist())
                                gt_point_cloud = np.stack([gt_x, gt_y, gt_z], axis=1)
                                
                                n_rows_orig = table.num_rows
                                n_rows_new = len(xyz_cat)
                                
                                # Reduce to minimum: truncate both table and arrays to match
                                n_rows = min(n_rows_orig, n_rows_new)
                                
                                # Slice table to reduced size
                                if n_rows < n_rows_orig:
                                    table = table.slice(0, n_rows)
                                
                                # Truncate arrays to match
                                xyz_cat = xyz_cat[:n_rows]
                                intensity_cat = intensity_cat[:n_rows]
                                offset_ns_cat = offset_ns_cat[:n_rows]
                                laser_num_cat = laser_num_cat[:n_rows]
                                cols = table.column_names
                                name_map = {name: idx for idx, name in enumerate(cols)}
                                def ensure_col(name: str, arr: np.ndarray, dtype=None):
                                    nonlocal table
                                    if dtype is not None:
                                        tensor = pa.array(arr, type=dtype)
                                    else:
                                        tensor = pa.array(arr)
                                    if name in name_map:
                                        table = table.set_column(name_map[name], name, tensor)
                                    else:
                                        table = table.append_column(name, tensor)

                                ensure_col("x", xyz_cat[:, 0])
                                ensure_col("y", xyz_cat[:, 1])
                                ensure_col("z", xyz_cat[:, 2])
                                ensure_col("intensity", intensity_cat)
                                ensure_col("offset_ns", offset_ns_cat, dtype=pa.int32())  # Explicitly set int32 to match original
                                ensure_col("laser_number", laser_num_cat)

                                feather.write_feather(table, str(tgt_pc_path))
                                # clear buffer for this timestamp
                                del sweep_buffer[sweep_time]
                                if sweep_time in sweep_file_paths:
                                    del sweep_file_paths[sweep_time]

                                # if vis_flag:
                                #     plot_lidar_points(
                                #         gt_point_cloud, vis_gt_output_path / f"gt-lidar_{lidar_name}.png"
                                #     )
                                #     plot_lidar_points(
                                #         xyz_cat, vis_output_path / f"lidar_{lidar_name}.png"
                                #     )
                                #     vis_flag = False

                            # import numpy as np
                            # np.save(output_path / lidar_name, median_point_cloud.cpu().numpy())

                            # if vis_flag:
                            #     plot_lidar_points(
                            #         gt_point_cloud.cpu().detach().numpy(), vis_gt_output_path / f"gt-lidar_{lidar_name}.png"
                            #     )
                            #     plot_lidar_points(
                            #         median_point_cloud.cpu().detach().numpy(), vis_output_path / f"lidar_{lidar_name}.png"
                            #     )
                            #     vis_flag = False

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.pose_source.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))


def plot_lidar_points(points, output_path, cmin=-6.0, cmax=5.0, width=1920, height=1080, ranges=[100, 200, 10]):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a 3D scatter plot
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=1.0,
            color=z,
            colorscale="Viridis",
            opacity=0.8,
            cmin=cmin,
            cmax=cmax,
        ),
    )

    x_range, y_range, z_range = ranges

    # Compute the aspect ratio
    max_range = 2 * max(x_range, y_range, z_range)
    aspect_ratio = dict(x=x_range / max_range, y=y_range / max_range, z=z_range / max_range)

    # Define the camera position
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=-0.07, z=0.02),
    )
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title="",
                range=[-x_range, x_range],
                showticklabels=False,
                ticks="",
                showline=False,
                showgrid=False,
            ),
            yaxis=dict(
                title="",
                range=[-y_range, y_range],
                showticklabels=False,
                ticks="",
                showline=False,
                showgrid=False,
            ),
            zaxis=dict(
                title="",
                range=[-z_range, z_range],
                showticklabels=False,
                ticks="",
                showline=False,
                showgrid=False,
            ),
            aspectmode="manual",
            aspectratio=aspect_ratio,
            camera=camera,
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.write_image(output_path, width=width, height=height, scale=1)


def streamline_ad_config(config):
    if getattr(config.pipeline.datamanager, "num_processes", None):
        config.pipeline.datamanager.num_processes = 0
    config.pipeline.model.eval_num_rays_per_chunk = 2**17
    if getattr(config.pipeline.datamanager.dataparser, "add_missing_points", None):
        config.pipeline.datamanager.dataparser.add_missing_points = False
    return config


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ShiftedDatasetRender, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
