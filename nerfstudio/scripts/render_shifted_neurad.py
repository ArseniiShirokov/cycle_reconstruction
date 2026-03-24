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
render.py
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
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command


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

    pose_source: Literal["train", "val", "test", "train+test", "train+val"] = "test"
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
            
            dataset.cameras.camera_to_worlds[..., :3, 3] += torch.tensor(self.shift, dtype=torch.float32)

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
                    # Try to get the original filename
                    image_name = (
                        Path(dataparser_outputs.image_filenames[camera_idx]).with_suffix("").relative_to(images_root)
                    )

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

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if "all" in rendered_output_names:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    elif rendered_output_names == ["none"]:
                        rendered_output_names = []
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1

                        output_path = self.output_path / split / rendered_output_name / image_name
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
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
                        if is_raw:
                            with gzip.open(output_path.parent / (output_path.name + ".npy.gz"), "wb") as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(output_path.parent / (output_path.name + ".png"), output_image, fmt="png")
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.parent / (output_path.name + ".jpg"),
                                output_image,
                                fmt="jpeg",
                                quality=self.jpeg_quality,
                            )
                        else:
                            raise ValueError(f"Unknown image format {self.image_format}")

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
                    with torch.no_grad():
                        output_path = self.output_path / split / "lidar"
                        output_path.mkdir(exist_ok=True, parents=True)

                        vis_output_path = self.output_path / split / "lidar_vis"
                        vis_output_path.mkdir(exist_ok=True, parents=True)

                        vis_gt_output_path = self.output_path / split / "lidar_vis_gt"
                        vis_gt_output_path.mkdir(exist_ok=True, parents=True)

                        for lidar_idx, (lidar, batch) in enumerate(
                            progress.track(lidar_dataloader, total=len(lidar_dataloader))
                        ):
                            device = lidar.lidar_to_worlds[:, :3, 3].device
                            lidar.lidar_to_worlds[:, :3, 3] += torch.tensor(self.shift, dtype=torch.float32).to(device)

                            lidar_output, _ = pipeline.model.get_outputs_for_lidar(lidar, batch=batch)
                            points_in_local = lidar_output["points"]
                            intensity = lidar_output["intensity"]
                            time = lidar_output["time"]

                            if "ray_drop_prob" in lidar_output:
                                mask = (lidar_output["ray_drop_prob"] < 0.5).squeeze(-1)
                                points_in_local = points_in_local[mask]
                                intensity = intensity[mask]
                                time = time[mask]

                            from nerfstudio.utils.pandaset_lidar_utils import save_pandaset_lidar
                            lidar_name = f"{lidar_idx:02d}"
                            save_pandaset_lidar(points_in_local, intensity, time, output_path / lidar_name)

                            # plot_lidar_points(
                            #     batch["lidar"][..., :3].cpu().detach().numpy(), vis_gt_output_path / f"gt-lidar_{lidar_name}.png"
                            # )
                            # plot_lidar_points(
                            #     points_in_local.cpu().detach().numpy(), vis_output_path / f"lidar_{lidar_name}.png"
                            # )

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
