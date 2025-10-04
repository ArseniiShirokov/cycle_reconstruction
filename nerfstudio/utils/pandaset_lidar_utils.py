"""Utilities for saving LiDAR data in PandaSet format."""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

# Constants from PandaSet data parser
MAX_REFLECTANCE_VALUE = 255.0
LIDAR_NAME_TO_INDEX = {
    "Pandar64": 0,
    "PandarGT": 1,
}


def save_pandaset_lidar(
    points: Union[torch.Tensor, np.ndarray],
    intensity: Union[torch.Tensor, np.ndarray],
    time_offset: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    sensor_idx: int = 0,
    normalize_intensity: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save simulated LiDAR points in PandaSet format.
    
    Args:
        points: Point cloud coordinates of shape [N, 3] containing x, y, z
        intensity: Intensity values of shape [N] 
        time_offset: Time offset values of shape [N] (relative to frame timestamp)
        output_path: Output file path (will save as .pkl file)
        sensor_idx: Sensor index (0 for Pandar64, 1 for PandarGT)
        normalize_intensity: Whether to normalize intensity values to [0, 1] range
        metadata: Optional metadata dictionary to save alongside the data
        
    Raises:
        ValueError: If input dimensions don't match
        OSError: If unable to write to output path
    """
    points = points.detach().cpu().numpy()
    intensity = intensity.detach().cpu().numpy()
    time_offset = time_offset.detach().cpu().numpy()
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add .pkl.gz extension if not present
    if not output_path.suffix:
        output_path = output_path.with_suffix('.pkl.gz')
    elif output_path.suffix == '.pkl':
        output_path = output_path.with_suffix('.pkl.gz')
    
    # Normalize intensity if requested
    if normalize_intensity:
        # Scale intensity values to match PandaSet format (0-255 range, then normalized to 0-1)
        if intensity.max() <= 1.0:
            # Assume intensity is already in [0, 1] range
            intensity_scaled = intensity * MAX_REFLECTANCE_VALUE
        else:
            # Scale to [0, 255] range
            intensity_scaled = (intensity / intensity.max()) * MAX_REFLECTANCE_VALUE
        # Then normalize back to [0, 1] as expected by the data parser
        intensity_normalized = intensity_scaled / MAX_REFLECTANCE_VALUE
    else:
        intensity_normalized = intensity
    
    # Create sensor index column
    sensor_indices = np.full(points.shape[0], sensor_idx, dtype=np.float32)
    
    # Combine all data: [x, y, z, intensity, time_offset, sensor_idx]
    lidar_data = np.hstack([
        points.astype(np.float32),
        intensity_normalized.reshape(-1, 1).astype(np.float32),
        time_offset.reshape(-1, 1).astype(np.float32),
        sensor_indices.reshape(-1, 1)
    ])
    
    # Create pandas DataFrame (PandaSet format uses this structure)
    df = pd.DataFrame(lidar_data, columns=['x', 'y', 'z', 'intensity', 'time_offset', 'sensor_idx'])
    
    # Add metadata if provided
    if metadata is not None:
        df.attrs.update(metadata)
    
    # Save as compressed pickle file (PandaSet format)
    # Calculate azimuth and elevation for validation
    # NOTE: For world coordinates, this assumes points are relative to origin
    # In true world coordinates, points should be distributed around actual world positions
    # print(f"World coordinate statistics:")
    # print(f"  X range: [{points[:, 0].min():.1f}m to {points[:, 0].max():.1f}m]")
    # print(f"  Y range: [{points[:, 1].min():.1f}m to {points[:, 1].max():.1f}m]") 
    # print(f"  Z range: [{points[:, 2].min():.1f}m to {points[:, 2].max():.1f}m]")
    # print(f"  Points near origin (within 5m): {(np.linalg.norm(points[:, :3], axis=1) < 5.0).sum()}")
    
    azimuth = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
    distance = np.linalg.norm(points[:, :3], axis=1)
    elevation = np.rad2deg(np.arcsin(points[:, 2] / distance))
    
    try:
        # Use compression='gzip' to match PandaSet format
        df.to_pickle(output_path, compression='gzip')
        # print(f"Saved LiDAR data to {output_path}")
        # print(f"Data shape: {df.shape}")
        # print(f"Intensity range: [{df['intensity'].min():.3f}, {df['intensity'].max():.3f}]")
        # print(f"Time offset range: [{df['time_offset'].min():.3f}, {df['time_offset'].max():.3f}]")
        
        # # Azimuth and elevation validation
        # print(f"Azimuth range: [{azimuth.min():.1f}° to {azimuth.max():.1f}°]")
        # print(f"Elevation range: [{elevation.min():.1f}° to {elevation.max():.1f}°]")
        # print(f"Distance range: [{distance.min():.1f}m to {distance.max():.1f}m]")
        
        # Check for proper 360° coverage
        azimuth_span = azimuth.max() - azimuth.min()
        elevation_span = elevation.max() - elevation.min()
        
        # print(f"Azimuth span: {azimuth_span:.1f}° (should be ~360° for full coverage)")
        # print(f"Elevation span: {elevation_span:.1f}° (typical lidar: 30-40°)")
        
        # Distribution analysis
        azimuth_bins = np.histogram(azimuth, bins=36)[0]  # 10° bins
        elevation_bins = np.histogram(elevation, bins=20)[0]  # ~2° bins
        
        # print(f"Azimuth distribution uniformity: {azimuth_bins.std()/azimuth_bins.mean():.2f} (lower is more uniform)")
        # print(f"Elevation distribution uniformity: {elevation_bins.std()/elevation_bins.mean():.2f}")
        
        # Warning checks
        # if azimuth_span < 300:
        #     print(f"WARNING: Limited azimuth coverage ({azimuth_span:.1f}°). Expected ~360° for full lidar scan.")
        
        # if elevation_span < 20:
        #     print(f"WARNING: Limited elevation coverage ({elevation_span:.1f}°). Expected 20-40° for typical lidar.")
            
        # Check for concentration (like your issue)
        unique_azimuths = len(np.unique(np.round(azimuth, 1)))
        unique_elevations = len(np.unique(np.round(elevation, 1)))
        
        # if unique_azimuths < 100:  # Less than 100 unique azimuth values (suspicious)
        #     print(f"WARNING: Low azimuth diversity ({unique_azimuths} unique values). Points may be concentrated.")
            
        # if unique_elevations < 10:  # Less than 10 unique elevation values
        #     print(f"WARNING: Low elevation diversity ({unique_elevations} unique values). Points may be concentrated.")
        
        # print()
        
    except Exception as e:
        raise OSError(f"Failed to save LiDAR data to {output_path}: {e}")
