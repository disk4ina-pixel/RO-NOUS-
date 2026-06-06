# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:03:44 2025
Updated on Fri Apr 03 16:20:26 2026

@author: Chaim
"""
# __init__.py
"""
RO-NOUS Clustering Package

This package provides tools for clustering point cloud data with
OpenCV integration and 3D grid-based analysis.
"""

__version__ = '0.1.0'

# Import main components for easier access
from . import point_cloud
from . import clustering
from . import opencv_integration
from . import grid
from . import visualization
from . import utils

# Import specific common functions for convenience
from .point_cloud import load_point_cloud, extract_frame_number, find_matching_frame
from .clustering import apply_dbscan, apply_hdbscan, filter_significant_clusters, assign_cluster_ids, apply_grid_clustering
from .opencv_integration import enhance_clustering_with_opencv, project_3d_to_2d
from .grid import OccupancyGrid
from .utils import setup_logging, save_clusters_to_files, save_cluster_ids_to_files

# Setup default logging
utils.setup_logging()