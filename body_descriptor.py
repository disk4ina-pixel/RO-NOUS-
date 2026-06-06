# -*- coding: utf-8 -*-
"""
body_descriptor.py

Body descriptor computation for the ro_nous pipeline.
Computes per-frame body metrics, pose classification, movement classification,
and limb sub-cluster analysis from JSON results.  Writes
frame_NNN_body_descriptor.json.

PATCH 2026-03-18 (ro_nous repair plan):
- BD-1: load_frame_data() facing_direction default changed from 'toward_camera'
         to 'unknown'.  The old default masked frames where facing was genuinely
         ambiguous or where results_s.json predates the facing_direction field.
         All downstream body_descriptor logic (classify_pose, classify_movement,
         compute_body_metrics) does NOT branch on facing_direction — it is
         metadata only — so this change is safe and improves audit accuracy.

Updated on Fri Apr 03 16:20:26 2026      
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON helper (same as skeleton_fitting.py)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 1)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [round(float(v), 1) if isinstance(v, (float, np.floating)) else v
                    for v in obj.tolist()]
        return super().default(obj)


def _round_for_json(obj, decimals=1):
    """Recursively round all floats in a nested dict/list to *decimals* places."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_for_json(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_for_json(v, decimals) for v in obj]
    return obj


# ===========================================================================
# SECTION 1 — FRAME DATA LOADER
# ===========================================================================

def _load_frame_json(results_dir: str, frame_num: int, suffix: str) -> Optional[Dict]:
    """Load frame_NNN_<suffix>.json, return parsed dict or None."""
    for fmt in (f'frame_{frame_num:03d}_{suffix}.json',
                f'frame_{frame_num:04d}_{suffix}.json'):
        path = os.path.join(results_dir, fmt)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def load_frame_data(results_dir: str, frame_num: int) -> Optional[Dict]:
    """
    Step 2 — FrameDataLoader.

    Merge results.json + results_s.json into one dict for a single frame.
    results_p.json is optional (written by Phase 3, may not exist yet).

    Returns None if results.json is missing (the minimum requirement for
    cluster geometry).
    """
    r = _load_frame_json(results_dir, frame_num, 'results')
    if r is None:
        return None

    merged = {
        'frame': frame_num,
        # --- from results.json ---
        'person_cluster_uuid': r.get('person_cluster_uuid'),
        'clusters_info': r.get('clusters_info', {}),
        'facing_info': r.get('facing_info'),
        'pose_3d': r.get('pose_3d'),
    }

    # Extract person cluster data
    label = merged['person_cluster_uuid']
    c_info = merged['clusters_info']
    person_data = c_info.get(label, {}) if label else {}
    merged['bbox'] = person_data.get('bbox')
    merged['actual_uuid'] = person_data.get('cluster_uuid')
    merged['voxel_data'] = person_data.get('voxel_data', {})
    merged['cluster_point_count'] = person_data.get('point_count', 0)

    # --- from results_s.json (Phase 2 output) ---
    s = _load_frame_json(results_dir, frame_num, 'results_s')
    if s is not None:
        merged['skeleton_type'] = s.get('skeleton_type', 'none')
        merged['skeleton_height_cm'] = s.get('skeleton_height_cm', 170.0)
        # BD-1 FIX: default to 'unknown' not 'toward_camera'.
        # 'toward_camera' as a default masked frames where facing was genuinely
        # ambiguous or where results_s.json predates the facing_direction field.
        # Downstream body_descriptor logic does not branch on facing_direction
        # so this change is safe; it improves audit-trail accuracy.
        merged['facing_direction'] = s.get('facing_direction', 'unknown')
        merged['grid_origin'] = s.get('grid_origin')
        merged['grid_resolution_cm'] = s.get('grid_resolution_cm', 2.0)
        merged['joints'] = s.get('joints', [])
        merged['avg_fitting_error_cm'] = s.get('avg_fitting_error_cm', 0.0)
        merged['iccs_origin_world'] = s.get('iccs_origin_world')
    else:
        merged['skeleton_type'] = 'none'
        merged['joints'] = []

    return merged


# ===========================================================================
# SECTION 2 — VOXEL GEOMETRY FROM JSON (no live grid needed)
# ===========================================================================

def _parse_voxel_keys(voxel_data: Dict) -> List[Tuple[int, int, int]]:
    """Parse string keys like '(41, 16, 9)' from voxel_data dict."""
    result = []
    for key in voxel_data.keys():
        try:
            clean = str(key).strip('() ')
            parts = clean.split(',')
            if len(parts) == 3:
                result.append((int(parts[0].strip()),
                               int(parts[1].strip()),
                               int(parts[2].strip())))
        except (ValueError, AttributeError):
            continue
    return result


def _compute_y_wall_stats_from_voxels(
        voxel_indices: List[Tuple[int, int, int]],
        grid_origin: List[float],
        resolution: float = 2.0,
) -> Dict:
    """
    Compute Y-wall statistics from voxel indices without a live grid.

    Returns dict with:
        'y_wall_count'     : int — number of distinct Y-planes with ≥1 voxel
        'y_levels'         : list of float — world-Y of each occupied plane
        'per_y_contour'    : dict {y_world: {'x_extent', 'z_extent', 'area', 'centroid', 'count'}}
        'total_voxels'     : int
        'y_range'          : (min_y, max_y) in world coords
        'z_range'          : (min_z, max_z) in world coords — the height span
    """
    if not voxel_indices or grid_origin is None:
        return {
            'y_wall_count': 0, 'y_levels': [], 'per_y_contour': {},
            'total_voxels': 0, 'y_range': (0, 0), 'z_range': (0, 0),
        }

    ox, oy, oz = grid_origin
    res = resolution

    # Group by Y index
    by_y: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for vi in voxel_indices:
        by_y[vi[1]].append(vi)

    y_levels = []
    per_y = {}
    all_z = []

    for y_idx in sorted(by_y.keys()):
        cells = by_y[y_idx]
        y_world = oy + (y_idx + 0.5) * res
        y_levels.append(y_world)

        xs = np.array([ox + (c[0] + 0.5) * res for c in cells])
        zs = np.array([oz + (c[2] + 0.5) * res for c in cells])
        all_z.extend(zs.tolist())

        x_ext = float(xs.max() - xs.min()) if len(xs) > 1 else res
        z_ext = float(zs.max() - zs.min()) if len(zs) > 1 else res

        per_y[round(y_world, 2)] = {
            'x_extent': x_ext,
            'z_extent': z_ext,
            'x_min': float(xs.min()),
            'x_max': float(xs.max()),
            'area': len(cells) * res * res,
            'centroid': [float(xs.mean()), float(zs.mean())],
            'count': len(cells),
        }

    y_arr = np.array(y_levels) if y_levels else np.array([0.0])
    z_arr = np.array(all_z) if all_z else np.array([0.0])

    return {
        'y_wall_count': len(y_levels),
        'y_levels': y_levels,
        'per_y_contour': per_y,
        'total_voxels': len(voxel_indices),
        'y_range': (float(y_arr.min()), float(y_arr.max())),
        'z_range': (float(z_arr.min()), float(z_arr.max())),
    }


# ===========================================================================
# SECTION 3 — BODY METRICS (Step 7)
# ===========================================================================

def compute_body_metrics(window_stats: List[Dict]) -> Dict:
    """
    Compute stable body metrics from a 5-frame window of Y-wall stats.

    Parameters
    ----------
    window_stats : list of dicts from _compute_y_wall_stats_from_voxels,
                   one per frame in the window.

    Returns
    -------
    dict with shoulder_width, hip_width, sh_ratio, height, volume,
    fat_thin, male_female, volume_variance, avg_area.
    """
    if not window_stats or all(ws['y_wall_count'] == 0 for ws in window_stats):
        return {'valid': False}

    # Collect cross-section data across the window
    all_x_extents = []
    all_z_extents = []
    all_areas = []
    heights = []
    volumes = []

    for ws in window_stats:
        if ws['y_wall_count'] == 0:
            continue

        z_min, z_max = ws['z_range']
        height = z_max - z_min
        if height > 0:
            heights.append(height)

        frame_areas = []
        for y_key, contour in ws['per_y_contour'].items():
            all_x_extents.append(contour['x_extent'])
            all_z_extents.append(contour['z_extent'])
            all_areas.append(contour['area'])
            frame_areas.append(contour['area'])

        if frame_areas:
            # Volume ≈ sum of cross-section areas × slice thickness
            res = 2.0  # cm
            volumes.append(sum(frame_areas) * res)

    if not heights:
        return {'valid': False}

    avg_height = float(np.mean(heights))

    # Shoulder width: global X-span of all voxels at top 25-35% of height
    # Hip width: global X-span of all voxels at 45-55% of height
    # Aggregate x_min/x_max across ALL Y-planes in the Z-band to get
    # true lateral width, not just the width within a single depth slice.
    shoulder_x_mins = []
    shoulder_x_maxs = []
    hip_x_mins = []
    hip_x_maxs = []

    for ws in window_stats:
        if ws['y_wall_count'] == 0:
            continue
        z_min, z_max = ws['z_range']
        h = z_max - z_min
        if h <= 0:
            continue
        shoulder_z_lo = z_max - 0.35 * h
        shoulder_z_hi = z_max - 0.25 * h
        hip_z_lo = z_max - 0.55 * h
        hip_z_hi = z_max - 0.45 * h

        for y_key, contour in ws['per_y_contour'].items():
            cz = contour['centroid'][1]  # Z component
            if shoulder_z_lo <= cz <= shoulder_z_hi:
                shoulder_x_mins.append(contour['x_min'])
                shoulder_x_maxs.append(contour['x_max'])
            if hip_z_lo <= cz <= hip_z_hi:
                hip_x_mins.append(contour['x_min'])
                hip_x_maxs.append(contour['x_max'])

    shoulder_width = (max(shoulder_x_maxs) - min(shoulder_x_mins)) if shoulder_x_mins else 40.0
    hip_width = (max(hip_x_maxs) - min(hip_x_mins)) if hip_x_mins else 35.0
    sh_ratio = shoulder_width / hip_width if hip_width > 0 else 1.0

    avg_area = float(np.mean(all_areas)) if all_areas else 0.0
    avg_volume = float(np.mean(volumes)) if volumes else 0.0
    vol_variance = float(np.var(volumes)) if len(volumes) > 1 else 0.0

    # Classification thresholds (area is now voxel-count × res², not bbox)
    male_female = 'male' if sh_ratio > 1.05 else 'female'
    fat_thin = 'average'
    if avg_area > 200:
        fat_thin = 'heavy'
    elif avg_area < 80:
        fat_thin = 'thin'

    return {
        'valid': True,
        'height_cm': round(avg_height, 1),
        'shoulder_width_cm': round(shoulder_width, 1),
        'hip_width_cm': round(hip_width, 1),
        'shoulder_hip_ratio': round(sh_ratio, 3),
        'male_female': male_female,
        'fat_thin': fat_thin,
        'avg_cross_section_area': round(avg_area, 1),
        'avg_volume_cm3': round(avg_volume, 0),
        'volume_variance': round(vol_variance, 1),
    }


# ===========================================================================
# SECTION 4 — POSE CLASSIFICATION (Step 8)
# ===========================================================================

def classify_pose(window_stats: List[Dict], body_metrics: Dict) -> str:
    """
    Classify pose from the 5-frame window.

    Returns 'standing', 'sitting', or 'transition'.
    """
    if not body_metrics.get('valid'):
        return 'unknown'

    heights = []
    for ws in window_stats:
        z_min, z_max = ws.get('z_range', (0, 0))
        h = z_max - z_min
        if h > 0:
            heights.append(h)

    if not heights:
        return 'unknown'

    avg_h = np.mean(heights)
    h_variance = np.var(heights) if len(heights) > 1 else 0.0
    expected_h = body_metrics.get('height_cm', 170.0)

    # Transition: height changing rapidly across the window
    if h_variance > 100:  # >10cm std dev
        return 'transition'

    # Sitting: cluster height < 65% of expected standing height
    if avg_h < expected_h * 0.65:
        return 'sitting'

    return 'standing'


# ===========================================================================
# SECTION 5 — MOVEMENT CLASSIFICATION (Step 9)
# ===========================================================================

def classify_movement(window_stats: List[Dict]) -> str:
    """
    Classify movement pattern from the 5-frame window.

    Returns 'static', 'translating', 'articulating', or 'occlusion'.
    """
    valid = [ws for ws in window_stats if ws['y_wall_count'] > 0]
    if len(valid) < 2:
        return 'unknown'

    # Check for occlusion: sudden drop in Y-wall count
    counts = [ws['y_wall_count'] for ws in valid]
    if min(counts) < max(counts) * 0.5:
        return 'occlusion'

    # Centroid displacement across frames
    centroids = []
    for ws in valid:
        all_cx = []
        all_cz = []
        for contour in ws['per_y_contour'].values():
            all_cx.append(contour['centroid'][0])
            all_cz.append(contour['centroid'][1])
        if all_cx:
            centroids.append([np.mean(all_cx), np.mean(all_cz)])

    if len(centroids) < 2:
        return 'static'

    centroids = np.array(centroids)
    displacements = np.sqrt(np.sum(np.diff(centroids, axis=0) ** 2, axis=1))
    total_displacement = float(np.sum(displacements))
    max_displacement = float(np.max(displacements))

    # Volume stability
    volumes = []
    for ws in valid:
        areas = [c['area'] for c in ws['per_y_contour'].values()]
        volumes.append(sum(areas) * 2.0)
    vol_cv = float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0

    # Classification
    if total_displacement > 30:  # >30cm total movement across 5 frames
        return 'translating'

    # Y-wall distribution change → articulating
    ywall_counts = [ws['y_wall_count'] for ws in valid]
    ywall_cv = float(np.std(ywall_counts) / np.mean(ywall_counts)) if np.mean(ywall_counts) > 0 else 0
    if ywall_cv > 0.15 or vol_cv > 0.15:
        return 'articulating'

    return 'static'


# ===========================================================================
# SECTION 6 — LIMB SUB-CLUSTERING (Step 10)
# ===========================================================================

def compute_limb_subclusters(window_stats: List[Dict],
                             body_metrics: Dict) -> Dict:
    """
    Identify Y-level bands that reliably contain two XZ sub-blobs.

    Returns dict with 'upper_band' and 'lower_band', each containing
    sub-cluster centroid pairs if detected, or None.
    """
    if not body_metrics.get('valid'):
        return {'upper_band': None, 'lower_band': None}

    # Define bands by Z-range relative to body height
    z_ranges = []
    for ws in window_stats:
        z_min, z_max = ws.get('z_range', (0, 0))
        if z_max > z_min:
            z_ranges.append((z_min, z_max))
    if not z_ranges:
        return {'upper_band': None, 'lower_band': None}

    avg_z_min = np.mean([z[0] for z in z_ranges])
    avg_z_max = np.mean([z[1] for z in z_ranges])
    h = avg_z_max - avg_z_min
    if h <= 0:
        return {'upper_band': None, 'lower_band': None}

    # Lower band: bottom 40% of height (legs)
    lower_z_lo = avg_z_min
    lower_z_hi = avg_z_min + 0.4 * h

    # Upper band: top 30-50% of height (arms/shoulders)
    upper_z_lo = avg_z_max - 0.5 * h
    upper_z_hi = avg_z_max - 0.3 * h

    def _find_two_blobs_in_band(z_lo, z_hi):
        """Check if contours in this Z-band split into two XZ blobs."""
        x_centroids = []
        for ws in window_stats:
            for y_key, contour in ws.get('per_y_contour', {}).items():
                cz = contour['centroid'][1]
                if z_lo <= cz <= z_hi:
                    x_centroids.append(contour['centroid'][0])

        if len(x_centroids) < 4:
            return None

        x_arr = np.array(x_centroids)
        x_mid = np.median(x_arr)
        left = x_arr[x_arr < x_mid - 2.0]  # >2cm gap
        right = x_arr[x_arr > x_mid + 2.0]

        if len(left) >= 2 and len(right) >= 2:
            return {
                'left_centroid_x': float(np.mean(left)),
                'right_centroid_x': float(np.mean(right)),
                'separation_cm': float(np.mean(right) - np.mean(left)),
            }
        return None

    return {
        'lower_band': _find_two_blobs_in_band(lower_z_lo, lower_z_hi),
        'upper_band': _find_two_blobs_in_band(upper_z_lo, upper_z_hi),
        'lower_z_range': [round(lower_z_lo, 1), round(lower_z_hi, 1)],
        'upper_z_range': [round(upper_z_lo, 1), round(upper_z_hi, 1)],
    }


# ===========================================================================
# SECTION 7 — Y-WALL COMPARISON: SHELL vs HUMANOID (Step 6)
# ===========================================================================

def count_ywall_comparison(
        window_stats: List[Dict],
        skeleton_joints: List[Dict],
        body_metrics: Dict,
) -> Dict:
    """
    Compare shell Y-wall count vs expected humanoid Y-wall count per region.

    Uses joint Z-coordinates to define limb regions, then counts how many
    Y-planes fall within each region.

    Returns dict {region_name: {'shell': int, 'ratio': float}}.
    """
    if not body_metrics.get('valid') or not skeleton_joints:
        return {}

    # Build limb Z-ranges from joint world_coords
    def _jt_z(idx):
        for jt in skeleton_joints:
            if jt.get('joint_idx') == idx and jt.get('valid'):
                wc = jt.get('world_coords', [0, 0, 0])
                return wc[2]
        return None

    # Define regions by joint pairs
    regions = {}
    pairs = [
        ('head',       0,  17),  # nose to head_center
        ('torso',      18, 19),  # shoulder_center to pelvis_center
        ('left_thigh', 11, 13),  # left_hip to left_knee
        ('right_thigh', 12, 14),
        ('left_shin',  13, 15),  # left_knee to left_ankle
        ('right_shin', 14, 16),
        ('left_upper_arm', 5, 7),   # left_shoulder to left_elbow
        ('right_upper_arm', 6, 8),
        ('left_forearm', 7, 9),
        ('right_forearm', 8, 10),
    ]

    for name, idx_a, idx_b in pairs:
        za = _jt_z(idx_a)
        zb = _jt_z(idx_b)
        if za is not None and zb is not None:
            regions[name] = {'z_min': min(za, zb), 'z_max': max(za, zb)}

    if not regions:
        return {}

    # Compute average total Y-wall count and total Z-span across window
    total_yw_counts = []
    total_z_spans = []
    for ws in window_stats:
        yw_count = ws.get('y_wall_count', 0)
        if yw_count > 0:
            total_yw_counts.append(yw_count)
            z_min, z_max = ws.get('z_range', (0, 0))
            z_span = z_max - z_min
            if z_span > 0:
                total_z_spans.append(z_span)

    avg_total_yw = float(np.mean(total_yw_counts)) if total_yw_counts else 0
    avg_total_z = float(np.mean(total_z_spans)) if total_z_spans else 1.0

    # Count shell Y-walls per region (average across window)
    result = {}
    for region_name, zr in regions.items():
        z_lo, z_hi = zr['z_min'], zr['z_max']
        shell_counts = []
        for ws in window_stats:
            count = 0
            for y_key, contour in ws.get('per_y_contour', {}).items():
                cz = contour['centroid'][1]
                if z_lo <= cz <= z_hi:
                    count += 1
            shell_counts.append(count)

        avg_shell = float(np.mean(shell_counts)) if shell_counts else 0
        # Expected: proportional share of total Y-walls based on
        # this region's Z-span relative to total body Z-span.
        region_z_span = z_hi - z_lo
        expected = avg_total_yw * (region_z_span / avg_total_z) if avg_total_z > 0 else 0.0
        ratio = avg_shell / expected if expected > 0 else 0.0

        result[region_name] = {
            'shell_ywalls': round(avg_shell, 1),
            'expected_humanoid': round(expected, 1),
            'ratio': round(ratio, 2),
        }

    return result


# ===========================================================================
# SECTION 8 — MAIN ENTRY POINT (Step 11)
# ===========================================================================

def run_descriptor_pass(
        results_dir: str,
        voxel_grids_by_frame: Optional[Dict] = None,
        state_bank=None,
) -> Dict[int, bool]:
    """
    Phase 2.5 entry point.  Called from run_clustering.py between Phase 2
    and Phase 3.

    For each frame:
      1. Build 5-frame sliding window of FrameDataLoader dicts
      2. Compute Y-wall stats from voxel data in JSONs
      3. Compute body metrics, pose class, movement class
      4. Compute limb sub-clusters and Y-wall comparison
      5. Write frame_NNN_body_descriptor.json
      6. Store in state_bank.body_descriptors_by_frame (if available)

    Parameters
    ----------
    results_dir : str — directory containing frame_NNN_results.json files
    voxel_grids_by_frame : optional {frame_num: EnhancedOccupancyGrid}
    state_bank : optional ClusterStateBank

    Returns
    -------
    Dict {frame_num: success_bool}
    """
    if not os.path.isdir(results_dir):
        logger.error(f"[BD] results_dir does not exist: {results_dir}")
        return {}

    # Discover frame numbers
    frame_nums = _discover_frame_nums(results_dir)
    if not frame_nums:
        logger.warning(f"[BD] No frame results found in {results_dir}")
        return {}

    logger.info(f"[BD] === Body Descriptor Pass starting — {len(frame_nums)} frames ===")

    # Load all frame data
    all_frames: Dict[int, Dict] = {}
    for fn in frame_nums:
        fd = load_frame_data(results_dir, fn)
        if fd is not None:
            all_frames[fn] = fd

    if not all_frames:
        logger.warning("[BD] No loadable frame data — aborting")
        return {}

    logger.info(f"[BD] Loaded {len(all_frames)}/{len(frame_nums)} frames")

    # Compute Y-wall stats per frame (from JSON voxel data — no live grid needed)
    yw_stats: Dict[int, Dict] = {}
    for fn, fd in all_frames.items():
        voxel_data = fd.get('voxel_data', {})
        grid_origin = fd.get('grid_origin')
        resolution = fd.get('grid_resolution_cm', 2.0)

        if not voxel_data and voxel_grids_by_frame and fn in voxel_grids_by_frame:
            # Fallback: use live grid if JSON voxel data is empty
            vg = voxel_grids_by_frame[fn]
            if hasattr(vg, 'occupied_cells') and hasattr(vg, 'bounds') and vg.bounds is not None:
                voxel_indices = list(vg.occupied_cells)
                grid_origin = [float(vg.bounds[0][i]) for i in range(3)]
                resolution = vg.resolution
                yw_stats[fn] = _compute_y_wall_stats_from_voxels(
                    voxel_indices, grid_origin, resolution)
                continue

        voxel_indices = _parse_voxel_keys(voxel_data)
        yw_stats[fn] = _compute_y_wall_stats_from_voxels(
            voxel_indices, grid_origin, resolution)

    # Ensure state_bank has body_descriptors_by_frame dict
    if state_bank is not None and not hasattr(state_bank, 'body_descriptors_by_frame'):
        state_bank.body_descriptors_by_frame = {}

    out_dir = results_dir
    results: Dict[int, bool] = {}
    sorted_frames = sorted(all_frames.keys())

    for center_idx, center_fn in enumerate(sorted_frames):
        # Build 5-frame window: center ±2
        window_fns = []
        for offset in range(-2, 3):
            idx = center_idx + offset
            if 0 <= idx < len(sorted_frames):
                window_fns.append(sorted_frames[idx])

        window_stats = [yw_stats[fn] for fn in window_fns if fn in yw_stats]
        window_frames = [all_frames[fn] for fn in window_fns if fn in all_frames]

        if not window_stats:
            results[center_fn] = False
            continue

        # Step 7: Body metrics
        metrics = compute_body_metrics(window_stats)

        # Step 8: Pose classification
        pose = classify_pose(window_stats, metrics)

        # Step 9: Movement classification
        movement = classify_movement(window_stats)

        # Step 10: Limb sub-clusters
        subclusters = compute_limb_subclusters(window_stats, metrics)

        # Step 6: Y-wall comparison
        center_fd = all_frames[center_fn]
        ywall_cmp = count_ywall_comparison(
            window_stats, center_fd.get('joints', []), metrics)

        # Package descriptor
        descriptor = {
            'frame': center_fn,
            'window_frames': window_fns,
            'window_size': len(window_fns),
            'body_metrics': metrics,
            'pose_class': pose,
            'movement_class': movement,
            'limb_subclusters': subclusters,
            'ywall_comparison': ywall_cmp,
            'shell_ywall_count': yw_stats.get(center_fn, {}).get('y_wall_count', 0),
            'grid_origin': center_fd.get('grid_origin'),
        }

        # Write JSON
        ok = _write_descriptor(out_dir, center_fn, descriptor)
        results[center_fn] = ok

        # Store in state_bank for in-memory access during Phase 3
        if state_bank is not None and hasattr(state_bank, 'body_descriptors_by_frame'):
            state_bank.body_descriptors_by_frame[center_fn] = descriptor

    n_ok = sum(1 for v in results.values() if v)
    logger.info(f"[BD] === Descriptor pass complete: {n_ok}/{len(results)} frames ===")
    return results


# ===========================================================================
# SECTION 9 — UTILITIES
# ===========================================================================

def _discover_frame_nums(results_dir: str) -> List[int]:
    """Return sorted frame numbers from results.json files."""
    nums = []
    for fname in os.listdir(results_dir):
        if fname.endswith('_results.json') and not fname.endswith('_results_p.json') \
                                             and not fname.endswith('_results_s.json'):
            stem = fname.replace('_results.json', '')
            if stem.startswith('frame_'):
                num_str = stem[len('frame_'):]
                if num_str.isdigit():
                    nums.append(int(num_str))
    return sorted(nums)


def _write_descriptor(out_dir: str, frame_num: int, descriptor: Dict) -> bool:
    """Write frame_NNN_body_descriptor.json (compact, 1-digit, 2-level newlines)."""
    path = os.path.join(out_dir, f'frame_{frame_num:03d}_body_descriptor.json')
    try:
        compact = _round_for_json(descriptor)

        def _compact(v):
            return json.dumps(v, separators=(',', ':'), cls=_NumpyEncoder)

        lines = ['{']
        top_keys = list(compact.keys())
        for i, k in enumerate(top_keys):
            v = compact[k]
            top_comma = ',' if i < len(top_keys) - 1 else ''
            if isinstance(v, dict) and v:
                # Dict value: open { on key line, each sub-key on its own line
                sub_keys = list(v.keys())
                lines.append(f'"{k}":{{')
                for j, sk in enumerate(sub_keys):
                    sv = _compact(v[sk])
                    if j < len(sub_keys) - 1:
                        lines.append(f'"{sk}":{sv},')
                    else:
                        # Last sub-key: close dict + top-level comma
                        lines.append(f'"{sk}":{sv}}}{top_comma}')
            else:
                lines.append(f'"{k}":{_compact(v)}{top_comma}')
        lines.append('}')

        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"[BD] Frame {frame_num}: wrote {os.path.basename(path)} "
                    f"(pose={descriptor.get('pose_class')}, "
                    f"move={descriptor.get('movement_class')}, "
                    f"ywalls={descriptor.get('shell_ywall_count')})")
        return True
    except Exception as exc:
        logger.error(f"[BD] Frame {frame_num}: failed to write descriptor: {exc}")
        return False