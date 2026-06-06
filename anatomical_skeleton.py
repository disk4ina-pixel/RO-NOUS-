# anatomical_skeleton.py
"""
Anatomical Skeleton Model with CORRECT ICCS and LCS

ICCS = Inner Cluster Coordinate System (World-Aligned)
  - Origin: Pelvis center (midpoint between hips) - MOVES with person
  - Z-axis: ALWAYS vertical UP [0, 0, 1] - NEVER tilts
  - Y-axis: ALWAYS horizontal, person's FACING direction
  - X-axis: ALWAYS horizontal, person's RIGHT direction
  - Only ONE rotation: yaw angle around Z
  
This means:
  - Z coordinate = HEIGHT relative to pelvis (always clear!)
  - X coordinate = LEFT/RIGHT relative to body center (always clear!)
  - Y coordinate = FORWARD/BACKWARD relative to facing (always clear!)
  - Hip tilt is captured in hip Z-values, NOT in axis tilt
  - Body bend is captured in keypoint positions, NOT in axis tilt

LCS = Local Coordinate System (per-segment, for joint angles)
  - Each bone segment has its own coordinate system
  - Used for computing joint rotations within anatomical limits

Author: RO-NOUS Project
Created on Thu Nov 20 14:00:00 2025
Updated on Thu Jun 04 13:28:29 2026 (manual dY sign corrected: user dY = -ICCS Y; dX/L-R/nose unchanged)
"""

import math
import numpy as np
# from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum
# import logging

from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import deque
import numpy as np
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enhanced_grid import EnhancedOccupancyGrid

logger = logging.getLogger(__name__)


# =============================================================================
# KEYPOINT INDICES (COCO 0-16 + Extended 17-20)
# =============================================================================
class KP(IntEnum):
    """
    Keypoint indices.
    
    0-16: Standard COCO keypoints (from MMPose)
    17-20: Extended skeletal joints (computed, not detected)
    
    Extended joints form the SPINE CHAIN:
      PELVIS_CENTER (19) [OK] SPINE_MID (20) [OK] SHOULDER_CENTER (18) [OK] HEAD_CENTER (17)
    
    This gives us a proper skeletal hierarchy with meaningful DoF.
    """
    # ----- COCO Keypoints (0-16) -----
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # ----- Extended Skeletal Joints (17-20) -----
    HEAD_CENTER = 17       # Midpoint between ears (base of skull)
    SHOULDER_CENTER = 18   # Midpoint between shoulders (upper spine)
    PELVIS_CENTER = 19     # Midpoint between hips (ICCS origin)
    SPINE_MID = 20         # Midpoint between shoulder_center and pelvis_center


# Total keypoint count
NUM_KEYPOINTS = 21  # 17 COCO + 4 extended

# =========================================================================
# LOCKED-LENGTH KINEMATIC CORE  (_lk_*)
# Rest skeleton built ONCE from stature H (ratio table); posed by ANGLES only
# via FK, outward from the pelvis, with anatomical swing limits (hinges cannot
# splay), contralateral mirroring for occluded limbs, rest fallback for
# unsignaled joints, and previous-frame angle seeding.  Frame: ICCS
# (X=right, Y=facing/depth, Z=up, pelvis=origin).  Bone lengths are
# structurally exact (rotations only) -> running-average bone drift and the
# splay/crouch are impossible by construction.
# =========================================================================
_LK_F = dict(ankle=0.040, knee=0.285, hip=0.530, spine_mid=0.675, shoulder=0.820,
             elbow=0.640, wrist=0.500, head_center=0.900, ears=0.910, eyes=0.930,
             nose=0.920, crown=1.000)
_LK_HALF = dict(hip=0.085, shoulder=0.110, head=0.040, eye=0.020)
_LK_NOSE_FWD = 0.120   # "Pinocchio" long nose: clearly clears the head sphere in the PLY (also marks facing)
_LK_EYE_FWD  = 0.012   # eyes stay modestly forward (decoupled from the nose length)
_LK_HALF_DEPTH_FRAC = 0.07  # locked anterior-posterior HALF body depth as a fraction of H.
                            # Pelvis depth = shell front-surface + this*H, seating the
                            # body center INSIDE the shell.  Locked (never the noisy
                            # cluster thickness / centroid), per the depth-anchor design.
# DEBUG switch: when True, emit the bare REST mannequin (all DoF at initial / zero),
# placed at the pelvis and rotated only by the ICCS yaw -- NO per-frame pose solve.
# Use to verify pelvis placement, locked bone lengths, and mannequin/shell scale
# in isolation.  Set False to re-enable the full locked-length FK pose solve.
_LK_REST_ONLY = True
# DEBUG switch: when True, pose ONLY the LEFT leg (KP 11->13->15) and the
# RIGHT arm (KP 6->8->10) from their OWN-side detected directions -- no
# contralateral mirror, no temporal blend, so left stays left / right stays
# right (sides cannot flip).  Every other bone gets no target -> stays at REST.
# Use to verify one leg + one arm pose correctly before enabling all four limbs.
# OFF for now: orientation (root yaw) is verified first in REST-ONLY so the
# facing/L-R is unambiguous; re-enable once the mannequin faces correctly.
_LK_TEST_LIMBS = False
# MANUAL FK test (frame 1 verification, per user request): rotate ONLY the
# named PARENT joint so its CHILD lands at (rest + offset); the child's own
# bone gets no target so it stays rigid at rest relative to its now-rotated
# parent ("keep its DoF as is").  Drives BOTH the PLY and the MP4 (both render
# the LK-FK fitted skeleton).  Offsets are ICCS cm with dX->lateral X and
# dY->DEPTH Y.  NB: a rest limb hangs vertically, so rotation can only move its
# endpoint perpendicular to the bone (lateral X / depth Y); a straight-up
# vertical (Z) move is NOT achievable by rotation (it's along the bone, and
# bone length is locked), so dY is depth.  Set False to return to REST/normal.
#   joint 5 (L-shoulder) DoF -> move joint 7  (L-elbow) by dX=-10, dY=-20
#   joint 11 (L-hip)     DoF -> move joint 13 (L-knee)  by         dY=+20
_LK_MANUAL_TEST = True
_LK_MANUAL_OFFSETS = {7:  (-10.0,  20.0, 0.0),   # L-elbow: dX=-10 (=ICCS X), dY=-20 (=ICCS Y +20)
                      13: (  0.0, -20.0, 0.0)}   # L-knee:  dY=+20 (=ICCS Y -20)
# NB: user dX maps to ICCS X directly (lateral, +X=person's right).  user dY is
# the OPPOSITE sign of ICCS Y: user +dY moves toward -Y (the facing/away side),
# user -dY moves toward +Y (toward camera).  i.e. ICCS_Y_offset = -user_dY.
# (Confirmed empirically: dX came out correct, dY came out reversed.)

def _lk_build_rest_skeleton(H):
    """21x3 ICCS rest pose (arms down, legs vertical, face -Y, LEFT=-X), locked to H."""
    z = {k: (v - _LK_F['hip']) * H for k, v in _LK_F.items()}
    kp = np.zeros((NUM_KEYPOINTS, 3))
    kp[KP.PELVIS_CENTER]   = [0, 0, 0]
    kp[KP.SPINE_MID]       = [0, 0, z['spine_mid']]
    kp[KP.SHOULDER_CENTER] = [0, 0, z['shoulder']]
    kp[KP.HEAD_CENTER]     = [0, 0, z['head_center']]
    hx = _LK_HALF['hip'] * H
    # ICCS X = RIGHT, so the person's LEFT side is -X and RIGHT side is +X
    # (matches the detected pose: right shoulder/hip at +X, left at -X).  This
    # was reversed (LEFT=+X), which mirrored the figure -> L/R appeared swapped
    # in the render while the nose (midline, X=0) still pointed correctly away.
    for sgn, hip, kne, ank in [(-1, KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE),
                               (+1, KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE)]:
        kp[hip] = [sgn*hx, 0, z['hip']]
        kp[kne] = [sgn*hx, 0, z['knee']]
        kp[ank] = [sgn*hx, 0, z['ankle']]
    sx = _LK_HALF['shoulder'] * H
    for sgn, sho, elb, wri in [(-1, KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST),
                               (+1, KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST)]:
        kp[sho] = [sgn*sx, 0, z['shoulder']]
        kp[elb] = [sgn*sx, 0, z['elbow']]
        kp[wri] = [sgn*sx, 0, z['wrist']]
    ex, hxh = _LK_HALF['eye']*H, _LK_HALF['head']*H
    kp[KP.NOSE]      = [0,    -_LK_NOSE_FWD*H,    z['nose']]   # -Y: face away from camera
    kp[KP.LEFT_EYE]  = [-ex,  -_LK_EYE_FWD*H, z['eyes']]
    kp[KP.RIGHT_EYE] = [+ex,  -_LK_EYE_FWD*H, z['eyes']]
    kp[KP.LEFT_EAR]  = [-hxh, 0, z['ears']]
    kp[KP.RIGHT_EAR] = [+hxh, 0, z['ears']]
    return kp

_LK_PARENT = {
    KP.SPINE_MID: KP.PELVIS_CENTER, KP.SHOULDER_CENTER: KP.SPINE_MID, KP.HEAD_CENTER: KP.SHOULDER_CENTER,
    KP.NOSE: KP.HEAD_CENTER, KP.LEFT_EYE: KP.HEAD_CENTER, KP.RIGHT_EYE: KP.HEAD_CENTER,
    KP.LEFT_EAR: KP.HEAD_CENTER, KP.RIGHT_EAR: KP.HEAD_CENTER,
    KP.LEFT_SHOULDER: KP.SHOULDER_CENTER, KP.LEFT_ELBOW: KP.LEFT_SHOULDER, KP.LEFT_WRIST: KP.LEFT_ELBOW,
    KP.RIGHT_SHOULDER: KP.SHOULDER_CENTER, KP.RIGHT_ELBOW: KP.RIGHT_SHOULDER, KP.RIGHT_WRIST: KP.RIGHT_ELBOW,
    KP.LEFT_HIP: KP.PELVIS_CENTER, KP.LEFT_KNEE: KP.LEFT_HIP, KP.LEFT_ANKLE: KP.LEFT_KNEE,
    KP.RIGHT_HIP: KP.PELVIS_CENTER, KP.RIGHT_KNEE: KP.RIGHT_HIP, KP.RIGHT_ANKLE: KP.RIGHT_KNEE,
}
_LK_ORDER = [KP.SPINE_MID, KP.SHOULDER_CENTER, KP.HEAD_CENTER, KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE,
             KP.LEFT_EAR, KP.RIGHT_EAR, KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST,
             KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST, KP.LEFT_HIP, KP.LEFT_KNEE,
             KP.LEFT_ANKLE, KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE]
_LK_FACE = {KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR}
# DOF limits (deg): (flexion lo,hi), abduction or None=LOCKED hinge, twist
_LK_LIM = {
    KP.LEFT_ELBOW: ((0,150), None, (-10,10)), KP.RIGHT_ELBOW: ((0,150), None, (-10,10)),
    KP.LEFT_KNEE:  ((0,150), None, (-10,10)), KP.RIGHT_KNEE:  ((0,150), None, (-10,10)),
    KP.LEFT_SHOULDER: ((-90,180),(-30,160),(-90,90)), KP.RIGHT_SHOULDER: ((-90,180),(-30,160),(-90,90)),
    KP.LEFT_HIP: ((-30,120),(-45,45),(-40,40)), KP.RIGHT_HIP: ((-30,120),(-45,45),(-40,40)),
}
_LK_DEFAULT_LIM = ((-45,45),(-45,45),(-30,30))
_LK_CONTRA = {KP.LEFT_SHOULDER:KP.RIGHT_SHOULDER, KP.RIGHT_SHOULDER:KP.LEFT_SHOULDER,
              KP.LEFT_ELBOW:KP.RIGHT_ELBOW, KP.RIGHT_ELBOW:KP.LEFT_ELBOW,
              KP.LEFT_WRIST:KP.RIGHT_WRIST, KP.RIGHT_WRIST:KP.LEFT_WRIST,
              KP.LEFT_HIP:KP.RIGHT_HIP, KP.RIGHT_HIP:KP.LEFT_HIP,
              KP.LEFT_KNEE:KP.RIGHT_KNEE, KP.RIGHT_KNEE:KP.LEFT_KNEE,
              KP.LEFT_ANKLE:KP.RIGHT_ANKLE, KP.RIGHT_ANKLE:KP.LEFT_ANKLE}

def _lk_Rx(t):
    c, s = np.cos(t), np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _lk_Ry(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _lk_Rz(t):
    # Yaw about the ICCS up-axis Z (CCW from above), same convention as
    # rotate_skeleton21_by_yaw: facing=toward->0deg, away->180deg.
    c, s = np.cos(t), np.sin(t); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def _lk_clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _lk_bone_basis(rest, child):
    """rest-world local basis (cols X=flexion, Y=abduction, Z=along-bone)."""
    Z = rest[child] - rest[_LK_PARENT[child]]; Z = Z/(np.linalg.norm(Z)+1e-12)
    ref = np.array([1.0,0,0])
    if abs(np.dot(ref, Z)) > 0.95: ref = np.array([0.0,1,0])
    X = ref - np.dot(ref, Z)*Z; X /= (np.linalg.norm(X)+1e-12)
    return np.column_stack([X, np.cross(Z, X), Z])

def _lk_solve_fk(H, rest, targets, root_pos, root_R=None, free_children=None):
    """Pose the locked rest skeleton by per-bone rotations (angles only).
    Bone lengths are preserved exactly; hinge abduction is zeroed.
    free_children: optional set of child indices whose bone is rotated straight
    to its target with NO DoF clamp (used by the manual FK test for exact
    control); all other bones stay clamped to anatomical limits."""
    if root_R is None: root_R = np.eye(3)
    pos = np.zeros((NUM_KEYPOINTS,3)); Rw = {KP.PELVIS_CENTER: root_R}
    pos[KP.PELVIS_CENTER] = np.asarray(root_pos, float)
    for c in _LK_ORDER:
        p = _LK_PARENT[c]; Rp = Rw[p]; Bc = _lk_bone_basis(rest, c)
        blen = float(np.linalg.norm(rest[c]-rest[p]))
        t = targets.get(c) if (targets and c not in _LK_FACE) else None
        if t is None:
            fx = ab = 0.0
        else:
            t = np.asarray(t, float); t /= (np.linalg.norm(t)+1e-12)
            d = Bc.T @ (Rp.T @ t)
            fx = -np.degrees(np.arcsin(_lk_clamp(d[1], -1, 1)))
            ab =  np.degrees(np.arctan2(d[0], d[2]))
            # DoF limits belong to the bone's PROXIMAL joint (the socket it
            # rotates about), not the distal endpoint.  Bone parent[c]->c is
            # articulated at parent[c]: thigh(HIP->KNEE) uses HIP ball limits,
            # shin(KNEE->ANKLE) uses KNEE hinge, upper-arm uses SHOULDER ball,
            # forearm uses ELBOW hinge.  (Was keyed by c -> off-by-one: thighs
            # were clamped as hinges and shins were free to splay.)
            if not (free_children and c in free_children):
                fl, abl, _tw = _LK_LIM.get(_LK_PARENT[c], _LK_DEFAULT_LIM)
                fx = _lk_clamp(fx, *fl)
                ab = 0.0 if abl is None else _lk_clamp(ab, *abl)
        Rs = _lk_Ry(np.radians(ab)) @ _lk_Rx(np.radians(fx))
        a_local = Rs @ np.array([0,0,1.0])
        a_world = Rp @ (Bc @ a_local)
        pos[c] = pos[p] + a_world * blen
        Rw[c] = Rp @ (Bc @ Rs @ Bc.T)
    return pos

def _lk_build_targets(detected, conf, depth_damp=0.25, conf_th=0.3,
                      prev_targets=None, follow=0.5):
    """Per-bone target dirs: front-surface (depth-damped) -> mirror twin -> rest."""
    detected = np.asarray(detected, float)
    valid = (conf > conf_th) & (np.abs(detected).sum(1) > 1e-6)
    raw = {}
    for c in _LK_ORDER:
        p = _LK_PARENT[c]
        if valid[c] and valid[p]:
            d = (detected[c]-detected[p]).astype(float); d[1] *= depth_damp
            n = np.linalg.norm(d)
            if n > 1e-6: raw[c] = d/n
    targets = {}
    for c in _LK_ORDER:
        if c in raw:
            tgt = raw[c]
        elif c in _LK_CONTRA and _LK_CONTRA[c] in raw:
            m = raw[_LK_CONTRA[c]].copy(); m[0] = -m[0]; tgt = m
        else:
            tgt = None
        if tgt is not None and prev_targets and prev_targets.get(c) is not None:
            b = (1-follow)*np.asarray(prev_targets[c]) + follow*tgt
            n = np.linalg.norm(b)
            if n > 1e-6: tgt = b/n
        targets[c] = tgt
    return targets, None


# ---------------------------------------------------------------------------
# LEG-FIT DoF FIX — the bone-constraint fit planted thighs/shins along the RAW
# detected direction (detected_knee - hip), which carries the unreliable ICCS
# Y (depth/facing) axis from the half-shell.  Noisy detected legs pushed the
# thigh toward horizontal (~90° hip), producing the splayed-leg artefact.
# These controls damp the depth component of the detected limb direction and
# limit how far a limb may tilt off its rest reference, so a standing limb
# stays down.  The damping is SIDE-VIEW AWARE: in a near-lateral (profile)
# view the ICCS-Y depth axis is the camera-facing axis and is almost entirely
# unreliable (and the far-side limb is occluded), so depth is damped harder.
#   - LIMB_FIT_DEPTH_DAMPING_FRONTAL: depth scale when facing the camera.
#   - LIMB_FIT_DEPTH_DAMPING_LATERAL: depth scale in profile (much smaller).
#   - LEG_FIT_MAX_TILT_FROM_DOWN_DEG: max thigh tilt off straight-down.
#   - ARM_FIT_MAX_TILT_FROM_DOWN_DEG: max upper-arm tilt off straight-down
#     (arms-down rest); the forearm is clamped relative to the upper arm.
LIMB_FIT_DEPTH_DAMPING_FRONTAL = 0.35
LIMB_FIT_DEPTH_DAMPING_LATERAL = 0.08
LEG_FIT_MAX_TILT_FROM_DOWN_DEG = 60.0
ARM_FIT_MAX_TILT_FROM_DOWN_DEG = 95.0

# Backward-compat aliases (older code/refs may import these names)
LEG_FIT_DEPTH_DAMPING = LIMB_FIT_DEPTH_DAMPING_FRONTAL


KEYPOINT_NAMES = [
    # COCO (0-16)
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    # Extended (17-20)
    'head_center', 'shoulder_center', 'pelvis_center', 'spine_mid'
]


# Which keypoints are computed (not detected by MMPose)
COMPUTED_KEYPOINTS = [KP.HEAD_CENTER, KP.SHOULDER_CENTER, KP.PELVIS_CENTER, KP.SPINE_MID]

# Spine chain (from bottom to top)
SPINE_CHAIN = [KP.PELVIS_CENTER, KP.SPINE_MID, KP.SHOULDER_CENTER, KP.HEAD_CENTER]

# =============================================================================
# FACE KEYPOINTS - Special depth handling for shell fitting
# =============================================================================
FACE_KEYPOINTS = {KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR}

# Depth offsets for face keypoints relative to head surface (in cm)
# Positive = deeper into head (away from visible surface)
# These are used when person is facing AWAY from camera
FACE_DEPTH_OFFSETS = {
    KP.NOSE: 2.0,       # Nose protrudes, but when facing away, it's deeper
    KP.LEFT_EYE: 1.5,   # Eyes are slightly recessed
    KP.RIGHT_EYE: 1.5,
    KP.LEFT_EAR: 0.0,   # Ears are on the surface
    KP.RIGHT_EAR: 0.0,
}


# =============================================================================
# 24-STATE FACING → YAW ANGLE & FITTING PATH
# =============================================================================
# Local copy of the angle map (canonical source is mmpose_integration.py)
# to avoid circular imports (anatomical_skeleton is imported BY mmpose_integration).
#
# Standard convention: 0°=toward_camera, 90°=side_right, 180°=away, 270°=side_left
# =============================================================================

FACING_ANGLE_MAP = {
    'toward_camera': 0.0,
    'front_right_slight': 15.0,
    'front_right': 30.0,
    'front_right_diagonal': 45.0,
    'side_right_front': 60.0,
    'side_right': 75.0,
    'side_right_back': 90.0,
    'back_right_diagonal': 105.0,
    'back_right': 120.0,
    'back_right_slight': 135.0,
    'away_right': 150.0,
    'away_from_camera_slight_right': 165.0,
    'away_from_camera': 180.0,
    'away_from_camera_slight_left': 195.0,
    'away_left': 210.0,
    'back_left_slight': 225.0,
    'back_left': 240.0,
    'back_left_diagonal': 255.0,
    'side_left_back': 270.0,
    'side_left': 285.0,
    'side_left_front': 300.0,
    'front_left_diagonal': 315.0,
    'front_left': 330.0,
    'front_left_slight': 345.0,
    # Legacy labels (for backward compatibility)
    'front': 0.0,
    'back': 180.0,
    'left': 285.0,
    'right': 75.0,
}

# Fitting path groups — which labels map to which shell-fitting strategy
_FP_TOWARD = {
    'toward_camera', 'front_right_slight', 'front_left_slight',
    'front',  # legacy
}
_FP_AWAY = {
    'away_from_camera', 'away_from_camera_slight_right',
    'away_from_camera_slight_left',
    'back',  # legacy
}
_FP_SIDE_RIGHT = {
    'front_right', 'front_right_diagonal',
    'side_right_front', 'side_right', 'side_right_back',
    'back_right_diagonal', 'back_right', 'back_right_slight',
    'away_right',
    'right',  # legacy
}
_FP_SIDE_LEFT = {
    'front_left', 'front_left_diagonal',
    'side_left_front', 'side_left', 'side_left_back',
    'back_left_diagonal', 'back_left', 'back_left_slight',
    'away_left',
    'left',  # legacy
}


def _resolve_fitting_path(facing_direction: str) -> str:
    """
    Map a 24-state (or legacy) facing label to a shell-fitting path.

    Returns:
        'frontal', 'dorsal', 'side_right', or 'side_left'.
    """
    if facing_direction in _FP_TOWARD:
        return 'frontal'
    if facing_direction in _FP_AWAY:
        return 'dorsal'
    if facing_direction in _FP_SIDE_RIGHT:
        return 'side_right'
    if facing_direction in _FP_SIDE_LEFT:
        return 'side_left'
    # Fallback
    return 'frontal'


def _facing_is_away(facing_direction: str) -> bool:
    """
    Check if a facing label corresponds to the person facing AWAY from camera.
    Replaces direct `== 'away_from_camera'` checks to handle 24-state labels.
    """
    return facing_direction in _FP_AWAY


# =============================================================================
# ICCS - INNER CLUSTER COORDINATE SYSTEM (WORLD-ALIGNED)
# =============================================================================
class ICCS:
    """
    Inner Cluster Coordinate System - World-Aligned.
    
    DEFINITION:
      - Origin: Pelvis center (midpoint between left and right hip)
      - Z-axis: ALWAYS [0, 0, 1] - vertical UP (NEVER tilts!)
      - Y-axis: ALWAYS horizontal - person's facing direction
      - X-axis: ALWAYS horizontal - person's right direction (Y [OK] Z)
    
    PROPERTIES:
      - Only ONE rotation parameter: yaw (rotation around Z)
      - Z-coordinate always means HEIGHT relative to pelvis
      - X-coordinate always means LEFT(-)/RIGHT(+) relative to body
      - Y-coordinate always means BACK(-)/FORWARD(+) relative to facing
    
    BENEFITS:
      - Hip tilt captured in hip Z-values, not axis tilt
      - Body bend captured in keypoint positions, not axis tilt
      - Simple rotation model (single angle)
      - Clear physical meaning for all coordinates
    """
    
    def __init__(self):
        # Origin in world coordinates
        self._origin = np.array([0.0, 0.0, 0.0])
        
        # Yaw rotation angle (degrees)
        # 0[OK] = facing world +Y
        # 90[OK] = facing world +X
        # 180[OK] = facing world -Y
        # -90[OK] = facing world -X
        self._yaw = 0.0
        
        # Cached rotation matrix (2D rotation in XY plane)
        self._R: Optional[np.ndarray] = None
        self._R_inv: Optional[np.ndarray] = None
        self._cache_valid = False
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def origin(self) -> np.ndarray:
        """Origin (pelvis center) in world coordinates"""
        return self._origin.copy()
    
    @origin.setter
    def origin(self, value: np.ndarray):
        self._origin = np.array(value, dtype=float)
    
    @property
    def yaw(self) -> float:
        """Yaw rotation in degrees (rotation around Z-axis)"""
        return self._yaw
    
    @yaw.setter
    def yaw(self, value: float):
        # Normalize to [-180, 180]
        self._yaw = ((value + 180) % 360) - 180
        self._cache_valid = False
    
    @property
    def x_axis(self) -> np.ndarray:
        """X-axis (RIGHT direction) in world coordinates - always horizontal"""
        rad = np.radians(self._yaw)
        return np.array([np.cos(rad), np.sin(rad), 0.0])
    
    @property
    def y_axis(self) -> np.ndarray:
        """Y-axis (FORWARD direction) in world coordinates - always horizontal"""
        rad = np.radians(self._yaw)
        return np.array([-np.sin(rad), np.cos(rad), 0.0])
    
    @property
    def z_axis(self) -> np.ndarray:
        """Z-axis (UP direction) - ALWAYS vertical [0, 0, 1]"""
        return np.array([0.0, 0.0, 1.0])
    
    # =========================================================================
    # ROTATION MATRICES
    # =========================================================================
    
    def _update_rotation_cache(self):
        """Update cached rotation matrices"""
        if self._cache_valid:
            return
        
        rad = np.radians(self._yaw)
        c, s = np.cos(rad), np.sin(rad)
        
        # Rotation matrix: ICCS to World
        # Columns are ICCS axes expressed in world coordinates
        self._R = np.array([
            [c, -s, 0],   # X-axis (right)
            [s,  c, 0],   # Y-axis (forward) 
            [0,  0, 1]    # Z-axis (up)
        ])
        
        # Inverse: World to ICCS
        self._R_inv = np.array([
            [ c, s, 0],
            [-s, c, 0],
            [ 0, 0, 1]
        ])
        
        self._cache_valid = True
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix (ICCS to World)"""
        self._update_rotation_cache()
        return self._R.copy()
    
    def get_inverse_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 inverse rotation matrix (World to ICCS)"""
        self._update_rotation_cache()
        return self._R_inv.copy()
    
    # =========================================================================
    # COORDINATE TRANSFORMS
    # =========================================================================
    
    def world_to_iccs(self, point_world: np.ndarray) -> np.ndarray:
        """
        Transform point from world coordinates to ICCS.
        
        Args:
            point_world: 3D point in world coordinates
            
        Returns:
            3D point in ICCS coordinates where:
              - X = left(-)/right(+) relative to body
              - Y = back(-)/forward(+) relative to facing
              - Z = height relative to pelvis
        """
        self._update_rotation_cache()
        translated = point_world - self._origin
        return self._R_inv @ translated
    
    def iccs_to_world(self, point_iccs: np.ndarray) -> np.ndarray:
        """
        Transform point from ICCS to world coordinates.
        
        Args:
            point_iccs: 3D point in ICCS coordinates
            
        Returns:
            3D point in world coordinates
        """
        self._update_rotation_cache()
        return self._R @ point_iccs + self._origin
    
    def world_to_iccs_batch(self, points_world: np.ndarray) -> np.ndarray:
        """Transform multiple points from world to ICCS (Nx3 array)"""
        self._update_rotation_cache()
        translated = points_world - self._origin
        return (self._R_inv @ translated.T).T
    
    def iccs_to_world_batch(self, points_iccs: np.ndarray) -> np.ndarray:
        """Transform multiple points from ICCS to world (Nx3 array)"""
        self._update_rotation_cache()
        return (self._R @ points_iccs.T).T + self._origin
    
    # =========================================================================
    # UPDATE METHODS
    # =========================================================================
    
    def update(self, origin: np.ndarray, yaw_degrees: float):
        """
        Update ICCS position and rotation.
        
        Args:
            origin: Pelvis center in world coordinates
            yaw_degrees: Facing direction (0[OK] = +Y, 90[OK] = +X)
        """
        self._origin = np.array(origin, dtype=float)
        self.yaw = yaw_degrees  # Uses setter for normalization
        logger.debug(f"ICCS updated: origin={self._origin}, yaw={self._yaw:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
    
    def update_from_keypoints(self, keypoints_world: np.ndarray) -> bool:
        """
        Update ICCS from detected keypoints.
        
        Computes:
          - Origin from hip midpoint
          - Yaw from shoulder/hip orientation
        
        Args:
            keypoints_world: 17x3 keypoints in world coordinates
            
        Returns:
            True if successful
        """
        left_hip = keypoints_world[KP.LEFT_HIP]
        right_hip = keypoints_world[KP.RIGHT_HIP]
        
        # Check hips are valid
        if np.allclose(left_hip, 0) or np.allclose(right_hip, 0):
            logger.warning("ICCS update failed: invalid hip keypoints")
            return False
        
        # Origin = pelvis center
        self._origin = (left_hip + right_hip) / 2
        
        # Yaw from hip orientation (right hip direction)
        hip_vec = right_hip - left_hip
        # X-axis points toward right hip, so yaw = atan2(hip_vec.y, hip_vec.x)
        self.yaw = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))
        
        logger.debug(f"ICCS from keypoints: origin={self._origin}, yaw={self._yaw:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
        return True
    
    def update_from_cluster(self, centroid: np.ndarray, yaw_degrees: float, 
                           ankle_z: Optional[float] = None):
        """
        Update ICCS from cluster data.
        
        Args:
            centroid: Cluster centroid in world coordinates
            yaw_degrees: Detected body rotation
            ankle_z: Optional ankle Z coordinate (to compute proper pelvis height)
        """
        # Use centroid as initial origin
        origin = np.array(centroid, dtype=float)
        
        # If we know ankle height, adjust pelvis height
        # Pelvis is typically ~53% of total height above ground
        # For 170cm person: pelvis at ~90cm
        if ankle_z is not None:
            # Estimate pelvis height from ankle
            # This is approximate - will be refined by skeleton fitting
            pass
        
        self.update(origin, yaw_degrees)
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def get_transform_matrix_4x4(self) -> np.ndarray:
        """Get 4x4 homogeneous transformation matrix (ICCS to World)"""
        self._update_rotation_cache()
        T = np.eye(4)
        T[:3, :3] = self._R
        T[:3, 3] = self._origin
        return T
    
    def get_inverse_transform_matrix_4x4(self) -> np.ndarray:
        """Get 4x4 inverse transformation matrix (World to ICCS)"""
        self._update_rotation_cache()
        T_inv = np.eye(4)
        T_inv[:3, :3] = self._R_inv
        T_inv[:3, 3] = -self._R_inv @ self._origin
        return T_inv
    
    def copy(self) -> 'ICCS':
        """Create a copy of this ICCS"""
        new_iccs = ICCS()
        new_iccs._origin = self._origin.copy()
        new_iccs._yaw = self._yaw
        return new_iccs
    
    def __repr__(self) -> str:
        return f"ICCS(origin={self._origin}, yaw={self._yaw:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°)"


# =============================================================================
# LCS - LOCAL COORDINATE SYSTEM (PER SEGMENT)
# =============================================================================
@dataclass
class SegmentLCS:
    """
    Local Coordinate System for a bone segment.
    
    Used for computing joint angles within anatomical limits.
    
    Convention:
      - Origin: Proximal joint (parent)
      - Z-axis: Along bone direction (toward child joint)
      - X-axis: Flexion/extension axis
      - Y-axis: Completes right-hand system
    """
    name: str
    parent_kp: int          # Parent keypoint index
    child_kp: int           # Child keypoint index
    
    # Joint rotation limits (degrees) - None means locked
    rx_limits: Optional[Tuple[float, float]] = None  # Flexion/Extension
    ry_limits: Optional[Tuple[float, float]] = None  # Abduction/Adduction
    rz_limits: Optional[Tuple[float, float]] = None  # Internal/External rotation
    
    # Current state
    bone_length: float = 0.0
    rx: float = 0.0  # Current flexion angle
    ry: float = 0.0  # Current abduction angle
    rz: float = 0.0  # Current rotation angle
    
    def set_angles(self, rx: float = 0, ry: float = 0, rz: float = 0):
        """Set joint angles with limit clamping"""
        if self.rx_limits:
            self.rx = np.clip(rx, self.rx_limits[0], self.rx_limits[1])
        else:
            self.rx = 0
            
        if self.ry_limits:
            self.ry = np.clip(ry, self.ry_limits[0], self.ry_limits[1])
        else:
            self.ry = 0
            
        if self.rz_limits:
            self.rz = np.clip(rz, self.rz_limits[0], self.rz_limits[1])
        else:
            self.rz = 0
    
    def get_dof(self) -> int:
        """Get degrees of freedom for this joint"""
        return sum(1 for lim in [self.rx_limits, self.ry_limits, self.rz_limits] if lim is not None)


# =============================================================================
# SEGMENT DEFINITIONS WITH ANATOMICAL LIMITS
# =============================================================================
def create_segments() -> Dict[str, SegmentLCS]:
    """
    Create segment definitions with anatomical joint limits.
    
    SKELETAL HIERARCHY:
    
    PELVIS_CENTER (19) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] ROOT (ICCS origin)
        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] LEFT_HIP (11) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] LEFT_KNEE (13) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] LEFT_ANKLE (15)
        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] RIGHT_HIP (12) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] RIGHT_KNEE (14) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] RIGHT_ANKLE (16)
        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] SPINE_MID (20)
                ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
                ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] SHOULDER_CENTER (18)
                        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
                        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] LEFT_SHOULDER (5) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] LEFT_ELBOW (7) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] LEFT_WRIST (9)
                        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
                        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] RIGHT_SHOULDER (6) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] RIGHT_ELBOW (8) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] RIGHT_WRIST (10)
                        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
                        ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] HEAD_CENTER (17)
                                ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK]
                                ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] NOSE (0) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] EYES (1,2) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â[OK] EARS (3,4)
    """
    
    return {
        # =====================================================================
        # SPINE CHAIN (Central axis with full DoF)
        # =====================================================================
        
        # Lower spine: Pelvis [OK] Spine_Mid
        'lower_spine': SegmentLCS(
            'lower_spine', 
            KP.PELVIS_CENTER, KP.SPINE_MID,
            rx_limits=(-30, 45),    # Forward bend / back arch
            ry_limits=(-30, 30),    # Side bend
            rz_limits=(-45, 45)     # Torso twist
        ),
        
        # Upper spine: Spine_Mid [OK] Shoulder_Center
        'upper_spine': SegmentLCS(
            'upper_spine',
            KP.SPINE_MID, KP.SHOULDER_CENTER,
            rx_limits=(-20, 30),    # Forward bend / back arch (less than lower)
            ry_limits=(-25, 25),    # Side bend
            rz_limits=(-35, 35)     # Torso twist
        ),
        
        # Neck: Shoulder_Center [OK] Head_Center
        'neck': SegmentLCS(
            'neck',
            KP.SHOULDER_CENTER, KP.HEAD_CENTER,
            rx_limits=(-40, 60),    # Nod down / look up
            ry_limits=(-45, 45),    # Tilt head sideways
            rz_limits=(-70, 70)     # Turn head left/right
        ),
        
        # Head: Head_Center [OK] Nose (simplified, mostly rotation)
        'head': SegmentLCS(
            'head',
            KP.HEAD_CENTER, KP.NOSE,
            rx_limits=(-20, 20),    # Minor nod adjustment
            ry_limits=(-10, 10),    # Minor tilt
            rz_limits=(-10, 10)     # Minor turn
        ),
        
        # =====================================================================
        # HIP JOINTS (Connect pelvis to legs)
        # =====================================================================
        
        # Pelvis width (structural, no DoF)
        'pelvis_width': SegmentLCS('pelvis_width', KP.LEFT_HIP, KP.RIGHT_HIP),
        
        # Left hip offset: Pelvis_Center [OK] Left_Hip (structural)
        'hip_offset_l': SegmentLCS('hip_offset_l', KP.PELVIS_CENTER, KP.LEFT_HIP),
        
        # Right hip offset: Pelvis_Center [OK] Right_Hip (structural)
        'hip_offset_r': SegmentLCS('hip_offset_r', KP.PELVIS_CENTER, KP.RIGHT_HIP),
        
        # =====================================================================
        # SHOULDER JOINTS (Connect upper spine to arms)
        # =====================================================================
        
        # Shoulder width (structural, no DoF)
        'shoulder_width': SegmentLCS('shoulder_width', KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER),
        
        # Left shoulder offset: Shoulder_Center [OK] Left_Shoulder
        'shoulder_offset_l': SegmentLCS(
            'shoulder_offset_l',
            KP.SHOULDER_CENTER, KP.LEFT_SHOULDER,
            # Clavicle can move slightly
            rx_limits=(-15, 30),    # Shoulder shrug up/down
            ry_limits=(-20, 20),    # Shoulder forward/back
            rz_limits=None          # No rotation
        ),
        
        # Right shoulder offset: Shoulder_Center [OK] Right_Shoulder
        'shoulder_offset_r': SegmentLCS(
            'shoulder_offset_r',
            KP.SHOULDER_CENTER, KP.RIGHT_SHOULDER,
            rx_limits=(-15, 30),
            ry_limits=(-20, 20),
            rz_limits=None
        ),
        
        # =====================================================================
        # LEFT ARM
        # =====================================================================
        
        'upper_arm_l': SegmentLCS(
            'upper_arm_l',
            KP.LEFT_SHOULDER, KP.LEFT_ELBOW,
            rx_limits=(-60, 180),   # Flexion (raise forward) / Extension (back)
            ry_limits=(-45, 180),   # Abduction (raise sideways)
            rz_limits=(-90, 90)     # Internal/external rotation
        ),
        
        'forearm_l': SegmentLCS(
            'forearm_l',
            KP.LEFT_ELBOW, KP.LEFT_WRIST,
            rx_limits=(0, 145),     # Elbow flexion only (no hyperextension)
            ry_limits=None,         # No abduction at elbow
            rz_limits=(-90, 90)     # Forearm pronation/supination
        ),
        
        # =====================================================================
        # RIGHT ARM
        # =====================================================================
        
        'upper_arm_r': SegmentLCS(
            'upper_arm_r',
            KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW,
            rx_limits=(-60, 180),
            ry_limits=(-180, 45),   # Mirrored abduction
            rz_limits=(-90, 90)
        ),
        
        'forearm_r': SegmentLCS(
            'forearm_r',
            KP.RIGHT_ELBOW, KP.RIGHT_WRIST,
            rx_limits=(0, 145),
            ry_limits=None,
            rz_limits=(-90, 90)
        ),
        
        # =====================================================================
        # LEFT LEG
        # =====================================================================
        
        'thigh_l': SegmentLCS(
            'thigh_l',
            KP.LEFT_HIP, KP.LEFT_KNEE,
            rx_limits=(-30, 120),   # Extension (back) / Flexion (kick forward)
            ry_limits=(-45, 45),    # Abduction (spread) / Adduction (cross)
            rz_limits=(-45, 45)     # Internal/external rotation
        ),
        
        'shin_l': SegmentLCS(
            'shin_l',
            KP.LEFT_KNEE, KP.LEFT_ANKLE,
            rx_limits=(0, 140),     # Knee flexion only
            ry_limits=None,
            rz_limits=None
        ),
        
        # =====================================================================
        # RIGHT LEG
        # =====================================================================
        
        'thigh_r': SegmentLCS(
            'thigh_r',
            KP.RIGHT_HIP, KP.RIGHT_KNEE,
            rx_limits=(-30, 120),
            ry_limits=(-45, 45),
            rz_limits=(-45, 45)
        ),
        
        'shin_r': SegmentLCS(
            'shin_r',
            KP.RIGHT_KNEE, KP.RIGHT_ANKLE,
            rx_limits=(0, 140),
            ry_limits=None,
            rz_limits=None
        ),
    }


# =============================================================================
# ANTHROPOMETRIC RATIOS (relative to height)
# =============================================================================
ANTHROPOMETRIC_RATIOS = {
    'shoulder_width': 0.235,    # 23.5% of height (~40cm for 170cm; biacromial 38-42cm adults)
    'hip_width': 0.19,          # 19% of height
    'torso': 0.32,              # 32% - shoulder to hip (was 30%, real anatomy 32-33%)
    'upper_arm': 0.186,         # 18.6%
    'forearm': 0.146,           # 14.6%
    'thigh': 0.26,              # 26% of height (was 24.5% — femur is longest bone)
    'shin': 0.215,              # 21.5% of height (was 24.6% — shin ≈ 83% of thigh)
    'head': 0.13,               # 13%
    'pelvis_height_ratio': 0.53,
    'lower_spine_ratio': 0.50,  # 50% of torso = lower spine (pelvis to spine_mid) — TRUE MIDPOINT
    'upper_spine_ratio': 0.50,  # 50% of torso = upper spine (spine_mid to shoulders) — TRUE MIDPOINT
    'neck_ratio': 0.55,         # 55% of head height = neck length (acromion-to-tragion ~13cm for 170cm)
    # ---- Face / head geometry (absolute cm, not height ratios) ----
    # These are adult averages; they do not scale linearly with height.
    # Source: anthropometric atlases (Farkas 1994, Gordon 1988)
    'face_half_ear_width':   8.5,   # half biauricular breadth  (full ~17 cm)
    'face_eye_lateral_sep':  3.2,   # half interocular distance (full ~6.4 cm)
    'face_eye_above_ears':   2.5,   # eyes sit ~2.5 cm above ear horizontal
    'face_nose_protrusion':  3.0,   # nose tip protrudes ~3 cm from face plane
    'face_nose_below_eyes':  1.5,   # nose tip is ~1.5 cm below eye level
}

# =============================================================================
# CLUSTER → TRUE HEIGHT CORRECTION
# =============================================================================
# The person_cluster bbox z_span covers:
#   shin  (0.215H) + thigh (0.26H) = 2 leg sections
#   lower_spine (0.16H) + upper_spine (0.16H) + neck (0.072H) = 3 spine sections
#   ─────────────────────────────────────────────────────────
#   z_span ≈ 0.867H  (the head crown is NOT captured by the point cloud)
#
# Therefore: true_height = z_span + CLUSTER_HEAD_CORRECTION_CM
# ~15 cm accounts for the head globe above the neck keypoint to the crown.
# All bone lengths are locked immediately from true_height via anatomical
# ratios — no multi-frame calibration window is correct or needed.
# =============================================================================
CLUSTER_HEAD_CORRECTION_CM = 15.0

# Symmetric bone pairs
SYMMETRIC_BONES = [
    ('upper_arm_l', 'upper_arm_r'),
    ('forearm_l', 'forearm_r'),
    ('thigh_l', 'thigh_r'),
    ('shin_l', 'shin_r'),
    ('torso_left', 'torso_right'),
]

# =============================================================================
# FLESH-SPACER RADII (Shell-as-Suit Architecture)
# =============================================================================
# Per-segment flesh thickness between bone axis and cluster surface.
# Values are FRACTIONS of person height H (scale: radius = H * ratio).
# Reference person: 170 cm.
#
# Source: shell_as_suit.docx Section 3 / Section 9.1
# =============================================================================
FLESH_RADII = {
    # radius / H  (single isotropic value for the 3D inward-normal offset)
    'head':         0.062,   # ~10.5cm for 170cm
    'neck':         0.029,   # ~5.0cm
    'upper_torso':  0.050,   # ~8.5cm avg of depth(7cm) and lateral(10cm)
    'lower_torso':  0.050,   # ~8.5cm
    'pelvis':       0.047,   # ~8.0cm
    'upper_arm':    0.026,   # ~4.5cm
    'forearm':      0.021,   # ~3.5cm
    'wrist':        0.012,   # ~2.0cm near-bone
    'thigh':        0.056,   # ~9.5cm (avg depth 9cm + lateral 10cm)
    'shin':         0.024,   # ~4.0cm
    'ankle':        0.018,   # ~3.0cm near-bone
    'foot':         0.012,   # ~2.0cm
}

# Map each COCO keypoint index to its flesh segment name.
KP_TO_FLESH_SEGMENT = {
    0:  'head',         # nose
    1:  'head',         # left eye
    2:  'head',         # right eye
    3:  'head',         # left ear
    4:  'head',         # right ear
    5:  'upper_torso',  # left shoulder
    6:  'upper_torso',  # right shoulder
    7:  'upper_arm',    # left elbow
    8:  'upper_arm',    # right elbow
    9:  'wrist',        # left wrist
    10: 'wrist',        # right wrist
    11: 'pelvis',       # left hip
    12: 'pelvis',       # right hip
    13: 'thigh',        # left knee
    14: 'thigh',        # right knee
    15: 'ankle',        # left ankle
    16: 'ankle',        # right ankle
}


# =============================================================================
# BODY ZONES — 6 anatomical regions for voxel zone assignment
# =============================================================================
BODY_ZONES = {
    'head':      0,
    'torso':     1,
    'left_arm':  2,
    'right_arm': 3,
    'left_leg':  4,
    'right_leg': 5,
}

ZONE_COLORS_RGB = {
    0: (255, 255,   0),   # head      → yellow
    1: (  0, 200,   0),   # torso     → green
    2: (  0, 100, 255),   # left_arm  → blue
    3: (  0, 255, 255),   # right_arm → cyan
    4: (255,  50,  50),   # left_leg  → red
    5: (255, 165,   0),   # right_leg → orange
}

# Map keypoint index → zone id
KP_TO_ZONE = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 17: 0,   # head
    18: 1, 19: 1, 20: 1,                       # torso (spine chain)
    5: 1, 6: 1,                                 # shoulders → torso (roots of arm chains)
    11: 1, 12: 1,                               # hips → torso (roots of leg chains)
    7: 2, 9: 2,                                 # left arm
    8: 3, 10: 3,                                # right arm
    13: 4, 15: 4,                               # left leg
    14: 5, 16: 5,                               # right leg
}

# =============================================================================
# KINEMATIC CHAIN ORDER — Option 4 propagation sequence
#
# Each entry: (child_kp, parent_kp, bone_key)
# Anchor joints (ankles) have parent_kp = None
# Computed joints (17-20) are derived after all COCO joints are placed
# =============================================================================
KINEMATIC_CHAIN_ORDER = [
    # --- Anchor joints (placed from ray + MiDaS depth, no parent) ---
    # Ankles first — highest confidence, densest CoP, floor-constrained
    (KP.LEFT_ANKLE,      None,               None),
    (KP.RIGHT_ANKLE,     None,               None),
    # --- Left leg chain (bottom to top) ---
    (KP.LEFT_KNEE,       KP.LEFT_ANKLE,      'shin_l'),
    (KP.LEFT_HIP,        KP.LEFT_KNEE,       'thigh_l'),
    # --- Right leg chain (bottom to top) ---
    (KP.RIGHT_KNEE,      KP.RIGHT_ANKLE,     'shin_r'),
    (KP.RIGHT_HIP,       KP.RIGHT_KNEE,      'thigh_r'),
    # --- Spine chain (bottom to top, from hip midpoint) ---
    # PELVIS_CENTER, SPINE_MID, SHOULDER_CENTER, HEAD_CENTER are COMPUTED
    # after hips are placed — they don't use ray+sphere
    # --- Left arm chain (from shoulder_center outward) ---
    (KP.LEFT_SHOULDER,   KP.SHOULDER_CENTER,  'shoulder_width'),
    (KP.LEFT_ELBOW,      KP.LEFT_SHOULDER,    'upper_arm_l'),
    (KP.LEFT_WRIST,      KP.LEFT_ELBOW,       'forearm_l'),
    # --- Right arm chain (from shoulder_center outward) ---
    (KP.RIGHT_SHOULDER,  KP.SHOULDER_CENTER,  'shoulder_width'),
    (KP.RIGHT_ELBOW,     KP.RIGHT_SHOULDER,   'upper_arm_r'),
    (KP.RIGHT_WRIST,     KP.RIGHT_ELBOW,      'forearm_r'),
    # --- Face (from head_center, all same parent) ---
    (KP.NOSE,            KP.HEAD_CENTER,       'head'),
    (KP.LEFT_EYE,        KP.HEAD_CENTER,       'head'),
    (KP.RIGHT_EYE,       KP.HEAD_CENTER,       'head'),
    (KP.LEFT_EAR,        KP.HEAD_CENTER,       'head'),
    (KP.RIGHT_EAR,       KP.HEAD_CENTER,       'head'),
]


# =============================================================================
# ANATOMICAL SKELETON
# =============================================================================
def _limb_dir_sanitize(raw_dir, depth_damping, max_tilt_deg,
                       reference_vec=None):
    """
    Sanitize a detected limb-segment direction (in ICCS) before it is used to
    plant a thigh/shin or upper-arm/forearm, fixing the splayed-limb DoF
    artefact (legs shooting forward, arm jutting out).

    Steps:
      1. Damp the ICCS-Y (depth/facing) component — the unreliable half-shell
         axis — by `depth_damping`, then renormalize.  Caller passes a smaller
         value in profile views (where depth is least reliable).
      2. Clamp the direction so it does not tilt more than `max_tilt_deg` away
         from its reference:
           - if reference_vec is None: reference is straight-down [0,0,-1]
             (thigh / upper-arm in the arms-down, legs-down rest pose).
           - else: reference is reference_vec (the parent segment), so genuine
             knee/elbow bend is preserved but the distal segment cannot fly off.

    Returns a unit ICCS direction vector.
    """
    d = np.asarray(raw_dir, dtype=float).copy()
    n = np.linalg.norm(d)
    if reference_vec is not None:
        rv = np.asarray(reference_vec, dtype=float)
        rvn = np.linalg.norm(rv)
        ref = rv / rvn if rvn > 1e-9 else np.array([0.0, 0.0, -1.0])
    else:
        ref = np.array([0.0, 0.0, -1.0])

    if n < 1e-9:
        return ref  # degenerate detection → use reference direction

    # 1. depth damping (ICCS Y is the facing/depth axis)
    d[1] *= depth_damping
    nd = np.linalg.norm(d)
    if nd < 1e-9:
        return ref
    d = d / nd

    # 2. tilt clamp toward the reference direction
    cosang = float(np.clip(np.dot(d, ref), -1.0, 1.0))
    ang = np.degrees(np.arccos(cosang))
    if ang > max_tilt_deg:
        perp = d - cosang * ref
        pn = np.linalg.norm(perp)
        if pn < 1e-9:
            return ref
        perp = perp / pn
        tr = np.radians(max_tilt_deg)
        d = np.cos(tr) * ref + np.sin(tr) * perp
        d = d / (np.linalg.norm(d) + 1e-10)
    return d


def _limb_depth_damping_for_view(detected_iccs, valid):
    """
    Pick the ICCS-Y depth damping based on how lateral (profile) the view is.

    In a profile view the torso has little spread along ICCS-X (shoulders/hips
    line up in depth instead of across), so the depth axis is least reliable
    and the far-side limb is occluded → damp depth harder.  Returns a damping
    factor interpolated between the lateral and frontal config values.
    """
    try:
        if valid[KP.LEFT_SHOULDER] and valid[KP.RIGHT_SHOULDER]:
            a, b = detected_iccs[KP.LEFT_SHOULDER], detected_iccs[KP.RIGHT_SHOULDER]
        elif valid[KP.LEFT_HIP] and valid[KP.RIGHT_HIP]:
            a, b = detected_iccs[KP.LEFT_HIP], detected_iccs[KP.RIGHT_HIP]
        else:
            return LIMB_FIT_DEPTH_DAMPING_FRONTAL
        dx = abs(float(a[0] - b[0]))   # lateral separation (ICCS X)
        dy = abs(float(a[1] - b[1]))   # depth separation   (ICCS Y)
        span = dx + dy + 1e-9
        # lateral_factor: 1 = full profile (all separation in depth), 0 = frontal
        lateral_factor = dy / span
        return (LIMB_FIT_DEPTH_DAMPING_LATERAL * lateral_factor
                + LIMB_FIT_DEPTH_DAMPING_FRONTAL * (1.0 - lateral_factor))
    except Exception:
        return LIMB_FIT_DEPTH_DAMPING_FRONTAL


class AnatomicalSkeleton:
    """
    Anatomical skeleton with World-Aligned ICCS.
    
    21 keypoints:
      - 0-16: COCO keypoints (from MMPose detection)
      - 17: HEAD_CENTER (midpoint between ears)
      - 18: SHOULDER_CENTER (midpoint between shoulders)
      - 19: PELVIS_CENTER (midpoint between hips) = ICCS origin
      - 20: SPINE_MID (midpoint between shoulder_center and pelvis_center)
    
    The skeleton stores keypoints in ICCS coordinates where:
      - X = left(-)/right(+) relative to body center
      - Y = back(-)/forward(+) relative to facing direction
      - Z = height relative to pelvis (DOWN is negative, UP is positive)
    
    SKELETAL HIERARCHY:
    
        PELVIS_CENTER (19) ── ROOT
            ├── LEFT_HIP ── LEFT_KNEE ── LEFT_ANKLE
            ├── RIGHT_HIP ── RIGHT_KNEE ── RIGHT_ANKLE
            └── SPINE_MID (20)
                    └── SHOULDER_CENTER (18)
                            ├── LEFT_SHOULDER ── LEFT_ELBOW ── LEFT_WRIST
                            ├── RIGHT_SHOULDER ── RIGHT_ELBOW ── RIGHT_WRIST
                            └── HEAD_CENTER (17)
                                    └── NOSE ── EYES ── EARS
    """
    
    def __init__(self, height_cm: float = 170.0):
        """
        Initialize skeleton.
        
        Args:
            height_cm: Person's height in centimeters
        """
        self.height = height_cm
        
        # ICCS - root coordinate system
        self.iccs = ICCS()
        
        # Segments with joint limits
        self.segments = create_segments()
        
        # Bone lengths — derived deterministically from height via anatomical ratios.
        # They are LOCKED immediately: once height is known, anatomy is known.
        # No multi-frame calibration window is needed or correct.
        self.bone_lengths: Dict[str, float] = {}
        self._init_bone_lengths_from_height(height_cm)
        self.is_calibrated = True   # ← locked from frame 1 by anatomy, not MMPose samples
        
        # Keypoints in ICCS coordinates (21 x 3) - includes extended joints!
        self.keypoints_iccs = np.zeros((NUM_KEYPOINTS, 3))
        self._init_rest_pose_iccs()
        
        # ── RIGID HEAD TEMPLATE ─────────────────────────────────────────
        # The head is a rigid isosceles trapezoid.  Internal dimensions are
        # computed ONCE from anthropometric bone_lengths and NEVER change.
        # Each frame only the HEAD_CENTER position and the head's 3D
        # orientation change — the template is rotated as a rigid body.
        self.rigid_head_template: Dict[int, np.ndarray] = {}
        self._build_rigid_head_template()

        # Keypoints in world coordinates (21 x 3) - updated by kinematics FK
        self.keypoints_world = np.zeros((NUM_KEYPOINTS, 3))
        
        # Calibration data
        self._calibration_samples: List[Dict[str, float]] = []

        self._kinematics = None
        
        # ----- PERSISTENT TEMPORAL STATE (Bug 9 fix) -----
        # These survive across frames because the skeleton instance persists
        # in ClusterStateBank.skeletons[uuid]. They enable:
        #   - Pelvis velocity clamping (Bug 2)
        #   - Per-joint velocity limits (Bug 3)
        #   - Per-joint fallback on bad fits (Bug 7)
        self.previous_keypoints_iccs = None   # (21,3) ndarray or None — last frame's ICCS positions
        self.previous_keypoints_world = None  # (21,3) ndarray or None — last frame's world positions
        self.previous_pelvis_world = None     # (3,) ndarray or None — last frame's world pelvis
        self._previous_yaw = None             # float or None — last frame's yaw angle (degrees)
        self.previous_segment_angles = None   # Dict[str, Tuple[float,float,float]] — last frame's segment angles (V5 angular velocity clamping)
        self.frame_count = 0                  # How many frames this skeleton has been fitted through
        self.fitting_errors_history = {}      # kp_idx -> last frame's fitting error (cm)
        
        logger.info(f"AnatomicalSkeleton created: height={height_cm:.1f}cm, {NUM_KEYPOINTS} keypoints")
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _init_bone_lengths_from_height(self, height_cm: float):
        """Initialize bone lengths from anthropometric ratios"""
        H = height_cm
        torso = H * ANTHROPOMETRIC_RATIOS['torso']
        head = H * ANTHROPOMETRIC_RATIOS['head']
        
        self.bone_lengths = {
            # Core structure
            'shoulder_width': H * ANTHROPOMETRIC_RATIOS['shoulder_width'],
            'hip_width': H * ANTHROPOMETRIC_RATIOS['hip_width'],
            'torso': torso,
            
            # Spine segments
            'lower_spine': torso * ANTHROPOMETRIC_RATIOS['lower_spine_ratio'],
            'upper_spine': torso * ANTHROPOMETRIC_RATIOS['upper_spine_ratio'],
            'neck': head * ANTHROPOMETRIC_RATIOS['neck_ratio'],
            'head': head * (1 - ANTHROPOMETRIC_RATIOS['neck_ratio']),
            
            # Arms
            'upper_arm_l': H * ANTHROPOMETRIC_RATIOS['upper_arm'],
            'upper_arm_r': H * ANTHROPOMETRIC_RATIOS['upper_arm'],
            'forearm_l': H * ANTHROPOMETRIC_RATIOS['forearm'],
            'forearm_r': H * ANTHROPOMETRIC_RATIOS['forearm'],
            
            # Legs
            'thigh_l': H * ANTHROPOMETRIC_RATIOS['thigh'],
            'thigh_r': H * ANTHROPOMETRIC_RATIOS['thigh'],
            'shin_l': H * ANTHROPOMETRIC_RATIOS['shin'],
            'shin_r': H * ANTHROPOMETRIC_RATIOS['shin'],

            # ---- Face / head geometry (absolute cm, fixed for adults) ----
            # Used by _enforce_face_geometry() to place eyes and nose from ears.
            # Stored in bone_lengths for calibration / override capability.
            'face_half_ear_width':  ANTHROPOMETRIC_RATIOS['face_half_ear_width'],
            'face_eye_lateral_sep': ANTHROPOMETRIC_RATIOS['face_eye_lateral_sep'],
            'face_eye_above_ears':  ANTHROPOMETRIC_RATIOS['face_eye_above_ears'],
            'face_nose_protrusion': ANTHROPOMETRIC_RATIOS['face_nose_protrusion'],
            'face_nose_below_eyes': ANTHROPOMETRIC_RATIOS['face_nose_below_eyes'],
        }
        
        # Update segment bone lengths
        for name, segment in self.segments.items():
            if name in self.bone_lengths:
                segment.bone_length = self.bone_lengths[name]
    
    def _init_rest_pose_iccs(self):
        """
        Initialize keypoints in ICCS for T-pose / rest pose.
        
        In ICCS:
          - Origin (0,0,0) = pelvis center = keypoint 19
          - X+ = right, X- = left
          - Y+ = forward, Y- = back
          - Z+ = up, Z- = down
        
        Extended keypoints (17-20) are computed from COCO keypoints.
        """
        kp = self.keypoints_iccs
        
        # Get dimensions
        hip_w = self.bone_lengths['hip_width'] / 2
        shoulder_w = self.bone_lengths['shoulder_width'] / 2
        lower_spine = self.bone_lengths['lower_spine']
        upper_spine = self.bone_lengths['upper_spine']
        neck_h = self.bone_lengths['neck']
        head_h = self.bone_lengths['head']
        upper_arm = self.bone_lengths['upper_arm_l']
        forearm = self.bone_lengths['forearm_l']
        thigh = self.bone_lengths['thigh_l']
        shin = self.bone_lengths['shin_l']
        
        torso_h = lower_spine + upper_spine
        
        # =====================================================================
        # EXTENDED JOINTS (17-20) - Spine chain
        # =====================================================================
        
        # PELVIS_CENTER (19) - ICCS origin
        kp[KP.PELVIS_CENTER] = np.array([0, 0, 0])
        
        # SPINE_MID (20) - halfway up torso
        kp[KP.SPINE_MID] = np.array([0, 0, lower_spine])
        
        # SHOULDER_CENTER (18) - top of spine
        kp[KP.SHOULDER_CENTER] = np.array([0, 0, torso_h])
        
        # HEAD_CENTER (17) - base of skull (between ears)
        kp[KP.HEAD_CENTER] = np.array([0, 0, torso_h + neck_h])
        
        # =====================================================================
        # COCO JOINTS (0-16)
        # =====================================================================
        
        # ----- HIPS (at origin level, Z=0) -----
        kp[KP.LEFT_HIP] = np.array([-hip_w, 0, 0])
        kp[KP.RIGHT_HIP] = np.array([+hip_w, 0, 0])
        
        # ----- SHOULDERS (at shoulder_center level) -----
        kp[KP.LEFT_SHOULDER] = np.array([-shoulder_w, 0, torso_h])
        kp[KP.RIGHT_SHOULDER] = np.array([+shoulder_w, 0, torso_h])
        
        # ----- HEAD — rest pose satisfies all three face constraints -----
        # Ear separation and positions define the face plane.
        # HEAD_CENTER = midpoint(ears).  Eyes coplanar with ears.
        # Nose perpendicular from eye midpoint.
        bl = self.bone_lengths
        half_ear_w     = bl.get('face_half_ear_width',  8.5)
        eye_lat        = bl.get('face_eye_lateral_sep', 3.2)
        eye_above_ears = bl.get('face_eye_above_ears',  2.5)
        nose_protrude  = bl.get('face_nose_protrusion', 3.0)
        nose_below_eye = bl.get('face_nose_below_eyes', 1.5)

        ear_z          = torso_h + neck_h               # ear height (base of skull)
        head_ctr_z     = ear_z                          # HEAD_CENTER at ear level

        # CONSTRAINT 1: ears collinear with HEAD_CENTER
        kp[KP.LEFT_EAR]   = np.array([-half_ear_w, 0.0, ear_z])
        kp[KP.RIGHT_EAR]  = np.array([+half_ear_w, 0.0, ear_z])
        kp[KP.HEAD_CENTER] = np.array([0.0, 0.0, head_ctr_z])  # midpoint(ears)

        # CONSTRAINT 2: eyes coplanar with ears (Y=0, same frontal plane)
        # In rest pose face_normal = +Y (facing forward)
        eye_z = ear_z + eye_above_ears
        kp[KP.LEFT_EYE]  = np.array([-eye_lat, 0.0, eye_z])
        kp[KP.RIGHT_EYE] = np.array([+eye_lat, 0.0, eye_z])

        # CONSTRAINT 3: nose = eye_midpoint + face_normal * protrusion
        # face_normal in rest pose = +Y (forward)
        eye_mid_z = eye_z
        kp[KP.NOSE] = np.array([0.0, nose_protrude, eye_mid_z - nose_below_eye])
        
        # ----- ARMS (hanging at sides in rest pose) -----
        # FIX A8: removed the +5cm fixed lateral offset so rest-pose bone
        # distances exactly match bone_lengths['upper_arm_l/r'].  The +5 caused
        # FK to propagate elbows/wrists 5cm too far laterally on every frame.
        # Left arm
        kp[KP.LEFT_ELBOW] = np.array([-shoulder_w, 0, torso_h - upper_arm])
        kp[KP.LEFT_WRIST] = np.array([-shoulder_w, 0, torso_h - upper_arm - forearm])

        # Right arm
        kp[KP.RIGHT_ELBOW] = np.array([+shoulder_w, 0, torso_h - upper_arm])
        kp[KP.RIGHT_WRIST] = np.array([+shoulder_w, 0, torso_h - upper_arm - forearm])
        
        # ----- LEGS (straight down) -----
        # Left leg
        kp[KP.LEFT_KNEE] = np.array([-hip_w, 0, -thigh])
        kp[KP.LEFT_ANKLE] = np.array([-hip_w, 0, -thigh - shin])
        
        # Right leg
        kp[KP.RIGHT_KNEE] = np.array([+hip_w, 0, -thigh])
        kp[KP.RIGHT_ANKLE] = np.array([+hip_w, 0, -thigh - shin])
        
        logger.debug(f"Rest pose initialized: {NUM_KEYPOINTS} keypoints in ICCS")

    def _build_rigid_head_template(self):
        """
        Build the RIGID head trapezoid template — local 3D offsets from
        HEAD_CENTER that NEVER change between frames.

        The head is a regular isosceles trapezoid:

            L_EAR ━━━━ HEAD_CENTER ━━━━ R_EAR    longer base (ear line)
               ╲                          ╱
                ╲   (equal legs)        ╱        isosceles
                 ╲                    ╱
            L_EYE ━━━ eye_mid ━━━ R_EYE          shorter base (eye line)
                       │
                       │  nose_line
                       │
                      NOSE

        Local coordinate system (aligned with ICCS rest-pose):
          X = lateral   (left_ear → right_ear)
          Y = forward   (face/nose protrusion direction)
          Z = up        (ears → eyes is +Z)

        HEAD_CENTER is at local origin [0, 0, 0].
        Ears are at Z=0 (same height as HEAD_CENTER).
        Eyes are above ears, coplanar in Y with ears (Y=0).
        Nose protrudes in +Y from the eye midpoint.

        All values come from bone_lengths (set once from anthropometric
        ratios in _init_bone_lengths_from_height) and are LOCKED.
        """
        bl = self.bone_lengths
        half_ear_w     = bl.get('face_half_ear_width',  8.5)
        eye_lat        = bl.get('face_eye_lateral_sep', 3.2)
        eye_above_ears = bl.get('face_eye_above_ears',  2.5)
        nose_protrude  = bl.get('face_nose_protrusion', 3.0)
        nose_below_eye = bl.get('face_nose_below_eyes', 1.5)

        # HEAD_CENTER = local origin
        self.rigid_head_template = {
            KP.LEFT_EAR:  np.array([-half_ear_w, 0.0, 0.0]),
            KP.RIGHT_EAR: np.array([+half_ear_w, 0.0, 0.0]),
            KP.LEFT_EYE:  np.array([-eye_lat,    0.0, +eye_above_ears]),
            KP.RIGHT_EYE: np.array([+eye_lat,    0.0, +eye_above_ears]),
            KP.NOSE:      np.array([0.0, +nose_protrude, +eye_above_ears - nose_below_eye]),
        }

        logger.info(
            f"[RigidHead] Template built: ear_width={2*half_ear_w:.1f}cm, "
            f"eye_sep={2*eye_lat:.1f}cm, nose_protrude={nose_protrude:.1f}cm"
        )
    
# ─────────────────────────────────────────────────────────────────────────────

    def _store_temporal_state(self, keypoints_world_21=None, fitting_errors=None):
        """
        Store current frame's state for next-frame temporal guards.
        
        Called at the END of each successful shell fit, AFTER STEP 10
        (bone length enforcement) and STEP 11 (angle back-calculation).
        
        This is how the skeleton "remembers where it was." Next frame,
        the shell fitter can compare new positions against these stored
        values and clamp impossible velocities.
        
        Args:
            keypoints_world_21: (21,3) world-space keypoints after fitting.
                If None, computed from self.keypoints_iccs via iccs_to_world.
            fitting_errors: Dict[int, float] — per-keypoint fitting errors (cm).
                Stored for next frame's per-joint confidence gating.
        """
        # Store ICCS positions (always available after fitting)
        self.previous_keypoints_iccs = self.keypoints_iccs.copy()
        
        # Store world positions
        if keypoints_world_21 is not None:
            self.previous_keypoints_world = np.array(keypoints_world_21).copy()
        
        # Store world pelvis = ICCS origin in world coordinates
        if self.iccs.origin is not None:
            self.previous_pelvis_world = np.array(self.iccs.origin).copy()
        
        # Store fitting errors for per-joint confidence gating
        if fitting_errors is not None:
            self.fitting_errors_history = dict(fitting_errors)
        
        self.frame_count += 1
        
        logger.debug(f"[TEMPORAL] Stored state: frame_count={self.frame_count}, "
                    f"pelvis_world={self.previous_pelvis_world}")
  
    def get_kinematics(self) -> 'SkeletonKinematics':
        """
        Get or create kinematics engine for this skeleton.
        
        Returns:
            SkeletonKinematics instance
        """
        if not hasattr(self, '_kinematics') or self._kinematics is None:
            self._kinematics = SkeletonKinematics(self)
        return self._kinematics
    
    def get_shell_fitter(self) -> 'SkeletonShellFitter':
        """
        Get or create shell fitter for this skeleton.
        
        The shell fitter is used to fit skeleton joints to cluster shell
        Y-plane cell centroids using IK.
        
        Returns:
            SkeletonShellFitter instance
        """
        if not hasattr(self, '_shell_fitter') or self._shell_fitter is None:
            self._shell_fitter = SkeletonShellFitter(self)
        return self._shell_fitter
    
    def fit_to_cluster_shell(self,
                             voxel_grid,
                             keypoints_2d_mapping: List[Dict],
                             keypoints_3d_mapping: List[Dict],
                             cluster_voxel_indices: Set[Tuple[int, int, int]],
                             facing_direction: str = 'toward_camera',
                             camera_params: Optional[Dict] = None,
                             ply_mesh=None,
                             poisson_ply_path: Optional[str] = None,
                             body_yaw_deg: Optional[float] = None,
                             spine_curve: Optional[list] = None,
                             mp33_arm_extended: bool = False,
                             pose_dof: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fit skeleton joints to cluster shell Y-plane cell centroids.

        mp33_arm_extended: when True, Option 4 prefers camera-side (p_near)
            for arm joints on LATERAL views.
        pose_dof: dict of {segment_name: {rx, ry, rz}} from POSE_DB match.
            When provided, these angles are applied via set_angles + propagate_fk
            after OPT4 placement, making the mannequin flexible.
        """
        fitter = self.get_shell_fitter()
        return fitter.fit_to_cluster_shell(
            voxel_grid=voxel_grid,
            keypoints_2d_mapping=keypoints_2d_mapping,
            keypoints_3d_mapping=keypoints_3d_mapping,
            cluster_voxel_indices=cluster_voxel_indices,
            facing_direction=facing_direction,
            camera_params=camera_params,
            ply_mesh=ply_mesh,
            poisson_ply_path=poisson_ply_path,
            body_yaw_deg=body_yaw_deg,
            spine_curve=spine_curve,
            mp33_arm_extended=mp33_arm_extended,
            pose_dof=pose_dof,
        )

    def fit_to_ply_surface(self,
                           ply_verts_world: np.ndarray,
                           canonical_kps_world: np.ndarray,
                           facing_direction: str = 'toward_camera') -> np.ndarray:
        """
        Fit a canonical Skeleton-21 humanoid to a Poisson PLY surface.

        This is the PLY-space equivalent of fit_to_cluster_shell().  Instead of
        fitting to voxel-grid Y-plane centroids it fits to the triangulated
        Poisson surface mesh — the same mesh exported to poisson_humanoid/*.ply.

        Args:
            ply_verts_world:     (N,3) Poisson mesh vertices converted to WORLD
                                 coords (use _ply_to_world_verts() helper).
            canonical_kps_world: (21,3) starting humanoid positions in world
                                 coords (from build_skeleton21_from_cluster_bbox
                                 or shell_fitted_21 joints).
            facing_direction:    'toward_camera' or 'away_from_camera'.

        Returns:
            (21,3) fitted world-space keypoints, or canonical_kps_world unchanged
            on failure.
        """
        fitter = self.get_shell_fitter()
        return fitter.fit_to_ply_surface(ply_verts_world, canonical_kps_world,
                                         facing_direction)

    # =========================================================================
    # CALIBRATION
    # =========================================================================
    
    def calibrate(self, keypoints_world: np.ndarray, confidence: float = 1.0) -> bool:
        """
        Calibrate skeleton from detected 3D pose.
        
        Args:
            keypoints_world: 17x3 COCO keypoints in world coordinates (MMPose output)
            confidence: Detection confidence (0-1)
            
        Returns:
            True if calibration successful
        """
        if keypoints_world is None or len(keypoints_world) < 17:
            return False
        
        # Check for valid keypoints (at least 8 needed)
        valid = np.abs(keypoints_world[:17]).sum(axis=1) > 0.1
        if valid.sum() < 8:
            logger.warning("Not enough valid keypoints for calibration")
            return False
        
        # Compute extended keypoints from COCO keypoints
        extended_world = self._compute_extended_keypoints(keypoints_world)
        
        # Update ICCS from keypoints
        if not self.iccs.update_from_keypoints(extended_world):
            return False
        
        # Measure bone lengths
        lengths = self._measure_bone_lengths(extended_world)
        
        if len(lengths) < 4:
            logger.warning("Could not measure enough bones for calibration")
            return False
        
        # Store sample
        self._calibration_samples.append({
            'lengths': lengths,
            'confidence': confidence
        })
        
        # Update bone lengths from samples
        self._update_bone_lengths_from_samples()
        
        # Update keypoints_iccs from current detection
        self._update_keypoints_iccs(extended_world)
        
        # Mark calibrated after enough samples
        if len(self._calibration_samples) >= 3:
            self.is_calibrated = True
            logger.info(f"Skeleton calibrated from {len(self._calibration_samples)} samples")
        
        return True
    
    def _compute_extended_keypoints(self, coco_keypoints: np.ndarray) -> np.ndarray:
        """
        Compute extended keypoints (17-20) from COCO keypoints (0-16).
        
        Args:
            coco_keypoints: 17x3 COCO keypoints
            
        Returns:
            21x3 array with all keypoints (COCO + extended)
        """
        # Start with zeros for all 21 keypoints
        extended = np.zeros((NUM_KEYPOINTS, 3))
        
        # Copy COCO keypoints
        extended[:17] = coco_keypoints[:17]
        
        # HEAD_CENTER (17) = midpoint between ears
        left_ear = coco_keypoints[KP.LEFT_EAR]
        right_ear = coco_keypoints[KP.RIGHT_EAR]
        if not np.allclose(left_ear, 0) and not np.allclose(right_ear, 0):
            extended[KP.HEAD_CENTER] = (left_ear + right_ear) / 2
        else:
            # Fallback: estimate from nose
            nose = coco_keypoints[KP.NOSE]
            if not np.allclose(nose, 0):
                extended[KP.HEAD_CENTER] = nose - np.array([0, 8, 5])  # Behind and below nose
        
        # SHOULDER_CENTER (18) = midpoint between shoulders
        left_shoulder = coco_keypoints[KP.LEFT_SHOULDER]
        right_shoulder = coco_keypoints[KP.RIGHT_SHOULDER]
        if not np.allclose(left_shoulder, 0) and not np.allclose(right_shoulder, 0):
            extended[KP.SHOULDER_CENTER] = (left_shoulder + right_shoulder) / 2
        
        # PELVIS_CENTER (19) = midpoint between hips
        left_hip = coco_keypoints[KP.LEFT_HIP]
        right_hip = coco_keypoints[KP.RIGHT_HIP]
        if not np.allclose(left_hip, 0) and not np.allclose(right_hip, 0):
            extended[KP.PELVIS_CENTER] = (left_hip + right_hip) / 2
        
        # SPINE_MID (20) = midpoint between shoulder_center and pelvis_center
        shoulder_center = extended[KP.SHOULDER_CENTER]
        pelvis_center = extended[KP.PELVIS_CENTER]
        if not np.allclose(shoulder_center, 0) and not np.allclose(pelvis_center, 0):
            extended[KP.SPINE_MID] = (shoulder_center + pelvis_center) / 2
        
        return extended
    
    def _measure_bone_lengths(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Measure bone lengths from keypoints (including extended)"""
        lengths = {}
        
        def measure(name, idx1, idx2):
            p1, p2 = keypoints[idx1], keypoints[idx2]
            if not np.allclose(p1, 0) and not np.allclose(p2, 0):
                lengths[name] = np.linalg.norm(p2 - p1)
        
        # Core structure
        measure('hip_width', KP.LEFT_HIP, KP.RIGHT_HIP)
        measure('shoulder_width', KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER)
        
        # Spine segments (using extended keypoints)
        measure('lower_spine', KP.PELVIS_CENTER, KP.SPINE_MID)
        measure('upper_spine', KP.SPINE_MID, KP.SHOULDER_CENTER)
        measure('neck', KP.SHOULDER_CENTER, KP.HEAD_CENTER)
        
        # Total torso (for validation)
        measure('torso', KP.PELVIS_CENTER, KP.SHOULDER_CENTER)
        
        # Arms
        measure('upper_arm_l', KP.LEFT_SHOULDER, KP.LEFT_ELBOW)
        measure('upper_arm_r', KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW)
        measure('forearm_l', KP.LEFT_ELBOW, KP.LEFT_WRIST)
        measure('forearm_r', KP.RIGHT_ELBOW, KP.RIGHT_WRIST)
        
        # Legs
        measure('thigh_l', KP.LEFT_HIP, KP.LEFT_KNEE)
        measure('thigh_r', KP.RIGHT_HIP, KP.RIGHT_KNEE)
        measure('shin_l', KP.LEFT_KNEE, KP.LEFT_ANKLE)
        measure('shin_r', KP.RIGHT_KNEE, KP.RIGHT_ANKLE)
        
        return lengths
    
    def _update_bone_lengths_from_samples(self):
        """Update bone lengths from calibration samples (weighted average)"""
        if not self._calibration_samples:
            return
        
        for bone_name in self.bone_lengths.keys():
            values, weights = [], []
            for sample in self._calibration_samples:
                if bone_name in sample['lengths']:
                    values.append(sample['lengths'][bone_name])
                    weights.append(sample['confidence'])
            
            if values:
                self.bone_lengths[bone_name] = np.average(values, weights=weights)
        
        # Enforce symmetry
        for left, right in SYMMETRIC_BONES:
            if left in self.bone_lengths and right in self.bone_lengths:
                avg = (self.bone_lengths[left] + self.bone_lengths[right]) / 2
                self.bone_lengths[left] = avg
                self.bone_lengths[right] = avg
        
        # Update segments
        for name, seg in self.segments.items():
            if name in self.bone_lengths:
                seg.bone_length = self.bone_lengths[name]
    
    def _update_keypoints_iccs(self, keypoints_world: np.ndarray):
        """Update keypoints_iccs from world keypoints (handles both 17 and 21 keypoints)"""
        n_kp = min(len(keypoints_world), NUM_KEYPOINTS)
        for i in range(n_kp):
            if not np.allclose(keypoints_world[i], 0):
                self.keypoints_iccs[i] = self.iccs.world_to_iccs(keypoints_world[i])
    
    # =========================================================================
    # POSE FITTING
    # =========================================================================
    
    def fit_to_detection(self, 
                         keypoints_world: np.ndarray,
                         cluster_centroid: Optional[np.ndarray] = None,
                         rotation_override: Optional[float] = None) -> np.ndarray:
        """
        Fit skeleton to detected keypoints.
        
        Updates ICCS and keypoints_iccs from detection, enforcing bone lengths.
        
        Args:
            keypoints_world: Detected 17x3 COCO keypoints in world coordinates
            cluster_centroid: Optional cluster centroid (uses pelvis center if None)
            rotation_override: Optional rotation angle override
            
        Returns:
            21x3 fitted keypoints in world coordinates (COCO + extended)
        """
        # Compute extended keypoints
        extended_world = self._compute_extended_keypoints(keypoints_world)
        
        # Update ICCS from keypoints (or use overrides)
        if rotation_override is not None and cluster_centroid is not None:
            self.iccs.update(cluster_centroid, rotation_override)
        else:
            self.iccs.update_from_keypoints(extended_world)
        
        # Convert detected keypoints to ICCS
        detected_iccs = np.zeros((NUM_KEYPOINTS, 3))
        valid = np.zeros(NUM_KEYPOINTS, dtype=bool)
        
        for i in range(NUM_KEYPOINTS):
            if not np.allclose(extended_world[i], 0):
                detected_iccs[i] = self.iccs.world_to_iccs(extended_world[i])
                valid[i] = True
        
        # Fit skeleton with bone length constraints
        fitted_iccs = self._fit_with_bone_constraints(detected_iccs, valid)
        
        # Store result
        self.keypoints_iccs = fitted_iccs
        
        # Convert back to world
        return self.iccs.iccs_to_world_batch(fitted_iccs)
    
    def fit_to_cluster(self,
                       cluster_centroid: np.ndarray,
                       rotation_angle: float,
                       detected_keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit skeleton to cluster (without full keypoint detection).
        
        Args:
            cluster_centroid: Cluster center in world coordinates
            rotation_angle: Body yaw rotation in degrees
            detected_keypoints: Optional sparse COCO keypoints for refinement
            
        Returns:
            21x3 keypoints in world coordinates
        """
        # Update ICCS
        self.iccs.update(cluster_centroid, rotation_angle)
        
        # If we have detected keypoints, use them to refine pose
        if detected_keypoints is not None:
            return self.fit_to_detection(detected_keypoints, cluster_centroid, rotation_angle)
        
        # Otherwise use rest pose
        return self.get_keypoints_world()
    
    def _fit_with_bone_constraints(self, detected_iccs: np.ndarray, 
                                    valid: np.ndarray) -> np.ndarray:
        """
        Fit detected keypoints while enforcing bone length constraints.
        
        Strategy:
          1. Use detected positions for core keypoints (hips, shoulders)
          2. Propagate to limbs using bone lengths
          3. Blend with detected positions where available
        """
        fitted = self.keypoints_iccs.copy()  # Start with rest pose
        
        # ----- CORE: Hips and Shoulders -----
        # These define the torso frame
        
        if valid[KP.LEFT_HIP] and valid[KP.RIGHT_HIP]:
            # Use detected hip positions (defines hip tilt)
            fitted[KP.LEFT_HIP] = detected_iccs[KP.LEFT_HIP]
            fitted[KP.RIGHT_HIP] = detected_iccs[KP.RIGHT_HIP]
            
            # Enforce hip width
            hip_center = (fitted[KP.LEFT_HIP] + fitted[KP.RIGHT_HIP]) / 2
            hip_dir = fitted[KP.RIGHT_HIP] - fitted[KP.LEFT_HIP]
            hip_dir_norm = hip_dir / (np.linalg.norm(hip_dir) + 1e-10)
            half_width = self.bone_lengths['hip_width'] / 2
            
            fitted[KP.LEFT_HIP] = hip_center - hip_dir_norm * half_width
            fitted[KP.RIGHT_HIP] = hip_center + hip_dir_norm * half_width
        
        if valid[KP.LEFT_SHOULDER] and valid[KP.RIGHT_SHOULDER]:
            fitted[KP.LEFT_SHOULDER] = detected_iccs[KP.LEFT_SHOULDER]
            fitted[KP.RIGHT_SHOULDER] = detected_iccs[KP.RIGHT_SHOULDER]
            
            # Enforce shoulder width
            shoulder_center = (fitted[KP.LEFT_SHOULDER] + fitted[KP.RIGHT_SHOULDER]) / 2
            shoulder_dir = fitted[KP.RIGHT_SHOULDER] - fitted[KP.LEFT_SHOULDER]
            shoulder_dir_norm = shoulder_dir / (np.linalg.norm(shoulder_dir) + 1e-10)
            half_width = self.bone_lengths['shoulder_width'] / 2
            
            fitted[KP.LEFT_SHOULDER] = shoulder_center - shoulder_dir_norm * half_width
            fitted[KP.RIGHT_SHOULDER] = shoulder_center + shoulder_dir_norm * half_width
        
        # ----- LIMBS: Propagate with bone lengths -----
        
        # View-aware depth damping (harder in profile, where ICCS-Y is least
        # reliable and the far-side limb is occluded).
        _depth_damp = _limb_depth_damping_for_view(detected_iccs, valid)

        # Arms.  DoF FIX: same depth-damp + tilt-clamp as legs.  Upper arm is
        # clamped toward straight-down (arms-down rest); forearm is clamped
        # relative to the upper arm so real elbow bend is preserved but the
        # forearm cannot shoot out horizontally (the "arm jutting forward").
        for side, shoulder_kp, elbow_kp, wrist_kp in [
            ('l', KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST),
            ('r', KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST)
        ]:
            shoulder = fitted[shoulder_kp]

            # Elbow (upper arm)
            if valid[elbow_kp]:
                elbow_dir = detected_iccs[elbow_kp] - shoulder
                elbow_dir = _limb_dir_sanitize(
                    elbow_dir, _depth_damp, ARM_FIT_MAX_TILT_FROM_DOWN_DEG)
                fitted[elbow_kp] = shoulder + elbow_dir * self.bone_lengths[f'upper_arm_{side}']

            # Wrist (forearm) — clamp relative to the upper arm
            elbow = fitted[elbow_kp]
            if valid[wrist_kp]:
                wrist_dir = detected_iccs[wrist_kp] - elbow
                upper_dir = elbow - shoulder
                wrist_dir = _limb_dir_sanitize(
                    wrist_dir, _depth_damp, ARM_FIT_MAX_TILT_FROM_DOWN_DEG,
                    reference_vec=upper_dir)
                fitted[wrist_kp] = elbow + wrist_dir * self.bone_lengths[f'forearm_{side}']

        # Legs
        for side, hip_kp, knee_kp, ankle_kp in [
            ('l', KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE),
            ('r', KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE)
        ]:
            hip = fitted[hip_kp]

            # Knee (thigh).  DoF FIX: damp the unreliable ICCS-Y (depth) part of
            # the detected direction and clamp the thigh's tilt away from
            # straight-down, so a standing leg cannot be planted horizontal.
            if valid[knee_kp]:
                knee_dir = detected_iccs[knee_kp] - hip
                knee_dir = _limb_dir_sanitize(
                    knee_dir, _depth_damp, LEG_FIT_MAX_TILT_FROM_DOWN_DEG)
                fitted[knee_kp] = hip + knee_dir * self.bone_lengths[f'thigh_{side}']

            # Ankle (shin).  Reference is the (already-sanitized) thigh
            # direction, so genuine knee bend relative to the thigh is preserved
            # while the same depth damping is applied.
            knee = fitted[knee_kp]
            if valid[ankle_kp]:
                ankle_dir = detected_iccs[ankle_kp] - knee
                thigh_dir = knee - hip
                ankle_dir = _limb_dir_sanitize(
                    ankle_dir, _depth_damp, LEG_FIT_MAX_TILT_FROM_DOWN_DEG,
                    reference_vec=thigh_dir)
                fitted[ankle_kp] = knee + ankle_dir * self.bone_lengths[f'shin_{side}']
        
        # ----- HEAD -----
        if valid[KP.NOSE]:
            # Keep detected nose position (head orientation)
            fitted[KP.NOSE] = detected_iccs[KP.NOSE]
        
        for kp in [KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR]:
            if valid[kp]:
                fitted[kp] = detected_iccs[kp]
        
        return fitted
    
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def get_keypoints_world(self) -> np.ndarray:
        """Get keypoints in world coordinates"""
        return self.iccs.iccs_to_world_batch(self.keypoints_iccs)
    
    def get_keypoints_iccs(self) -> np.ndarray:
        """Get keypoints in ICCS coordinates"""
        return self.keypoints_iccs.copy()
    
    def get_iccs(self) -> ICCS:
        """Get the ICCS object"""
        return self.iccs
    
    def get_bone_length(self, name: str) -> float:
        """Get bone length by name"""
        return self.bone_lengths.get(name, 0.0)
    
    def get_segment(self, name: str) -> Optional[SegmentLCS]:
        """Get segment by name"""
        return self.segments.get(name)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_keypoints(self, keypoints_iccs: np.ndarray) -> Dict:
        """
        Validate keypoints against anatomical constraints.
        
        Returns dict with validation results.
        """
        results = {
            'valid': True,
            'bone_length_errors': {},
            'joint_limit_violations': []
        }
        
        # Check bone lengths
        bone_checks = [
            ('hip_width', KP.LEFT_HIP, KP.RIGHT_HIP),
            ('shoulder_width', KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER),
            ('upper_arm_l', KP.LEFT_SHOULDER, KP.LEFT_ELBOW),
            ('upper_arm_r', KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW),
            ('forearm_l', KP.LEFT_ELBOW, KP.LEFT_WRIST),
            ('forearm_r', KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
            ('thigh_l', KP.LEFT_HIP, KP.LEFT_KNEE),
            ('thigh_r', KP.RIGHT_HIP, KP.RIGHT_KNEE),
            ('shin_l', KP.LEFT_KNEE, KP.LEFT_ANKLE),
            ('shin_r', KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
        ]
        
        for name, idx1, idx2 in bone_checks:
            p1, p2 = keypoints_iccs[idx1], keypoints_iccs[idx2]
            if np.allclose(p1, 0) or np.allclose(p2, 0):
                continue
            
            measured = np.linalg.norm(p2 - p1)
            expected = self.bone_lengths.get(name, measured)
            error = abs(measured - expected) / expected if expected > 0 else 0
            
            if error > 0.1:  # More than 10% error
                results['bone_length_errors'][name] = {
                    'expected': expected,
                    'measured': measured,
                    'error_pct': error * 100
                }
                results['valid'] = False
        
        return results
    
    # =========================================================================
    # DEBUG
    # =========================================================================
    
    def print_info(self):
        """Print skeleton information"""
        print("\n" + "=" * 60)
        print("ANATOMICAL SKELETON")
        print("=" * 60)
        print(f"Height: {self.height:.1f} cm")
        print(f"Calibrated: {self.is_calibrated}")
        print(f"Calibration samples: {len(self._calibration_samples)}")
        
        print(f"\nICCS:")
        print(f"  Origin: [{self.iccs.origin[0]:.1f}, {self.iccs.origin[1]:.1f}, {self.iccs.origin[2]:.1f}]")
        print(f"  Yaw: {self.iccs.yaw:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
        print(f"  X-axis (right): {self.iccs.x_axis}")
        print(f"  Y-axis (forward): {self.iccs.y_axis}")
        print(f"  Z-axis (up): {self.iccs.z_axis}")
        
        print(f"\nBone lengths (cm):")
        for name in sorted(self.bone_lengths.keys()):
            print(f"  {name}: {self.bone_lengths[name]:.1f}")
        
        print(f"\nKeypoints in ICCS:")
        for i, name in enumerate(KEYPOINT_NAMES):
            kp = self.keypoints_iccs[i]
            print(f"  {i:2d} {name:15s}: [{kp[0]:7.1f}, {kp[1]:7.1f}, {kp[2]:7.1f}]")
        
        print("=" * 60 + "\n")


class SkeletonKinematics:
    """
    Forward and Inverse Kinematics engine for AnatomicalSkeleton.
    
    Provides:
    1. FK: Set joint angles [OK] propagate positions down kinematic chain
    2. IK: Set end-effector target [OK] solve joint angles (iterative)
    3. Y-plane snapping: Align keypoints to populated cluster voxels
    
    KINEMATIC CHAINS (from PELVIS_CENTER root):
    
    LEFT LEG:   PELVIS_CENTER [OK] LEFT_HIP [OK] LEFT_KNEE [OK] LEFT_ANKLE
    RIGHT LEG:  PELVIS_CENTER [OK] RIGHT_HIP [OK] RIGHT_KNEE [OK] RIGHT_ANKLE
    SPINE:      PELVIS_CENTER [OK] SPINE_MID [OK] SHOULDER_CENTER [OK] HEAD_CENTER
    LEFT ARM:   SHOULDER_CENTER [OK] LEFT_SHOULDER [OK] LEFT_ELBOW [OK] LEFT_WRIST
    RIGHT ARM:  SHOULDER_CENTER [OK] RIGHT_SHOULDER [OK] RIGHT_ELBOW [OK] RIGHT_WRIST
    HEAD:       HEAD_CENTER [OK] NOSE [OK] EYES/EARS
    
    Usage:
        skeleton = AnatomicalSkeleton(height=175)
        kinematics = SkeletonKinematics(skeleton)
        
        # Set hip flexion and propagate to knee/ankle
        kinematics.set_joint_angles(KP.LEFT_HIP, rx=30.0)  # 30[OK] flexion
        kinematics.propagate_fk(KP.LEFT_HIP)
        
        # Snap ankle to cluster Y-plane
        kinematics.snap_to_cluster_yplane(KP.LEFT_ANKLE, cluster_grid)
    """
    
    # =========================================================================
    # KINEMATIC CHAIN DEFINITIONS
    # =========================================================================
    
    # Parent [OK] Children mapping (defines the tree structure)
    KINEMATIC_TREE: Dict[int, List[int]] = {
        KP.PELVIS_CENTER: [KP.LEFT_HIP, KP.RIGHT_HIP, KP.SPINE_MID],
        KP.SPINE_MID: [KP.SHOULDER_CENTER],
        KP.SHOULDER_CENTER: [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.HEAD_CENTER],
        KP.HEAD_CENTER: [KP.NOSE],
        KP.NOSE: [KP.LEFT_EYE, KP.RIGHT_EYE],
        KP.LEFT_EYE: [KP.LEFT_EAR],
        KP.RIGHT_EYE: [KP.RIGHT_EAR],
        KP.LEFT_HIP: [KP.LEFT_KNEE],
        KP.LEFT_KNEE: [KP.LEFT_ANKLE],
        KP.RIGHT_HIP: [KP.RIGHT_KNEE],
        KP.RIGHT_KNEE: [KP.RIGHT_ANKLE],
        KP.LEFT_SHOULDER: [KP.LEFT_ELBOW],
        KP.LEFT_ELBOW: [KP.LEFT_WRIST],
        KP.RIGHT_SHOULDER: [KP.RIGHT_ELBOW],
        KP.RIGHT_ELBOW: [KP.RIGHT_WRIST],
    }
    
    # Child [OK] Parent mapping (for IK traversal)
    PARENT_MAP: Dict[int, int] = {}  # Built in __init__
    
    # Segment name mapping (keypoint pair [OK] segment name in create_segments())
    SEGMENT_MAP: Dict[Tuple[int, int], str] = {
        (KP.PELVIS_CENTER, KP.SPINE_MID): 'lower_spine',
        (KP.SPINE_MID, KP.SHOULDER_CENTER): 'upper_spine',
        (KP.SHOULDER_CENTER, KP.HEAD_CENTER): 'neck',
        (KP.LEFT_HIP, KP.LEFT_KNEE): 'thigh_l',
        (KP.LEFT_KNEE, KP.LEFT_ANKLE): 'shin_l',
        (KP.RIGHT_HIP, KP.RIGHT_KNEE): 'thigh_r',
        (KP.RIGHT_KNEE, KP.RIGHT_ANKLE): 'shin_r',
        (KP.LEFT_SHOULDER, KP.LEFT_ELBOW): 'upper_arm_l',
        (KP.LEFT_ELBOW, KP.LEFT_WRIST): 'forearm_l',
        (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW): 'upper_arm_r',
        (KP.RIGHT_ELBOW, KP.RIGHT_WRIST): 'forearm_r',
    }
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(self, skeleton: 'AnatomicalSkeleton'):
        """
        Initialize kinematics engine with reference to skeleton.
        
        Args:
            skeleton: AnatomicalSkeleton instance with keypoints and segments
        """
        self.skeleton = skeleton
        self.segments = skeleton.segments  # Dict[str, SegmentLCS]
        
        # Build parent map from kinematic tree
        self._build_parent_map()
        
        # Cache for rotation matrices (avoid recomputation)
        self._rotation_cache: Dict[int, np.ndarray] = {}
        
        # Track which joints have been modified (for partial updates)
        self._dirty_joints: Set[int] = set()
        
        logger.info("SkeletonKinematics initialized")
    
    def _build_parent_map(self):
        """Build child[OK][OK] ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢parent mapping from KINEMATIC_TREE."""
        SkeletonKinematics.PARENT_MAP = {}
        for parent, children in self.KINEMATIC_TREE.items():
            for child in children:
                SkeletonKinematics.PARENT_MAP[child] = parent
    
    # =========================================================================
    # FORWARD KINEMATICS - Set angles, propagate positions
    # =========================================================================
    
    def set_joint_angles(self, joint_kp: int, 
                         rx: Optional[float] = None,
                         ry: Optional[float] = None, 
                         rz: Optional[float] = None,
                         clamp: bool = True) -> bool:
        """
        Set rotation angles for a joint (with optional clamping to DoF limits).
        
        Args:
            joint_kp: Keypoint index of the joint (parent of the bone)
            rx: Flexion/Extension angle in degrees (around X-axis)
            ry: Abduction/Adduction angle in degrees (around Y-axis)
            rz: Internal/External rotation in degrees (around Z-axis)
            clamp: If True, clamp angles to anatomical limits
            
        Returns:
            True if angles were set successfully
        """
        # Find the segment that starts at this joint
        segment = self._get_segment_from_parent(joint_kp)
        if segment is None:
            logger.warning(f"No segment found starting at keypoint {joint_kp}")
            return False
        
        # Set angles (with clamping)
        if rx is not None:
            if clamp and segment.rx_limits:
                rx = np.clip(rx, segment.rx_limits[0], segment.rx_limits[1])
            segment.rx = rx if segment.rx_limits else 0
            
        if ry is not None:
            if clamp and segment.ry_limits:
                ry = np.clip(ry, segment.ry_limits[0], segment.ry_limits[1])
            segment.ry = ry if segment.ry_limits else 0
            
        if rz is not None:
            if clamp and segment.rz_limits:
                rz = np.clip(rz, segment.rz_limits[0], segment.rz_limits[1])
            segment.rz = rz if segment.rz_limits else 0
        
        # Mark joint as dirty
        self._dirty_joints.add(joint_kp)
        
        # Invalidate rotation cache for this and all children
        self._invalidate_cache_subtree(joint_kp)
        
        logger.debug(f"Set angles for {segment.name}: rx={segment.rx:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, "
                    f"ry={segment.ry:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, rz={segment.rz:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
        return True
    
    def get_joint_angles(self, joint_kp: int) -> Optional[Tuple[float, float, float]]:
        """
        Get current rotation angles for a joint.
        
        Returns:
            Tuple of (rx, ry, rz) in degrees, or None if no segment
        """
        segment = self._get_segment_from_parent(joint_kp)
        if segment is None:
            return None
        return (segment.rx, segment.ry, segment.rz)
    
    def propagate_fk(self, start_joint: int = None):
        """
        Propagate forward kinematics from a joint down to all descendants.
        
        This applies the rotation angles stored in SegmentLCS to compute
        new world positions for child keypoints.
        
        Args:
            start_joint: Starting joint keypoint index (default: PELVIS_CENTER)
        """
        if start_joint is None:
            start_joint = KP.PELVIS_CENTER
        
        # BFS traversal from start_joint
        queue = deque([start_joint])
        visited = set()
        
        while queue:
            current_kp = queue.popleft()
            if current_kp in visited:
                continue
            visited.add(current_kp)
            
            # Get children of current joint
            children = self.KINEMATIC_TREE.get(current_kp, [])
            
            for child_kp in children:
                # Compute new position for child based on parent + rotation
                self._apply_fk_to_child(current_kp, child_kp)
                queue.append(child_kp)
        
        # Clear dirty flags
        self._dirty_joints.clear()
        
        logger.debug(f"FK propagated from {KEYPOINT_NAMES[start_joint]}, "
                    f"updated {len(visited)} joints")
    
    def _apply_fk_to_child(self, parent_kp: int, child_kp: int):
        """
        Apply FK to compute child position from parent position + rotation.
        
        Uses the rotation angles stored in the segment connecting parent[OK][OK] ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢child.
        """
        # Get segment for this bone
        segment = self._get_segment_from_endpoints(parent_kp, child_kp)
        if segment is None:
            # No segment defined - use fixed offset (for head details etc.)
            return
        
        # Get parent position in ICCS
        parent_pos = self.skeleton.keypoints_iccs[parent_kp].copy()
        
        # Get bone length
        bone_length = segment.bone_length
        if bone_length <= 0:
            # Try to get from anthropometrics
            bone_length = self._get_bone_length(segment.name)
        
        # Build rotation matrix from Euler angles (XYZ order)
        # X = flexion/extension, Y = abduction, Z = internal rotation
        rot = Rotation.from_euler('xyz', [segment.rx, segment.ry, segment.rz], degrees=True)
        
        # Default bone direction (in rest pose, pointing down for legs, etc.)
        rest_direction = self._get_rest_direction(parent_kp, child_kp)
        
        # Rotate rest direction by joint angles
        rotated_direction = rot.apply(rest_direction)
        
        # Compute new child position
        new_child_pos = parent_pos + rotated_direction * bone_length
        
        # Update skeleton
        self.skeleton.keypoints_iccs[child_kp] = new_child_pos
        
        # Update world coordinates
        self.skeleton.keypoints_world[child_kp] = \
            self.skeleton.iccs.iccs_to_world(new_child_pos)
        
        # Cache rotation matrix
        self._rotation_cache[parent_kp] = rot.as_matrix()
    
    def _get_rest_direction(self, parent_kp: int, child_kp: int) -> np.ndarray:
        """
        Get the rest pose direction for a bone (before any rotation).
        
        Returns unit vector in ICCS coordinates.
        """
        # Define rest directions based on anatomy
        rest_directions = {
            # Legs point DOWN (negative Z) in rest pose
            (KP.LEFT_HIP, KP.LEFT_KNEE): np.array([0, 0, -1]),
            (KP.LEFT_KNEE, KP.LEFT_ANKLE): np.array([0, 0, -1]),
            (KP.RIGHT_HIP, KP.RIGHT_KNEE): np.array([0, 0, -1]),
            (KP.RIGHT_KNEE, KP.RIGHT_ANKLE): np.array([0, 0, -1]),
            
            # Spine points UP (positive Z)
            (KP.PELVIS_CENTER, KP.SPINE_MID): np.array([0, 0, 1]),
            (KP.SPINE_MID, KP.SHOULDER_CENTER): np.array([0, 0, 1]),
            (KP.SHOULDER_CENTER, KP.HEAD_CENTER): np.array([0, 0, 1]),
            
            # Arms point OUT (positive/negative X) and slightly DOWN
            (KP.LEFT_SHOULDER, KP.LEFT_ELBOW): np.array([-0.9, 0, -0.4]),
            (KP.LEFT_ELBOW, KP.LEFT_WRIST): np.array([-0.9, 0, -0.4]),
            (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW): np.array([0.9, 0, -0.4]),
            (KP.RIGHT_ELBOW, KP.RIGHT_WRIST): np.array([0.9, 0, -0.4]),
        }
        
        direction = rest_directions.get((parent_kp, child_kp))
        if direction is None:
            # Default: use current direction from skeleton
            current_vec = (self.skeleton.keypoints_iccs[child_kp] - 
                          self.skeleton.keypoints_iccs[parent_kp])
            length = np.linalg.norm(current_vec)
            if length > 0:
                return current_vec / length
            return np.array([0, 0, -1])  # Default down
        
        return direction / np.linalg.norm(direction)
    
    # =========================================================================
    # INVERSE KINEMATICS - Set target position, solve angles
    # =========================================================================
    
    # =========================================================================
    # INVERSE KINEMATICS - Multiple Algorithms
    # =========================================================================
    
    def solve_ik(self, end_effector_kp: int, 
                 target_pos: np.ndarray,
                 method: str = 'fabrik',
                 max_iterations: int = 50,
                 tolerance: float = 0.5,
                 constrain_angles: bool = True) -> bool:
        """
        Solve inverse kinematics to move end-effector to target position.
        
        Args:
            end_effector_kp: Keypoint index of the end effector (e.g., LEFT_ANKLE)
            target_pos: Target position in ICCS coordinates
            method: IK algorithm - 'fabrik', 'ccd', 'jacobian', or 'analytical'
            max_iterations: Maximum IK iterations
            tolerance: Distance tolerance in cm
            constrain_angles: If True, apply joint angle limits
            
        Returns:
            True if IK converged within tolerance
        """
        if method == 'fabrik':
            return self._solve_ik_fabrik(end_effector_kp, target_pos, 
                                         max_iterations, tolerance, constrain_angles)
        elif method == 'ccd':
            return self._solve_ik_ccd(end_effector_kp, target_pos,
                                      max_iterations, tolerance, constrain_angles)
        elif method == 'jacobian':
            return self._solve_ik_jacobian(end_effector_kp, target_pos,
                                           max_iterations, tolerance, constrain_angles)
        elif method == 'analytical':
            return self._solve_ik_analytical(end_effector_kp, target_pos, constrain_angles)
        else:
            logger.error(f"Unknown IK method: {method}")
            return False
    
    # =========================================================================
    # FABRIK - Forward And Backward Reaching Inverse Kinematics
    # =========================================================================
    
    def _solve_ik_fabrik(self, end_effector_kp: int,
                         target_pos: np.ndarray,
                         max_iterations: int = 50,
                         tolerance: float = 0.5,
                         constrain_angles: bool = True) -> bool:
        """
        FABRIK Algorithm - Fast, robust, handles joint limits well.
        
        Paper: "FABRIK: A fast, iterative solver for the Inverse Kinematics problem"
               by Aristidou & Lasenby (2011)
        
        Algorithm:
        1. BACKWARD PASS: Move chain from end-effector toward target
        2. FORWARD PASS: Re-anchor chain to root, maintaining bone lengths
        3. Repeat until convergence
        
        Advantages over CCD:
        - Faster convergence (typically 2-10 iterations)
        - More natural-looking poses
        - Better handling of joint limits
        - No gimbal lock issues
        """
        # Build chain from root to end-effector (reversed from _get_chain_to_root)
        chain = self._get_chain_to_root(end_effector_kp)
        chain.reverse()  # Now: [root, ..., end_effector]
        
        if len(chain) < 2:
            logger.warning(f"Chain too short for FABRIK: {chain}")
            return False
        
        # Get bone lengths for each segment
        bone_lengths = []
        for i in range(len(chain) - 1):
            parent_kp = chain[i]
            child_kp = chain[i + 1]
            segment = self._get_segment_from_endpoints(parent_kp, child_kp)
            if segment and segment.bone_length > 0:
                bone_lengths.append(segment.bone_length)
            else:
                # Calculate from current positions
                length = np.linalg.norm(
                    self.skeleton.keypoints_iccs[child_kp] - 
                    self.skeleton.keypoints_iccs[parent_kp]
                )
                bone_lengths.append(max(length, 1.0))  # Minimum 1cm
        
        # Check if target is reachable
        total_length = sum(bone_lengths)
        root_pos = self.skeleton.keypoints_iccs[chain[0]].copy()
        dist_to_target = np.linalg.norm(target_pos - root_pos)
        
        if dist_to_target > total_length:
            logger.warning(f"Target unreachable: distance={dist_to_target:.1f}cm, "
                          f"max_reach={total_length:.1f}cm")
            # Stretch toward target as much as possible
            target_pos = root_pos + (target_pos - root_pos) / dist_to_target * total_length * 0.99
        
        # Extract current joint positions
        positions = [self.skeleton.keypoints_iccs[kp].copy() for kp in chain]
        
        # FABRIK iterations
        for iteration in range(max_iterations):
            # Check convergence
            error = np.linalg.norm(positions[-1] - target_pos)
            if error < tolerance:
                # Apply final positions to skeleton
                self._apply_fabrik_positions(chain, positions, bone_lengths, constrain_angles)
                logger.info(f"FABRIK converged in {iteration+1} iterations, error={error:.2f}cm")
                return True
            
            # Store original root position (anchor point)
            root_anchor = positions[0].copy()
            
            # =====================================================
            # BACKWARD PASS: End-effector to root
            # =====================================================
            positions[-1] = target_pos.copy()
            
            for i in range(len(chain) - 2, -1, -1):
                # Direction from current to next (toward end-effector)
                direction = positions[i] - positions[i + 1]
                dir_length = np.linalg.norm(direction)
                
                if dir_length < 0.001:
                    direction = np.array([0, 0, 1])  # Default up
                else:
                    direction = direction / dir_length
                
                # Place joint at bone_length distance from next joint
                positions[i] = positions[i + 1] + direction * bone_lengths[i]
            
            # =====================================================
            # FORWARD PASS: Root to end-effector
            # =====================================================
            positions[0] = root_anchor  # Re-anchor root
            
            for i in range(len(chain) - 1):
                # Direction from current to next
                direction = positions[i + 1] - positions[i]
                dir_length = np.linalg.norm(direction)
                
                if dir_length < 0.001:
                    direction = np.array([0, 0, -1])  # Default down
                else:
                    direction = direction / dir_length
                
                # Place next joint at bone_length distance
                positions[i + 1] = positions[i] + direction * bone_lengths[i]
                
                # Apply joint angle constraints if enabled
                if constrain_angles:
                    positions[i + 1] = self._apply_fabrik_constraint(
                        chain[i], chain[i + 1], 
                        positions[i], positions[i + 1],
                        bone_lengths[i]
                    )
        
        # Did not converge - apply best result anyway
        self._apply_fabrik_positions(chain, positions, bone_lengths, constrain_angles)
        final_error = np.linalg.norm(positions[-1] - target_pos)
        logger.warning(f"FABRIK did not fully converge after {max_iterations} iterations, "
                      f"error={final_error:.2f}cm")
        return final_error < tolerance * 2  # Accept if close
    
    def _apply_fabrik_constraint(self, parent_kp: int, child_kp: int,
                                  parent_pos: np.ndarray, child_pos: np.ndarray,
                                  bone_length: float) -> np.ndarray:
        """
        Apply joint angle constraints during FABRIK.
        
        Constrains the child position to lie within the valid cone defined
        by the joint's rotation limits.
        """
        segment = self._get_segment_from_endpoints(parent_kp, child_kp)
        if segment is None:
            return child_pos
        
        # Current direction
        direction = child_pos - parent_pos
        dir_length = np.linalg.norm(direction)
        if dir_length < 0.001:
            return child_pos
        direction = direction / dir_length
        
        # Get rest direction for this bone
        rest_dir = self._get_rest_direction(parent_kp, child_kp)
        
        # Calculate angle from rest pose
        dot = np.clip(np.dot(direction, rest_dir), -1, 1)
        angle = np.degrees(np.arccos(dot))
        
        # Get max allowed angle from joint limits
        max_angle = self._get_max_joint_angle(segment)
        
        if angle > max_angle:
            # Constrain to max angle
            # Find rotation axis
            axis = np.cross(rest_dir, direction)
            axis_len = np.linalg.norm(axis)
            
            if axis_len > 0.001:
                axis = axis / axis_len
                
                # Rotate rest direction by max_angle
                rot = Rotation.from_rotvec(np.radians(max_angle) * axis)
                constrained_dir = rot.apply(rest_dir)
                
                return parent_pos + constrained_dir * bone_length
        
        return child_pos
    
    def _get_max_joint_angle(self, segment: 'SegmentLCS') -> float:
        """Get maximum rotation angle allowed for a joint."""
        max_angles = []
        
        if segment.rx_limits:
            max_angles.append(max(abs(segment.rx_limits[0]), abs(segment.rx_limits[1])))
        if segment.ry_limits:
            max_angles.append(max(abs(segment.ry_limits[0]), abs(segment.ry_limits[1])))
        if segment.rz_limits:
            max_angles.append(max(abs(segment.rz_limits[0]), abs(segment.rz_limits[1])))
        
        if max_angles:
            # Use the largest allowed angle as the cone radius
            return max(max_angles)
        return 180.0  # No limits
    
    def _apply_fabrik_positions(self, chain: List[int], positions: List[np.ndarray],
                                 bone_lengths: List[float], constrain_angles: bool):
        """Apply FABRIK-computed positions to skeleton and update joint angles."""
        # Update positions
        for i, kp in enumerate(chain):
            self.skeleton.keypoints_iccs[kp] = positions[i].copy()
            self.skeleton.keypoints_world[kp] = self.skeleton.iccs.iccs_to_world(positions[i])
        
        # Back-calculate joint angles from positions
        for i in range(len(chain) - 1):
            parent_kp = chain[i]
            child_kp = chain[i + 1]
            
            segment = self._get_segment_from_endpoints(parent_kp, child_kp)
            if segment is None:
                continue
            
            # Current direction
            current_dir = positions[i + 1] - positions[i]
            current_dir = current_dir / np.linalg.norm(current_dir)
            
            # Rest direction
            rest_dir = self._get_rest_direction(parent_kp, child_kp)
            
            # Calculate rotation from rest to current
            rotation = self._rotation_between_vectors(rest_dir, current_dir)
            
            # Convert to Euler angles
            euler = rotation.as_euler('xyz', degrees=True)
            
            # Apply to segment (with clamping if constrain_angles)
            segment.set_angles(euler[0], euler[1], euler[2])
    
    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> Rotation:
        """Calculate rotation that transforms v1 to v2."""
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        if np.linalg.norm(cross) < 0.0001:
            if dot > 0:
                return Rotation.identity()
            else:
                # 180 degree rotation - find perpendicular axis
                perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
                axis = np.cross(v1, perp)
                axis = axis / np.linalg.norm(axis)
                return Rotation.from_rotvec(np.pi * axis)
        
        # Rodrigues' rotation formula
        angle = np.arccos(np.clip(dot, -1, 1))
        axis = cross / np.linalg.norm(cross)
        return Rotation.from_rotvec(angle * axis)
    
    # =========================================================================
    # CCD - Cyclic Coordinate Descent
    # =========================================================================
    
    def _solve_ik_ccd(self, end_effector_kp: int,
                      target_pos: np.ndarray,
                      max_iterations: int = 50,
                      tolerance: float = 0.5,
                      constrain_angles: bool = True) -> bool:
        """
        CCD Algorithm - Classic iterative IK solver.
        
        Iteratively rotates each joint to minimize end-effector distance to target.
        Good for single chains but can produce unnatural poses.
        """
        chain = self._get_chain_to_root(end_effector_kp)
        if len(chain) < 2:
            logger.warning(f"Chain too short for CCD: {chain}")
            return False
        
        for iteration in range(max_iterations):
            # Check convergence
            current_pos = self.skeleton.keypoints_iccs[end_effector_kp]
            error = np.linalg.norm(target_pos - current_pos)
            
            if error < tolerance:
                logger.info(f"CCD converged in {iteration+1} iterations, error={error:.2f}cm")
                return True
            
            # CCD: iterate through chain from end-effector to root
            for i in range(len(chain) - 1):
                joint_kp = chain[i + 1]
                
                ee_pos = self.skeleton.keypoints_iccs[end_effector_kp]
                joint_pos = self.skeleton.keypoints_iccs[joint_kp]
                
                to_current = ee_pos - joint_pos
                to_target = target_pos - joint_pos
                
                to_current_len = np.linalg.norm(to_current)
                to_target_len = np.linalg.norm(to_target)
                
                if to_current_len < 0.001 or to_target_len < 0.001:
                    continue
                
                to_current = to_current / to_current_len
                to_target = to_target / to_target_len
                
                # Calculate rotation
                rotation = self._rotation_between_vectors(to_current, to_target)
                euler_delta = rotation.as_euler('xyz', degrees=True)
                
                # Apply delta to current angles
                current_angles = self.get_joint_angles(joint_kp)
                if current_angles:
                    new_rx = current_angles[0] + euler_delta[0] * 0.5  # Damping
                    new_ry = current_angles[1] + euler_delta[1] * 0.5
                    new_rz = current_angles[2] + euler_delta[2] * 0.5
                    
                    self.set_joint_angles(joint_kp, rx=new_rx, ry=new_ry, rz=new_rz,
                                         clamp=constrain_angles)
                
                self.propagate_fk(joint_kp)
        
        final_error = np.linalg.norm(target_pos - self.skeleton.keypoints_iccs[end_effector_kp])
        logger.warning(f"CCD did not converge after {max_iterations} iterations, "
                      f"error={final_error:.2f}cm")
        return False
    
    # =========================================================================
    # JACOBIAN - Gradient-based IK
    # =========================================================================
    
    def _solve_ik_jacobian(self, end_effector_kp: int,
                           target_pos: np.ndarray,
                           max_iterations: int = 100,
                           tolerance: float = 0.5,
                           constrain_angles: bool = True) -> bool:
        """
        Jacobian Transpose IK - Gradient-based approach.
        
        Uses the Jacobian matrix to compute how joint angles affect end-effector.
        More physically accurate but slower than FABRIK/CCD.
        """
        chain = self._get_chain_to_root(end_effector_kp)
        chain.reverse()  # Root to end-effector
        
        if len(chain) < 2:
            return False
        
        # Learning rate (step size)
        alpha = 0.1
        
        for iteration in range(max_iterations):
            # Current end-effector position
            ee_pos = self.skeleton.keypoints_iccs[end_effector_kp]
            error_vec = target_pos - ee_pos
            error = np.linalg.norm(error_vec)
            
            if error < tolerance:
                logger.info(f"Jacobian IK converged in {iteration+1} iterations, error={error:.2f}cm")
                return True
            
            # Build Jacobian matrix (3 x num_dof)
            # For simplicity, we use numerical differentiation
            jacobian = self._compute_jacobian(chain, end_effector_kp)
            
            if jacobian is None:
                continue
            
            # Jacobian Transpose method: delta_theta = alpha * J^T * error
            delta_angles = alpha * jacobian.T @ error_vec
            
            # Apply angle changes
            angle_idx = 0
            for i in range(len(chain) - 1):
                joint_kp = chain[i]
                segment = self._get_segment_from_parent(joint_kp)
                if segment is None:
                    continue
                
                current_angles = list(self.get_joint_angles(joint_kp) or (0, 0, 0))
                
                if segment.rx_limits:
                    current_angles[0] += delta_angles[angle_idx]
                    angle_idx += 1
                if segment.ry_limits:
                    current_angles[1] += delta_angles[angle_idx]
                    angle_idx += 1
                if segment.rz_limits:
                    current_angles[2] += delta_angles[angle_idx]
                    angle_idx += 1
                
                self.set_joint_angles(joint_kp, rx=current_angles[0], 
                                     ry=current_angles[1], rz=current_angles[2],
                                     clamp=constrain_angles)
            
            self.propagate_fk(chain[0])
        
        final_error = np.linalg.norm(target_pos - self.skeleton.keypoints_iccs[end_effector_kp])
        logger.warning(f"Jacobian IK did not converge, error={final_error:.2f}cm")
        return False
    
    def _compute_jacobian(self, chain: List[int], end_effector_kp: int) -> Optional[np.ndarray]:
        """Compute Jacobian matrix numerically."""
        delta = 0.1  # Small angle change for numerical differentiation
        
        # Count total DoF
        num_dof = 0
        for i in range(len(chain) - 1):
            segment = self._get_segment_from_parent(chain[i])
            if segment:
                num_dof += segment.get_dof()
        
        if num_dof == 0:
            return None
        
        jacobian = np.zeros((3, num_dof))
        
        # Current end-effector position
        ee_pos = self.skeleton.keypoints_iccs[end_effector_kp].copy()
        
        # Numerical differentiation for each DoF
        col_idx = 0
        for i in range(len(chain) - 1):
            joint_kp = chain[i]
            segment = self._get_segment_from_parent(joint_kp)
            if segment is None:
                continue
            
            original_angles = (segment.rx, segment.ry, segment.rz)
            
            for axis_idx, (limits, angle) in enumerate([
                (segment.rx_limits, segment.rx),
                (segment.ry_limits, segment.ry),
                (segment.rz_limits, segment.rz)
            ]):
                if limits is None:
                    continue
                
                # Perturb angle
                angles = list(original_angles)
                angles[axis_idx] += delta
                segment.set_angles(*angles)
                self.propagate_fk(joint_kp)
                
                # Measure change in end-effector
                new_ee_pos = self.skeleton.keypoints_iccs[end_effector_kp]
                jacobian[:, col_idx] = (new_ee_pos - ee_pos) / delta
                col_idx += 1
                
                # Restore original
                segment.set_angles(*original_angles)
        
        # Restore original pose
        self.propagate_fk(chain[0])
        
        return jacobian
    
    # =========================================================================
    # ANALYTICAL IK - Closed-form solution for specific chains
    # =========================================================================
    
    def _solve_ik_analytical(self, end_effector_kp: int,
                             target_pos: np.ndarray,
                             constrain_angles: bool = True) -> bool:
        """
        Analytical IK for 2-bone chains (arm or leg).
        
        Uses law of cosines for exact solution - fastest method for simple chains.
        Only works for: shoulder[OK][OK] ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢elbow[OK][OK] ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢wrist or hip[OK][OK] ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢knee[OK][OK] ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ankle
        """
        # Determine which chain this is
        if end_effector_kp == KP.LEFT_WRIST:
            root_kp, mid_kp, end_kp = KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST
            upper_segment, lower_segment = 'upper_arm_l', 'forearm_l'
        elif end_effector_kp == KP.RIGHT_WRIST:
            root_kp, mid_kp, end_kp = KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST
            upper_segment, lower_segment = 'upper_arm_r', 'forearm_r'
        elif end_effector_kp == KP.LEFT_ANKLE:
            root_kp, mid_kp, end_kp = KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE
            upper_segment, lower_segment = 'thigh_l', 'shin_l'
        elif end_effector_kp == KP.RIGHT_ANKLE:
            root_kp, mid_kp, end_kp = KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE
            upper_segment, lower_segment = 'thigh_r', 'shin_r'
        else:
            logger.warning(f"Analytical IK not supported for keypoint {end_effector_kp}")
            return self._solve_ik_fabrik(end_effector_kp, target_pos, 
                                         constrain_angles=constrain_angles)
        
        # Get bone lengths
        L1 = self._get_bone_length(upper_segment)
        L2 = self._get_bone_length(lower_segment)
        
        # Root position (fixed)
        root_pos = self.skeleton.keypoints_iccs[root_kp].copy()
        
        # Vector from root to target
        D = target_pos - root_pos
        dist = np.linalg.norm(D)
        
        # Check reachability
        if dist > L1 + L2:
            # Stretch toward target
            D_unit = D / dist
            target_pos = root_pos + D_unit * (L1 + L2 - 0.1)
            D = target_pos - root_pos
            dist = np.linalg.norm(D)
            logger.debug(f"Analytical IK: target adjusted (unreachable)")
        
        if dist < abs(L1 - L2):
            # Target too close
            logger.warning(f"Analytical IK: target too close to root")
            return False
        
        # Law of cosines to find elbow/knee angle
        # cos(angle_at_mid) = (L1[OK] + L2[OK] - DÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) / (2 * L1 * L2)
        cos_mid = (L1*L1 + L2*L2 - dist*dist) / (2 * L1 * L2)
        cos_mid = np.clip(cos_mid, -1, 1)
        angle_mid = np.arccos(cos_mid)  # Angle at middle joint
        
        # Angle at root joint
        # cos(angle_at_root) = (L1[OK] + D[OK] - L2ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²) / (2 * L1 * D)
        cos_root = (L1*L1 + dist*dist - L2*L2) / (2 * L1 * dist)
        cos_root = np.clip(cos_root, -1, 1)
        angle_root = np.arccos(cos_root)
        
        # Direction from root to target
        D_unit = D / dist
        
        # Find plane normal (for rotation axis)
        # Use world up as reference
        world_up = np.array([0, 0, 1])
        
        # Calculate positions
        # Upper bone: rotate from rest direction by angle_root toward target
        rest_dir = self._get_rest_direction(root_kp, mid_kp)
        
        # Rotation axis (perpendicular to plane containing rest_dir and D_unit)
        axis = np.cross(rest_dir, D_unit)
        axis_len = np.linalg.norm(axis)
        
        if axis_len > 0.001:
            axis = axis / axis_len
            
            # Rotate rest direction
            rot = Rotation.from_rotvec(angle_root * axis)
            upper_dir = rot.apply(rest_dir)
        else:
            upper_dir = D_unit
        
        # Middle joint position
        mid_pos = root_pos + upper_dir * L1
        
        # Lower bone direction (toward target)
        lower_dir = (target_pos - mid_pos)
        lower_dir = lower_dir / np.linalg.norm(lower_dir)
        
        # End effector position
        end_pos = mid_pos + lower_dir * L2
        
        # Update skeleton
        self.skeleton.keypoints_iccs[mid_kp] = mid_pos
        self.skeleton.keypoints_iccs[end_kp] = end_pos
        self.skeleton.keypoints_world[mid_kp] = self.skeleton.iccs.iccs_to_world(mid_pos)
        self.skeleton.keypoints_world[end_kp] = self.skeleton.iccs.iccs_to_world(end_pos)
        
        # Back-calculate joint angles
        self._apply_fabrik_positions([root_kp, mid_kp, end_kp], 
                                     [root_pos, mid_pos, end_pos],
                                     [L1, L2], constrain_angles)
        
        error = np.linalg.norm(end_pos - target_pos)
        logger.info(f"Analytical IK solved, error={error:.2f}cm")
        return error < 15.5
    
    # =========================================================================
    # MULTI-TARGET IK - Solve for multiple end-effectors simultaneously
    # =========================================================================
    
    def solve_multi_target_ik(self, targets: Dict[int, np.ndarray],
                              method: str = 'fabrik',
                              max_iterations: int = 100,
                              tolerance: float = 1.0) -> Dict[int, bool]:
        """
        Solve IK for multiple end-effectors simultaneously.
        
        Useful for constraining both feet to ground, or matching full pose.
        
        Args:
            targets: Dict of {keypoint_index: target_position}
            method: IK method to use
            max_iterations: Maximum iterations
            tolerance: Distance tolerance per target
            
        Returns:
            Dict of {keypoint_index: converged_bool}
        """
        results = {}
        
        # Sort targets by chain independence
        # (e.g., solve arms before spine to avoid interference)
        sorted_targets = sorted(targets.items(), 
                               key=lambda x: self._get_chain_priority(x[0]))
        
        for iteration in range(max_iterations):
            all_converged = True
            
            for kp, target in sorted_targets:
                current_pos = self.skeleton.keypoints_iccs[kp]
                error = np.linalg.norm(target - current_pos)
                
                if error > tolerance:
                    all_converged = False
                    # Single iteration of chosen method
                    if method == 'fabrik':
                        self._fabrik_single_iteration(kp, target)
                    else:
                        self._ccd_single_iteration(kp, target)
                
                results[kp] = error <= tolerance
            
            if all_converged:
                logger.info(f"Multi-target IK converged in {iteration+1} iterations")
                return results
        
        logger.warning(f"Multi-target IK: some targets did not converge")
        return results
    
    def _fabrik_single_iteration(self, end_effector_kp: int, target_pos: np.ndarray):
        """Single FABRIK iteration for multi-target solving."""
        chain = self._get_chain_to_root(end_effector_kp)
        chain.reverse()
        
        if len(chain) < 2:
            return
        
        # Get bone lengths and positions
        bone_lengths = []
        positions = []
        
        for i, kp in enumerate(chain):
            positions.append(self.skeleton.keypoints_iccs[kp].copy())
            if i < len(chain) - 1:
                next_kp = chain[i + 1]
                length = np.linalg.norm(
                    self.skeleton.keypoints_iccs[next_kp] - 
                    self.skeleton.keypoints_iccs[kp]
                )
                bone_lengths.append(max(length, 1.0))
        
        root_anchor = positions[0].copy()
        
        # Backward pass
        positions[-1] = target_pos.copy()
        for i in range(len(chain) - 2, -1, -1):
            direction = positions[i] - positions[i + 1]
            dir_len = np.linalg.norm(direction)
            if dir_len > 0.001:
                direction = direction / dir_len
            else:
                direction = np.array([0, 0, 1])
            positions[i] = positions[i + 1] + direction * bone_lengths[i]
        
        # Forward pass
        positions[0] = root_anchor
        for i in range(len(chain) - 1):
            direction = positions[i + 1] - positions[i]
            dir_len = np.linalg.norm(direction)
            if dir_len > 0.001:
                direction = direction / dir_len
            else:
                direction = np.array([0, 0, -1])
            positions[i + 1] = positions[i] + direction * bone_lengths[i]
        
        # Apply positions
        for i, kp in enumerate(chain):
            self.skeleton.keypoints_iccs[kp] = positions[i]
            self.skeleton.keypoints_world[kp] = self.skeleton.iccs.iccs_to_world(positions[i])
    
    def _ccd_single_iteration(self, end_effector_kp: int, target_pos: np.ndarray):
        """Single CCD iteration for multi-target solving."""
        chain = self._get_chain_to_root(end_effector_kp)
        
        for i in range(len(chain) - 1):
            joint_kp = chain[i + 1]
            
            ee_pos = self.skeleton.keypoints_iccs[end_effector_kp]
            joint_pos = self.skeleton.keypoints_iccs[joint_kp]
            
            to_current = ee_pos - joint_pos
            to_target = target_pos - joint_pos
            
            if np.linalg.norm(to_current) < 0.001 or np.linalg.norm(to_target) < 0.001:
                continue
            
            to_current = to_current / np.linalg.norm(to_current)
            to_target = to_target / np.linalg.norm(to_target)
            
            rotation = self._rotation_between_vectors(to_current, to_target)
            euler_delta = rotation.as_euler('xyz', degrees=True) * 0.3  # Damping
            
            current_angles = self.get_joint_angles(joint_kp)
            if current_angles:
                self.set_joint_angles(joint_kp,
                                     rx=current_angles[0] + euler_delta[0],
                                     ry=current_angles[1] + euler_delta[1],
                                     rz=current_angles[2] + euler_delta[2])
            
            self.propagate_fk(joint_kp)
    
    def _get_chain_priority(self, kp: int) -> int:
        """Get solving priority for a keypoint (lower = solve first)."""
        # Extremities first, then joints, then spine
        if kp in [KP.LEFT_WRIST, KP.RIGHT_WRIST, KP.LEFT_ANKLE, KP.RIGHT_ANKLE]:
            return 0
        elif kp in [KP.LEFT_ELBOW, KP.RIGHT_ELBOW, KP.LEFT_KNEE, KP.RIGHT_KNEE]:
            return 1
        elif kp in [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, KP.LEFT_HIP, KP.RIGHT_HIP]:
            return 2
        else:
            return 3
    
    def _get_chain_to_root(self, end_kp: int) -> List[int]:
        """Get kinematic chain from end-effector to root (PELVIS_CENTER)."""
        chain = [end_kp]
        current = end_kp
        
        while current in self.PARENT_MAP:
            parent = self.PARENT_MAP[current]
            chain.append(parent)
            current = parent
            
            if current == KP.PELVIS_CENTER:
                break
        
        return chain
    
    # =========================================================================
    # Y-PLANE SNAPPING - Align keypoints to cluster voxels
    # =========================================================================
    
    def snap_to_cluster_yplane(self, keypoint_kp: int, 
                               cluster_grid: 'EnhancedOccupancyGrid',
                               snap_threshold: float = 4.0) -> bool:
        """
        Snap a keypoint to the nearest populated Y-plane in the cluster.
        
        This adjusts the keypoint's X and Z to match a populated voxel
        while keeping it in the correct Y-plane (depth layer).
        
        Args:
            keypoint_kp: Keypoint index to snap
            cluster_grid: EnhancedOccupancyGrid with populated voxels
            snap_threshold: Maximum distance in cm to snap
            
        Returns:
            True if snapped successfully
        """
        if cluster_grid is None:
            return False
        
        # Get keypoint position in ICCS
        kp_pos = self.skeleton.keypoints_iccs[keypoint_kp].copy()
        
        # Convert to world coordinates for grid lookup
        world_pos = self.skeleton.iccs.iccs_to_world(kp_pos)
        
        # Find Y-plane index
        y_value = world_pos[1]  # Y is depth in world coords
        resolution = cluster_grid.resolution
        min_bound = cluster_grid.min_bound if hasattr(cluster_grid, 'min_bound') else np.array([0, 0, 0])
        
        y_plane_idx = int((y_value - min_bound[1]) / resolution)
        
        # Get points in this Y-plane (and adjacent planes for robustness)
        plane_points = []
        for dy in [-1, 0, 1]:  # Check adjacent Y-planes too
            check_y_idx = y_plane_idx + dy
            
            # Iterate through occupied cells
            for cell_idx, cell_points in cluster_grid.cell_points.items():
                if len(cell_idx) >= 2 and cell_idx[1] == check_y_idx:
                    plane_points.extend(cell_points)
        
        if not plane_points:
            logger.debug(f"No points in Y-plane {y_plane_idx} for snapping")
            return False
        
        plane_points = np.array(plane_points)
        
        # Find nearest point in XZ plane (ignoring Y)
        xz_current = world_pos[[0, 2]]  # X and Z
        xz_plane = plane_points[:, [0, 2]]
        
        distances = np.linalg.norm(xz_plane - xz_current, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist > snap_threshold:
            logger.debug(f"Nearest point too far ({min_dist:.1f}cm > {snap_threshold}cm)")
            return False
        
        # Snap X and Z to nearest point
        nearest_point = plane_points[min_idx]
        snapped_world = np.array([nearest_point[0], world_pos[1], nearest_point[2]])
        
        # Convert back to ICCS
        snapped_iccs = self.skeleton.iccs.world_to_iccs(snapped_world)
        
        # Update keypoint
        self.skeleton.keypoints_iccs[keypoint_kp] = snapped_iccs
        self.skeleton.keypoints_world[keypoint_kp] = snapped_world
        
        logger.info(f"Snapped {KEYPOINT_NAMES[keypoint_kp]} to Y-plane {y_plane_idx}, "
                   f"distance={min_dist:.1f}cm")
        return True
    
    def snap_chain_to_cluster(self, chain_root: int, 
                              cluster_grid: 'EnhancedOccupancyGrid',
                              end_effector_only: bool = True):
        """
        Snap a kinematic chain to cluster Y-planes.
        
        Args:
            chain_root: Root keypoint of the chain (e.g., LEFT_HIP for leg)
            cluster_grid: EnhancedOccupancyGrid with populated voxels
            end_effector_only: If True, only snap the end effector
        """
        # Get all descendants
        descendants = self._get_all_descendants(chain_root)
        
        if end_effector_only:
            # Find leaf nodes (no children)
            leaves = [kp for kp in descendants 
                     if kp not in self.KINEMATIC_TREE or not self.KINEMATIC_TREE[kp]]
            targets = leaves
        else:
            targets = descendants
        
        # Snap each target
        for kp in targets:
            self.snap_to_cluster_yplane(kp, cluster_grid)
    
    # =========================================================================
    # CONVENIENCE METHODS FOR COMMON OPERATIONS
    # =========================================================================
    
    def flex_leg(self, side: str, hip_flex: float, knee_flex: float = None,
                 cluster_grid: 'EnhancedOccupancyGrid' = None):
        """
        Flex a leg forward (positive) or back (negative).
        
        Args:
            side: 'left' or 'right'
            hip_flex: Hip flexion angle in degrees
            knee_flex: Knee flexion angle (default: auto-computed for natural bend)
            cluster_grid: Optional grid for Y-plane snapping
        """
        if side == 'left':
            hip_kp, knee_kp, ankle_kp = KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE
        else:
            hip_kp, knee_kp, ankle_kp = KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE
        
        # Set hip flexion
        self.set_joint_angles(hip_kp, rx=hip_flex)
        
        # Set knee flexion (default: partial follow of hip for natural walk)
        if knee_flex is None:
            knee_flex = max(0, hip_flex * 0.3)  # Knee bends ~30% of hip
        self.set_joint_angles(knee_kp, rx=knee_flex)
        
        # Propagate FK from hip
        self.propagate_fk(hip_kp)
        
        # Optional: snap ankle to cluster
        if cluster_grid:
            self.snap_to_cluster_yplane(ankle_kp, cluster_grid)
        
        logger.info(f"Flexed {side} leg: hip={hip_flex:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, knee={knee_flex:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
    
    def raise_arm(self, side: str, shoulder_flex: float, shoulder_abduct: float = 0,
                  elbow_flex: float = 0, cluster_grid: 'EnhancedOccupancyGrid' = None):
        """
        Raise an arm.
        
        Args:
            side: 'left' or 'right'
            shoulder_flex: Shoulder flexion (forward raise) in degrees
            shoulder_abduct: Shoulder abduction (sideways raise) in degrees
            elbow_flex: Elbow flexion in degrees
            cluster_grid: Optional grid for Y-plane snapping
        """
        if side == 'left':
            shoulder_kp, elbow_kp, wrist_kp = KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST
        else:
            shoulder_kp, elbow_kp, wrist_kp = KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST
        
        self.set_joint_angles(shoulder_kp, rx=shoulder_flex, ry=shoulder_abduct)
        self.set_joint_angles(elbow_kp, rx=elbow_flex)
        
        self.propagate_fk(shoulder_kp)
        
        if cluster_grid:
            self.snap_to_cluster_yplane(wrist_kp, cluster_grid)
        
        logger.info(f"Raised {side} arm: shoulder_flex={shoulder_flex:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, "
                   f"abduct={shoulder_abduct:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, elbow={elbow_flex:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
    
    def bend_spine(self, lower_flex: float = 0, upper_flex: float = 0,
                   side_bend: float = 0, twist: float = 0):
        """
        Bend the spine.
        
        Args:
            lower_flex: Lower spine forward bend (positive) or arch (negative)
            upper_flex: Upper spine forward bend
            side_bend: Side bend angle (positive = right)
            twist: Torso twist angle (positive = right)
        """
        self.set_joint_angles(KP.PELVIS_CENTER, rx=lower_flex, ry=side_bend, rz=twist)
        self.set_joint_angles(KP.SPINE_MID, rx=upper_flex, ry=side_bend*0.5, rz=twist*0.5)
        
        self.propagate_fk(KP.PELVIS_CENTER)
        
        logger.info(f"Bent spine: lower={lower_flex:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, upper={upper_flex:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, "
                   f"side={side_bend:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°, twist={twist:.1f}ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_segment_from_parent(self, parent_kp: int) -> Optional['SegmentLCS']:
        """Get segment that starts at the given parent keypoint."""
        for (p, c), name in self.SEGMENT_MAP.items():
            if p == parent_kp:
                return self.segments.get(name)
        return None
    
    def _get_segment_from_endpoints(self, parent_kp: int, child_kp: int) -> Optional['SegmentLCS']:
        """Get segment connecting parent and child keypoints."""
        name = self.SEGMENT_MAP.get((parent_kp, child_kp))
        if name:
            return self.segments.get(name)
        return None
    
    def _get_bone_length(self, segment_name: str) -> float:
        """Get bone length from anthropometrics."""
        # Map segment names to anthropometric keys
        length_map = {
            'thigh_l': 'thigh_l', 'thigh_r': 'thigh_r',
            'shin_l': 'shin_l', 'shin_r': 'shin_r',
            'upper_arm_l': 'upper_arm_l', 'upper_arm_r': 'upper_arm_r',
            'forearm_l': 'forearm_l', 'forearm_r': 'forearm_r',
            'lower_spine': 'lower_spine', 'upper_spine': 'upper_spine',
            'neck': 'neck',
        }
        
        key = length_map.get(segment_name)
        if key and hasattr(self.skeleton, 'bone_lengths'):
            return self.skeleton.bone_lengths.get(key, 20.0)  # Default 20cm
        return 20.0
    
    def _invalidate_cache_subtree(self, root_kp: int):
        """Invalidate rotation cache for a joint and all descendants."""
        self._rotation_cache.pop(root_kp, None)
        for child in self.KINEMATIC_TREE.get(root_kp, []):
            self._invalidate_cache_subtree(child)
    
    def _get_all_descendants(self, root_kp: int) -> List[int]:
        """Get all descendant keypoints of a root joint."""
        descendants = []
        queue = deque([root_kp])
        
        while queue:
            current = queue.popleft()
            descendants.append(current)
            queue.extend(self.KINEMATIC_TREE.get(current, []))
        
        return descendants
    
    def reset_to_rest_pose(self):
        """Reset all joint angles to zero (rest/T-pose)."""
        for segment in self.segments.values():
            segment.rx = 0
            segment.ry = 0
            segment.rz = 0
        
        self._dirty_joints.clear()
        self._rotation_cache.clear()
        
        # Reinitialize keypoints to rest pose
        self.skeleton._init_rest_pose_iccs()
        
        logger.info("Reset skeleton to rest pose")
    
    def get_pose_angles(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all current joint angles as a dictionary."""
        angles = {}
        for name, segment in self.segments.items():
            angles[name] = (segment.rx, segment.ry, segment.rz)
        return angles
    
    def set_pose_angles(self, angles: Dict[str, Tuple[float, float, float]]):
        """Set all joint angles from a dictionary."""
        for name, (rx, ry, rz) in angles.items():
            segment = self.segments.get(name)
            if segment:
                segment.set_angles(rx, ry, rz)
        
        # Propagate from root
        self.propagate_fk(KP.PELVIS_CENTER)


# =============================================================================
# SKELETON SHELL FITTER - Fit skeleton joints to Y-plane cell centroids
# =============================================================================
class SkeletonShellFitter:
    """
    Fits anatomical skeleton joints to cluster shell Y-plane cell centroids.
    
    The fitting process:
    1. KP HINTS (2D/3D) narrow down the search region for each joint
    2. Find candidate Y-plane CELLS in that region
    3. SKELETON KINEMATICS moves joints to reach cell centroids via IK
    4. VALIDATE that all joints land on valid surface cells
    
    Special handling for face keypoints:
    - FACING TOWARD camera: nose/eyes on FRONT surface (smaller Y)
    - FACING AWAY from camera: nose/eyes DEEPER than surface (larger Y)
    """
    
    def __init__(self, skeleton: 'AnatomicalSkeleton'):
        """
        Initialize fitter with reference to skeleton.
        
        Args:
            skeleton: AnatomicalSkeleton instance
        """
        self.skeleton = skeleton
        self.kinematics = skeleton.get_kinematics()
        
        # Fitting parameters
        self.search_radius_xy = 10.0  # cm - search radius in X/Z from hint
        self.search_radius_y = 20.0   # cm - search radius in Y (depth)
        self.max_candidates = 5       # Maximum candidate cells per keypoint
        
        # Fitting state
        self.fitted_cells: Dict[int, Tuple[int, int, int]] = {}  # kp_idx -> voxel_idx
        self.fitting_errors: Dict[int, float] = {}  # kp_idx -> error in cm
        self._voxel_zones: Dict[Tuple[int, int, int], int] = {}  # voxel_tuple -> zone_id
        
        logger.info("SkeletonShellFitter initialized")

    # =========================================================================
    # PHASE 2C: Expose pose angles from the segments written by Step 11
    # =========================================================================

    def get_pose_angles(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get current pose angles from all segments.

        Delegates to skeleton.segments which were written by
        back_calculate_all_segment_angles() in Step 11.

        Returns:
            {segment_name: (rx, ry, rz)} for all segments with DoF > 0
        """
        angles = {}
        for seg_name, seg in self.skeleton.segments.items():
            if seg.get_dof() > 0:
                angles[seg_name] = (seg.rx, seg.ry, seg.rz)
        return angles

    # =========================================================================
    # MAIN FITTING METHOD
    # =========================================================================

    def fit_to_cluster_shell(self,
                             voxel_grid,
                             keypoints_2d_mapping: List[Dict],
                             keypoints_3d_mapping: List[Dict],
                             cluster_voxel_indices: Set[Tuple[int, int, int]],
                             facing_direction: str = 'toward_camera',
                             camera_params: Optional[Dict] = None,
                             ply_mesh=None,
                             poisson_ply_path: Optional[str] = None,
                             body_yaw_deg: Optional[float] = None,
                             spine_curve: Optional[list] = None,
                             mp33_arm_extended: bool = False,
                             pose_dof: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fit skeleton joints to cluster shell Y-plane cell centroids.

        ply_mesh: trimesh.Trimesh balloon MESH (watertight) for PLY containment.
        poisson_ply_path: path to per-frame poisson.ply for Y-depth correction.
            These are TWO DIFFERENT meshes for two different purposes.
            Pass both for full correction; each is optional independently.
            NEVER pass surface_poisson.ply (accumulated) — use per-frame only.

        camera_params is required by the revised fitting strategy (FitStrategy doc):
          - Step 3: unproject 2D MMPose pixels to rays, derive joint angles depth-invariantly
          - Step 5: depth-only shell correction (Y-axis in ICCS) via ray-shell intersection
        Accepted as Optional so existing callers without camera_params continue to work
        (they fall back to the legacy voxel-candidate path).
        
        Args:
            voxel_grid: OccupancyGrid with Y-plane data
            keypoints_2d_mapping: List of 2D keypoint mappings with voxel_under_pixel
            keypoints_3d_mapping: List of 3D keypoint mappings with grid_index + world_pos
            cluster_voxel_indices: Set of voxel indices belonging to the person cluster
            facing_direction: 24-state or legacy facing label for fitting path routing
            camera_params: Dict(focal_length, camera_position, camera_target, field_of_view)
            body_yaw_deg: CONTINUOUS body yaw angle (0=toward, 180=away) from ICCS 3D.
                          When provided, this is used DIRECTLY for mannequin rotation
                          instead of snapping to 15-degree buckets from facing_direction.
                          This gives the mannequin the exact same orientation as the
                          person's shoulder line in the point cloud.
            
        Returns:
            Dict with fitting results:
              - 'success': bool
              - 'fitted_keypoints_world': np.ndarray (17x3)
              - 'fitted_keypoints_iccs': np.ndarray (17x3)
              - 'cell_assignments': Dict[int, tuple] - kp_idx -> voxel_idx
              - 'errors': Dict[int, float] - fitting errors per keypoint
        """
        logger.info("="*60)
        logger.info(f"[SHELL_FIT] Starting fit, facing={facing_direction}, "
                    f"path={_resolve_fitting_path(facing_direction)}, "
                    f"body_yaw={body_yaw_deg:.1f}" if body_yaw_deg is not None else
                    f"[SHELL_FIT] Starting fit, facing={facing_direction}, "
                    f"path={_resolve_fitting_path(facing_direction)}, body_yaw=None")
        logger.info(f"[SHELL_FIT] Cluster has {len(cluster_voxel_indices)} voxels")

        # Store facing_direction on fitter so sub-methods called later
        # (e.g. _reanchor_computed_keypoints) can pass it to _enforce_face_geometry
        self._facing_direction = facing_direction
        self._fitting_path = _resolve_fitting_path(facing_direction)
        self._body_yaw_deg = body_yaw_deg
        self._spine_curve = spine_curve  # Previous frame's spine for blanket slicing
        self._mp33_arm_extended = mp33_arm_extended
        self._pose_dof = pose_dof or {}

        # Reset state
        self.fitted_cells = {}
        self.fitting_errors = {}

        # =====================================================================
        # OPTION 4: RAY + BONE-LENGTH SPHERE (NEW PRIMARY PATH)
        #
        # When camera_params are available, use the Option 4 pipeline:
        #   1. Build yaw-rotated mannequin from cluster bbox + facing angle
        #   2. Assign voxels to body zones using mannequin
        #   3. Place each joint via ray+sphere intersection in kinematic order
        #   4. Each joint satisfies BOTH 2D reprojection AND bone-length constraints
        #
        # Falls through to VUP/CoP if camera_params not available or if
        # Option 4 places fewer than 10 joints.
        # =====================================================================
        opt4_ok = False
        if camera_params is not None and voxel_grid is not None and len(cluster_voxel_indices) > 20:
            # Build yaw-rotated mannequin for zone assignment
            _mannequin_21 = None
            try:
                from visualization import build_skeleton21_from_cluster_bbox, rotate_skeleton21_by_yaw
                _vres = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
                _bmin = np.array(voxel_grid.bounds[0], dtype=float)
                _xs = [_bmin[0] + (v[0] + 0.5) * _vres for v in cluster_voxel_indices]
                _ys = [_bmin[1] + (v[1] + 0.5) * _vres for v in cluster_voxel_indices]
                _zs = [_bmin[2] + (v[2] + 0.5) * _vres for v in cluster_voxel_indices]
                _bbox = {
                    'min': [min(_xs), min(_ys), min(_zs)],
                    'max': [max(_xs), max(_ys), max(_zs)],
                }
                _mannequin_21 = build_skeleton21_from_cluster_bbox(_bbox)
                if _mannequin_21 is not None:
                    # Determine yaw angle for mannequin rotation.
                    # Default mannequin faces AWAY (≈180°).
                    # Rotation delta = target_yaw - 180°.
                    # PRIORITY: body_yaw_deg (continuous ICCS-derived) > 
                    #           FACING_ANGLE_MAP (discrete 15° buckets) >
                    #           stored ICCS yaw history
                    if body_yaw_deg is not None:
                        # BEST: exact continuous angle from ICCS 3D shoulder line.
                        _yaw_deg = body_yaw_deg - 180.0
                    else:
                        _target = FACING_ANGLE_MAP.get(facing_direction, 180.0)
                        _yaw_deg = _target - 180.0
                        if abs(_yaw_deg) < 0.1 and facing_direction != 'away_from_camera':
                            if hasattr(self.skeleton, 'iccs') and self.skeleton.iccs.yaw != 0:
                                _yaw_deg = self.skeleton.iccs.yaw - 180.0
                    _fitting_path = _resolve_fitting_path(facing_direction)
                    _mannequin_21 = rotate_skeleton21_by_yaw(_mannequin_21, _yaw_deg)
                    logger.info(f"[OPT4] Mannequin rotated by delta={_yaw_deg:.1f} "
                               f"(body_yaw={body_yaw_deg if body_yaw_deg is not None else '?'}, "
                               f"default=180, facing={facing_direction}, path={_fitting_path}) "
                               f"for zone assignment")
            except ImportError:
                logger.debug("[OPT4] Could not import rotate_skeleton21_by_yaw — proceeding without zones")
            except Exception as _e_mann:
                logger.debug(f"[OPT4] Mannequin construction failed: {_e_mann}")

            opt4_placed = self._fit_via_ray_sphere_chain(
                voxel_grid=voxel_grid,
                keypoints_2d_mapping=keypoints_2d_mapping,
                keypoints_3d_mapping=keypoints_3d_mapping,
                cluster_voxel_indices=cluster_voxel_indices,
                facing_direction=facing_direction,
                camera_params=camera_params,
                mannequin_world_21=_mannequin_21,
                mp33_arm_extended=mp33_arm_extended,
                pose_dof=pose_dof,
            )

            _l_hip_ok = not np.allclose(self.skeleton.keypoints_world[KP.LEFT_HIP], 0)
            _r_hip_ok = not np.allclose(self.skeleton.keypoints_world[KP.RIGHT_HIP], 0)

            if opt4_placed >= 10 and _l_hip_ok and _r_hip_ok:
                iccs_ok = self._establish_iccs_from_placed_joints(facing_direction)
                if iccs_ok:
                    opt4_ok = True
                    self._compute_extended_joints_from_placed(facing_direction)
                    self._compute_reprojection_errors(keypoints_2d_mapping, camera_params)
                    logger.info(f"[OPT4] PRIMARY PATH OK: {opt4_placed}/17 joints placed "
                               f"via ray+sphere chain")

            if not opt4_ok and opt4_placed > 0:
                logger.info(f"[OPT4] Insufficient ({opt4_placed}/17) — falling back to VUP/CoP")

        # =====================================================================
        # NEW STEP 0: VOXEL-SURFACE PRIMARY PLACEMENT
        #
        # Place ALL body joints (KP 5-16) from voxel_under_pixel + flesh.
        # If enough joints placed (≥8 AND both hips), establish ICCS,
        # solve two-bone IK for knees/elbows, compute extended joints,
        # and skip old CoP Steps 1-8.
        # SKIPPED when opt4_ok=True — joints already placed via ray+sphere.
        # =====================================================================
        vup_ok = False
        if not opt4_ok:
            vup_placed = self._place_all_from_voxel_surface(
                voxel_grid, keypoints_2d_mapping, keypoints_3d_mapping,
                cluster_voxel_indices, facing_direction
            )
            _l_hip_ok = not np.allclose(self.skeleton.keypoints_world[KP.LEFT_HIP], 0)
            _r_hip_ok = not np.allclose(self.skeleton.keypoints_world[KP.RIGHT_HIP], 0)

            if vup_placed >= 8 and _l_hip_ok and _r_hip_ok:
                iccs_ok = self._establish_iccs_from_placed_joints(facing_direction)
                if iccs_ok:
                    vup_ok = True
                    ik_chains = self._enforce_bone_lengths_two_bone_ik()
                    self._compute_extended_joints_from_placed(facing_direction)
                    if camera_params is not None:
                        self._compute_reprojection_errors(keypoints_2d_mapping, camera_params)
                    logger.info(f"[VUP] Primary placement OK: {vup_placed}/12 joints, "
                               f"{ik_chains} IK chains solved")

            if not vup_ok and vup_placed > 0:
                logger.info(f"[VUP] Insufficient ({vup_placed}/12) — falling back to CoP")
        
        # =====================================================================
        # STEPS 1-3: CoP 3D-DRIVEN PLACEMENT  (FitStrategy §4, revised)
        # SKIPPED when opt4_ok=True or vup_ok=True.
        # =====================================================================
        cop3d_ok = False
        if not opt4_ok and not vup_ok:
            # Compute world-space Y bounds of the cluster so _establish_iccs_from_cop3d
            # can ground the pelvis inside the cluster when CoP places it outside.
            _cluster_y_bounds = None
            if cluster_voxel_indices and hasattr(voxel_grid, 'bounds') and voxel_grid.bounds is not None:
                _vres = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
                _y0   = voxel_grid.bounds[0][1]
                _ys   = [_y0 + (v[1] + 0.5) * _vres for v in cluster_voxel_indices]
                _cluster_y_bounds = (min(_ys), max(_ys))
            cop3d_ok = self._establish_iccs_from_cop3d(keypoints_3d_mapping,
                                                        cluster_y_bounds=_cluster_y_bounds)

        if not opt4_ok and not vup_ok and not cop3d_ok:
            # -----------------------------------------------------------------
            # FALLBACK: CoP world_pos unavailable — use old voxel-candidate path
            # -----------------------------------------------------------------
            logger.warning("[SHELL_FIT] world_pos unavailable for hips, using candidate fallback")
            hint_targets = self._build_hint_targets(
                voxel_grid, keypoints_2d_mapping, keypoints_3d_mapping,
                cluster_voxel_indices=cluster_voxel_indices
            )
            candidate_cells = self._find_candidate_cells(voxel_grid, hint_targets, cluster_voxel_indices)
            if not self._establish_iccs_from_candidates(voxel_grid, candidate_cells):
                logger.warning("[SHELL_FIT] ICCS establishment failed in fallback too")
                return {'success': False, 'error': 'ICCS establishment failed'}
            self._place_limb_roots_from_candidates(voxel_grid, candidate_cells)
            self._fit_end_effectors(voxel_grid, candidate_cells, facing_direction)
            self._fit_intermediate_joints(voxel_grid, candidate_cells)
            self._fit_face_keypoints(voxel_grid, candidate_cells, facing_direction)
        elif not opt4_ok and not vup_ok:
            # -----------------------------------------------------------------
            # STEP 2: Place all 17 keypoints from CoP 3D world_pos
            # -----------------------------------------------------------------
            placed = self._place_all_keypoints_from_cop3d(keypoints_3d_mapping)
            logger.info(f"[SHELL_FIT] Placed {placed}/15 non-hip keypoints from CoP 3D world_pos")

            # -----------------------------------------------------------------
            # STEP 3: 2D MMPose pixel → ray → bone-sphere refinement.
            #
            # Level 1 guidance (per skeleton21_humanoid_mannequin_v3.docx §7
            # and RO_NOUS_21Joint_Pipeline.docx §3 Step 5):
            # For KP 7,8 (elbows), KP 9,10 (wrists), KP 13,14 (knees):
            #   - Read child joint middle_panel_pixel from keypoints_2d_mapping
            #   - Unproject to ray direction (depth-invariant, sub-pixel accurate)
            #   - Intersect ray with sphere of radius=bone_length centred on parent
            #   - Near-side root chosen (joint in front, not behind torso)
            #   - Accept if discriminant ≥ 0 and 0.1L < distance < 2.5L
            #   - Updates keypoints_iccs + keypoints_world for the child joint
            #
            # Three defects previously disabled this step:
            #   3a: read 'left_panel_pixel' → FIXED: uses middle_panel_pixel
            #   3b: hardcoded panel size → FIXED: reads panel_width/height from camera_params
            #   3c: far-side root first → FIXED: near-side root chosen first
            #
            # Ankles (KP 15,16) are end-effectors — no child defines direction.
            # They stay from CoP 3D (placed in Step 2). Per spec §2.
            # -----------------------------------------------------------------
            if camera_params is not None:
                refined = self._refine_limbs_from_2d_rays(
                    keypoints_2d_mapping, keypoints_3d_mapping, camera_params
                )
                if refined > 0:
                    logger.info(f"[SHELL_FIT] Step 3: 2D MMPose guided {refined} distal joints via pixel→ray")

            # -----------------------------------------------------------------
            # STEP 4: Face keypoints depth adjustment (facing-aware)
            # -----------------------------------------------------------------
            self._fit_face_keypoints_cop3d(keypoints_3d_mapping, facing_direction)

            # -----------------------------------------------------------------
            # STEP 5: Derive the 4 anatomical keypoints (17-20)
            # SHOULDER_CENTER, SPINE_MID, NECK_BASE, HEAD_CENTER
            # from the placed COCO positions, not from voxel search.
            # -----------------------------------------------------------------
            self._derive_anatomical_keypoints()

            # -----------------------------------------------------------------
            # STEP 5b: Compute reprojection errors (BUG #3 FIX).
            #
            # In the CoP-3D path, self.fitting_errors was never populated —
            # it stayed empty, so avg_fitting_error_cm = 0.00 every frame.
            # This disabled the entire quality-gate system:
            #   • Calibration guard (avg_error < 5.0) always passed → bad frames
            #     fed into calibration and corrupted bone-length learning.
            #   • Per-joint fallback (fit_err > 12.0) never fired → joints with
            #     badly placed positions were never reverted to the previous frame.
            #   • Velocity clamping quality check was effectively dead.
            #
            # Fix: project each placed 3D ICCS position back to 2D using the
            # ICCS→world→pixel transform, then compare to the MMPose
            # middle_panel_pixel. The pixel distance (converted to cm) is stored
            # in self.fitting_errors[kp_idx] so every downstream quality gate
            # now has real signal to work with.
            # -----------------------------------------------------------------
            if camera_params is not None:
                self._compute_reprojection_errors(keypoints_2d_mapping, camera_params)

        # =====================================================================
        # STEP 6: Convert ALL ICCS positions -> world
        # =====================================================================
        for kp_idx in range(NUM_KEYPOINTS):
            self.skeleton.keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(
                self.skeleton.keypoints_iccs[kp_idx]
            )

        keypoints_world = np.zeros((17, 3))
        for kp_idx in range(17):
            keypoints_world[kp_idx] = self.skeleton.keypoints_world[kp_idx].copy()

        # =====================================================================
        # STEP 7: Depth correction — Y-axis (shell depth) only.
        # SKIPPED when opt4_ok=True or vup_ok=True (joints already placed
        # correctly — re-running depth correction would fight the placement).
        # =====================================================================
        correction_result = {'corrected_count': 0, 'method': 'vup_skip'}
        if not opt4_ok and not vup_ok:
            if cop3d_ok:
                correction_result = self._correct_depth_from_shell(
                    voxel_grid, cluster_voxel_indices
                )
            else:
                correction_result = self._validate_and_correct_from_2d_anchors(
                    voxel_grid, keypoints_2d_mapping, cluster_voxel_indices
                )

            # Rebuild world coords after depth correction
            if correction_result.get('corrected_count', 0) > 0:
                for kp_idx in range(17):
                    keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(
                        self.skeleton.keypoints_iccs[kp_idx]
                    )

        # =====================================================================
        # STEP 7b: POISSON PLY DEPTH REFINEMENT (FitStrategy v3 Steps 4-6)
        #
        # The voxel shell correction (Step 7) has 2cm resolution. The Poisson
        # PLY has sub-centimeter precision. If a per-frame poisson.ply was
        # loaded and passed, refine every joint's Y (depth) from the PLY
        # surface + flesh_radius. X and Z are unchanged.
        #
        # This is the fix for:
        #  - BUG B3: joints placed ON surface (no flesh offset) — now corrected
        #  - 10-12cm backward drift in knees/ankles — pulled back to surface
        #  - Right wrist depth anomaly (12.7cm forward) — caught by 30cm clamp
        # =====================================================================
        if not opt4_ok and not vup_ok and poisson_ply_path is not None:
            poisson_corrected = self._apply_poisson_depth_correction(
                poisson_ply_path, facing_direction
            )
            if poisson_corrected > 0:
                # Rebuild world coords after Poisson correction
                for kp_idx in range(17):
                    keypoints_world[kp_idx] = self.skeleton.keypoints_world[kp_idx].copy()
                logger.info(f"[STEP 7b] Poisson PLY depth correction: {poisson_corrected} joints refined")

                # ─────────────────────────────────────────────────────────────
                # STEP 7c: POST-POISSON BILATERAL Y-EQUALIZATION (world space)
                #
                # Poisson depth correction picks one PLY vertex per joint.
                # MiDaS 2.5D depth noise means L and R vertices at identical
                # XZ positions can report wildly different Y.  Re-equalize the
                # 6 bilateral pairs in WORLD Y *immediately* so downstream
                # Steps 9/10 start from symmetric positions.
                # Threshold: 2cm — tighter than Step 9 (3-15cm) because at
                # this point the only Y divergence is Poisson noise.
                # ─────────────────────────────────────────────────────────────
                _BILATERAL_PAIRS_7c = [
                    (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, 'shoulders'),
                    (KP.LEFT_ELBOW,    KP.RIGHT_ELBOW,    'elbows'),
                    (KP.LEFT_WRIST,    KP.RIGHT_WRIST,    'wrists'),
                    (KP.LEFT_HIP,      KP.RIGHT_HIP,      'hips'),
                    (KP.LEFT_KNEE,     KP.RIGHT_KNEE,     'knees'),
                    (KP.LEFT_ANKLE,    KP.RIGHT_ANKLE,    'ankles'),
                ]
                _pairs_eq_7c = 0
                for l_kp, r_kp, label in _BILATERAL_PAIRS_7c:
                    lw = self.skeleton.keypoints_world[l_kp]
                    rw = self.skeleton.keypoints_world[r_kp]
                    if np.allclose(lw, 0) or np.allclose(rw, 0):
                        continue
                    y_diff = abs(lw[1] - rw[1])
                    if y_diff > 2.0:
                        avg_y = (lw[1] + rw[1]) / 2.0
                        self.skeleton.keypoints_world[l_kp][1] = avg_y
                        self.skeleton.keypoints_world[r_kp][1] = avg_y
                        # Sync back to ICCS
                        self.skeleton.keypoints_iccs[l_kp] = \
                            self.skeleton.iccs.world_to_iccs(self.skeleton.keypoints_world[l_kp])
                        self.skeleton.keypoints_iccs[r_kp] = \
                            self.skeleton.iccs.world_to_iccs(self.skeleton.keypoints_world[r_kp])
                        _pairs_eq_7c += 1
                if _pairs_eq_7c > 0:
                    # Rebuild keypoints_world local array
                    for kp_idx in range(17):
                        keypoints_world[kp_idx] = self.skeleton.keypoints_world[kp_idx].copy()
                    logger.info(f"[STEP 7c] Post-Poisson bilateral Y-equalized "
                                f"{_pairs_eq_7c} pairs (depth noise removal)")


        #
        # Previously STEP 8 only ran in the fallback (cop3d_ok=False) path.
        # In the CoP path, fitting_errors contained raw reprojection distances
        # (25–141 cm) — never corrected — so _apply_velocity_clamping_iccs
        # saw large errors on every joint and had no signal to act on.
        #
        # Fix: run STEP 8 in BOTH paths.  For each body joint (shoulders,
        # elbows, wrists, hips, knees, ankles) whose current world position
        # disagrees with its voxel_under_pixel centroid by more than 5 cm,
        # move the joint to the centroid, enforce sub-chain bone lengths, and
        # update fitting_errors to the post-correction residual.
        #
        # =====================================================================
        # STEP 8: Validate and correct from voxel_under_pixel (CoP path only)
        # SKIPPED when opt4_ok=True or vup_ok=True.
        # =====================================================================
        if not opt4_ok and not vup_ok and cop3d_ok:
            step8_result = self._validate_and_correct_from_2d_anchors(
                voxel_grid, keypoints_2d_mapping, cluster_voxel_indices
            )
            if step8_result.get('corrected_count', 0) > 0:
                logger.info(f"[STEP 8] CoP path: corrected {step8_result['corrected_count']} joints, "
                           f"avg {step8_result.get('avg_error_before', 0):.1f}cm → "
                           f"{step8_result.get('avg_error_after', 0):.1f}cm")
                # Rebuild world coords after step 8 corrections
                for kp_idx in range(17):
                    keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(
                        self.skeleton.keypoints_iccs[kp_idx]
                    )

        # =====================================================================
        # STEP 9: Cross-body bilateral depth sanity check (FIRST PASS)
        # Operates in ICCS before bone-length enforcement.
        # =====================================================================
        self._enforce_bilateral_depth_sanity()

        # =====================================================================
        # STEP 10: ENFORCE LOCKED BONE LENGTHS
        # Preserves each bone's DIRECTION but normalizes LENGTH to locked
        # proportional value.  Guarantees frame-to-frame consistency.
        # =====================================================================
        self._enforce_bone_lengths()
        self._reanchor_computed_keypoints()

        # =====================================================================
        # STEP 10b: PER-JOINT VELOCITY CLAMPING + DIRECTION CONSISTENCY
        #
        # Compares each joint's new ICCS position against previous frame.
        # Caps displacement magnitude AND dampens direction reversals.
        # After clamping: re-enforce bone lengths, then RE-RUN bilateral
        # sanity (because clamping can break L/R symmetry).
        # =====================================================================
        clamped = self._apply_velocity_clamping_iccs()
        if clamped > 0:
            self._enforce_bone_lengths()
            self._reanchor_computed_keypoints()
            self._enforce_bilateral_depth_sanity()  # SECOND PASS after clamping
            self._enforce_bone_lengths()             # Re-enforce after bilateral fix
            self._reanchor_computed_keypoints()
            logger.info(f"[STEP 10b] Velocity-clamped {clamped} joints, "
                       f"bone lengths + bilateral re-enforced")

        # Rebuild world coordinates from final ICCS positions
        for kp_idx in range(NUM_KEYPOINTS):
            self.skeleton.keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(
                self.skeleton.keypoints_iccs[kp_idx]
            )

        # Rebuild the 17-kp world array for the return dict
        keypoints_world = np.zeros((17, 3))
        for kp_idx in range(17):
            keypoints_world[kp_idx] = self.skeleton.keypoints_world[kp_idx].copy()

        # =====================================================================
        # STEP 11: Back-calculate ALL segment angles from fitted positions
        # =====================================================================
        try:
            from movement_index import back_calculate_all_segment_angles
            back_calculate_all_segment_angles(self.skeleton)
            logger.info("[STEP 11] All segment angles back-calculated from fitted positions")
        except ImportError:
            logger.warning("[STEP 11] movement_index not available, skipping angle back-calculation")
        except Exception as e:
            logger.warning(f"[STEP 11] Angle back-calculation failed: {e}")

        # =====================================================================
        # STEP 12: ENFORCE ROM (Range of Motion) LIMITS
        #
        # After angles are back-calculated, verify each segment's angles
        # fall within anatomical ROM.  If violated, clamp to nearest
        # valid angle and re-propagate the affected chain via FK.
        # This prevents physically impossible poses from entering the
        # temporal history (which would poison future velocity clamping).
        # =====================================================================
        rom_violations = self._enforce_rom_limits()
        if rom_violations > 0:
            # Positions changed by ROM clamping → re-enforce bone lengths
            # and rebuild world coordinates
            self._enforce_bone_lengths()
            self._reanchor_computed_keypoints()
            self._enforce_face_geometry(facing_direction)
            for kp_idx in range(NUM_KEYPOINTS):
                self.skeleton.keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(
                    self.skeleton.keypoints_iccs[kp_idx]
                )
            keypoints_world = np.zeros((17, 3))
            for kp_idx in range(17):
                keypoints_world[kp_idx] = self.skeleton.keypoints_world[kp_idx].copy()
            logger.info(f"[STEP 12] ROM enforced on {rom_violations} segments, "
                       f"positions re-propagated")


        # =====================================================================
        # Calculate final fitting errors
        # =====================================================================
        total_error = sum(self.fitting_errors.values())
        avg_error = total_error / max(len(self.fitting_errors), 1)

        logger.info(f"[SHELL_FIT] Complete: avg_error={avg_error:.2f}cm")
        logger.info("=" * 60)

        # Build full 21-keypoint world array (BUG 7 FIX)
        keypoints_world_21 = np.zeros((NUM_KEYPOINTS, 3))
        for kp_idx in range(NUM_KEYPOINTS):
            keypoints_world_21[kp_idx] = self.skeleton.keypoints_world[kp_idx].copy()

        # =====================================================================
        # LIMB DoF SANITIZE (live shell-fit path) — fixes splayed legs / jutting
        # arms.  The shell fitter places limbs from CoP-3D + 2D-ray data that
        # carries the unreliable depth axis, so thighs/upper-arms can end up
        # near-horizontal.  Convert the FINAL joints to ICCS (Z=up, Y=depth),
        # damp the depth component (harder in profile views) and clamp each
        # limb's tilt away from straight-down, then convert back.  Bone lengths
        # are preserved (direction-only correction at the stored length).
        # Operates here, on the real fitter output, so it actually reaches the
        # rendered fitted_keypoints_world_21 (the previous edit was on the
        # unused update_from_detection path).
        # =====================================================================
        # -- LOCKED-LENGTH FK SOLVE (supersedes the limb sanitize above) -------
        # The placement above yields per-frame DETECTED joint directions on a
        # noisy half-shell.  Re-pose a rest skeleton whose lengths are LOCKED to
        # stature H (ratio x H, never re-measured) by ANGLES only: FK outward
        # from the pelvis, anatomical swing limits (hinges cannot splay),
        # occluded limbs mirrored, unsignaled joints relaxed to rest, seeded by
        # the previous frame.  Bone lengths exact; splay/crouch impossible.
        # Frame: ICCS (Z=up, Y=facing/depth, pelvis=origin).
        try:
            _lk_H = float(getattr(self.skeleton, 'height', 0.0)) or 170.0
            _lk_rest = _lk_build_rest_skeleton(_lk_H)
            _lk_det = np.zeros((NUM_KEYPOINTS, 3)); _lk_conf = np.zeros(NUM_KEYPOINTS)
            for _lk_i in range(NUM_KEYPOINTS):
                _lk_w = keypoints_world_21[_lk_i]
                if np.abs(_lk_w).sum() > 1e-6:
                    _lk_det[_lk_i] = self.skeleton.iccs.world_to_iccs(_lk_w)
                    _lk_conf[_lk_i] = 1.0
            if _LK_MANUAL_TEST:
                # Manual FK test: rotate ONLY each listed child's PARENT bone so
                # the child reaches (rest + offset).  No target for the child's
                # own bone -> it stays rigid at rest relative to its parent
                # ("keep its DoF as is").  Targets are unit directions parent->
                # desired-child, in ICCS.
                _lk_targets = {}
                for _cj, _off in _LK_MANUAL_OFFSETS.items():
                    _pj = _LK_PARENT[_cj]
                    _desired = _lk_rest[_cj] + np.asarray(_off, float)
                    _dir = _desired - _lk_rest[_pj]
                    _n = np.linalg.norm(_dir)
                    if _n > 1e-6:
                        _lk_targets[_cj] = _dir / _n
                        logger.info(f"[LK-MANUAL] parent {_pj} -> child {_cj}: "
                                    f"offset(ICCS X,Y,Z)={tuple(_off)} cm via parent DoF "
                                    f"(child bone kept at rest)")
            elif _LK_TEST_LIMBS:
                # Pose ONLY the LEFT leg (thigh 11->13, shin 13->15) and RIGHT
                # arm (upper 6->8, forearm 8->10), each from its OWN-side
                # detected direction.  No contralateral mirror, no temporal
                # blend -> sides cannot flip.  All other bones -> REST.
                _lk_targets = {}
                _lk_test_children = (KP.LEFT_KNEE, KP.LEFT_ANKLE,      # left thigh, shin
                                     KP.RIGHT_ELBOW, KP.RIGHT_WRIST)   # right upper-arm, forearm
                for _c in _lk_test_children:
                    _p = _LK_PARENT[_c]
                    if _lk_conf[_c] > 0.3 and _lk_conf[_p] > 0.3:
                        _d = (_lk_det[_c] - _lk_det[_p]).astype(float)
                        _d[1] *= 0.25                  # damp unreliable ICCS depth axis
                        _n = np.linalg.norm(_d)
                        if _n > 1e-6:
                            _lk_targets[_c] = _d / _n
            elif _LK_REST_ONLY:
                _lk_targets = {}                       # bare rest pose; yaw comes from ICCS
            else:
                _lk_prev = getattr(self.skeleton, '_lk_prev_targets', None)
                _lk_targets, _ = _lk_build_targets(_lk_det, _lk_conf, prev_targets=_lk_prev)
            # ---- ORIENTATION: handled ENTIRELY by iccs_to_world ---------------
            # The skeleton is built body-local in ICCS (LEFT=+X, nose=-Y) and the
            # ICCS->world transform applies Rz(iccs.yaw); iccs.yaw comes from
            # FACING_ANGLE_MAP (toward=0, away=180, ...), so iccs_to_world alone
            # rotates the -Y rest nose to the correct world facing (e.g. away ->
            # nose at +Y, away from the low-Y camera).  We therefore solve the FK
            # with NO root rotation (root_R = identity).
            #
            # (Earlier this applied root_R=Rz(yaw) derived from _lk_det's shoulder
            # line + a facing hemisphere flip.  That was wrong twice over: _lk_det
            # is BODY-LOCAL — world_to_iccs at the placement step already removed
            # iccs.yaw — so its shoulders sit ~rest-aligned and the hemisphere
            # flip forced a spurious 180; and even a correct value would DOUBLE-
            # rotate, stacking with iccs.yaw (180+180 -> 0) and leaving the face
            # pointing AT the camera.  Orientation belongs to the ICCS frame, not
            # a second rotation here.)
            _lk_free = (set(_LK_MANUAL_OFFSETS.keys()) if _LK_MANUAL_TEST else None)
            _lk_posed = _lk_solve_fk(_lk_H, _lk_rest, _lk_targets,
                                     _lk_rest[KP.PELVIS_CENTER].copy(),
                                     free_children=_lk_free)
            self.skeleton._lk_prev_targets = _lk_targets
            keypoints_world_21 = self.skeleton.iccs.iccs_to_world_batch(_lk_posed)
            # ---- DEPTH ANCHOR: seat the pelvis at shell FRONT-SURFACE + locked
            # half-body-depth, INSIDE the shell.  Root cause this fixes: the
            # world transform is `world = R*iccs + origin` with origin = pelvis
            # world center, and the FK roots at the canonical rest pelvis, so the
            # rendered pelvis lands at whatever depth the placement wrote (~10-20cm
            # in front of the shell).  _correct_depth_from_shell snapped joints to
            # the surface earlier but the FK rejoin rebuilt all world coords from
            # origin, discarding it.  Re-seat the whole posed figure in world-Y
            # (a rigid translation -> pose/orientation preserved) so the pelvis
            # sits half a body-depth behind the front surface.  Front surface =
            # 2nd-pct of cluster-voxel world-Y (camera at low Y looks toward high
            # Y); depth is the genuinely unreliable axis, so it is anchored to the
            # shell + a LOCKED offset, never the detected/centroid depth.
            try:
                _lk_vres_y = float(getattr(voxel_grid, 'resolution', 2.0))
                _lk_by = float(voxel_grid.bounds[0][1])
                _lk_cys = np.fromiter((_lk_by + (v[1] + 0.5) * _lk_vres_y
                                       for v in cluster_voxel_indices), float)
                if _lk_cys.size:
                    _lk_front_y = float(np.percentile(_lk_cys, 2.0))
                    _lk_pelvis_y0 = float(keypoints_world_21[KP.PELVIS_CENTER][1])
                    _lk_pelvis_yT = _lk_front_y + _LK_HALF_DEPTH_FRAC * _lk_H
                    _lk_dy = _lk_pelvis_yT - _lk_pelvis_y0
                    keypoints_world_21[:, 1] += _lk_dy
                    logger.info(f"[LK-DEPTH] front_y={_lk_front_y:.1f} "
                                f"half_depth={_LK_HALF_DEPTH_FRAC*_lk_H:.1f}cm "
                                f"pelvis_y {_lk_pelvis_y0:.1f}->{_lk_pelvis_yT:.1f} "
                                f"(dy={_lk_dy:+.1f}cm) -> seated inside shell")
            except Exception as _lk_depth_exc:
                logger.warning(f"[LK-DEPTH] skipped: {_lk_depth_exc}")
            # Floor = low percentile of cluster-voxel world-Z (shell bottom),
            # using the same voxel->world convention as line ~3535.  Drop the
            # whole posed figure by one dz so the lower ankle rests on the floor.
            # World and ICCS are both Z-up (yaw is about Z), so shift both by dz.
            try:
                _lk_vres = float(getattr(voxel_grid, 'resolution', 2.0))
                # voxel_grid.bounds is the FIXED COP capture box (min Z = -170),
                # NOT the floor -- so it is used ONLY as the origin for the
                # voxel-index -> world conversion below, never as the floor.
                _lk_bz = float(voxel_grid.bounds[0][2])
                _lk_czs = np.fromiter((_lk_bz + (v[2] + 0.5) * _lk_vres
                                       for v in cluster_voxel_indices), float)
                if _lk_czs.size:
                    # Floor = the cluster's LOWEST captured voxel (the person's
                    # ground contact: they stand on the floor and sub-floor
                    # points are pre-filtered at ~-152.8, so the lowest body
                    # voxel sits on the ground).  Use the actual MIN -- it is a
                    # steady -153 across the run -- so the feet land exactly on
                    # the floor (0.5-pct sat ~2cm high at -151).  NOT the COP box
                    # bottom (-170), NOT the 2nd-pct (-149, rejected the feet).
                    _lk_floor_z = float(_lk_czs.min())
                    _lk_low_ankle = min(float(keypoints_world_21[KP.LEFT_ANKLE][2]),
                                        float(keypoints_world_21[KP.RIGHT_ANKLE][2]))
                    _lk_dz = _lk_floor_z - _lk_low_ankle
                    keypoints_world_21[:, 2] += _lk_dz
                    logger.info(f"[LK-LAND] floor_z={_lk_floor_z:.1f} (cluster floor-contact, "
                                f"cluster_min={_lk_czs.min():.1f} box_bottom={_lk_bz:.1f}) "
                                f"dz={_lk_dz:+.1f}cm -> ankle on floor")
            except Exception as _lk_land_exc:
                logger.warning(f"[LK-LAND] skipped: {_lk_land_exc}")
            # World coords now carry the depth (Y) + floor (Z) re-seating; derive
            # ICCS from the FINAL world so the two stay consistent (the world-Y
            # shift does not map onto _lk_posed's ICCS Y after the Z-yaw).
            for _lk_i in range(NUM_KEYPOINTS):
                self.skeleton.keypoints_world[_lk_i] = keypoints_world_21[_lk_i]
                self.skeleton.keypoints_iccs[_lk_i] = \
                    self.skeleton.iccs.world_to_iccs(keypoints_world_21[_lk_i])
            _lk_mode = ('MANUAL-TEST j5->j7,j11->j13' if _LK_MANUAL_TEST
                        else 'TEST-LIMBS L-leg(11/13/15)+R-arm(6/8/10)' if _LK_TEST_LIMBS
                        else 'REST (DoF=0)' if _LK_REST_ONLY
                        else 'locked-length FK')
            # Orientation check in WORLD: camera sits at LOW Y looking toward
            # HIGH Y, so for away_from_camera the nose must be DEEPER than the
            # head (nose_dY > 0); for toward_camera, nose_dY < 0.  Orientation
            # is produced by iccs.yaw (logged), NOT a root rotation here.
            _lk_nose_dY = float(keypoints_world_21[KP.NOSE][1]
                                - keypoints_world_21[KP.HEAD_CENTER][1])
            _lk_iccs_yaw = float(getattr(self.skeleton.iccs, 'yaw', 0.0))
            logger.info(f"[LK-FK] {_lk_mode}: H={_lk_H:.0f}cm iccs.yaw={_lk_iccs_yaw:.1f} "
                        f"facing={facing_direction} | "
                        f"nose_dY={_lk_nose_dY:+.1f} (away:want>0, toward:want<0) "
                        f"L_sh_X={keypoints_world_21[KP.LEFT_SHOULDER][0]:+.1f} "
                        f"R_sh_X={keypoints_world_21[KP.RIGHT_SHOULDER][0]:+.1f}")
        except Exception as _lk_exc:
            logger.warning(f"[LK-FK] solve skipped: {_lk_exc}")

        # =====================================================================
        # STEP 13: LIVE BONE-LENGTH CALIBRATION
        #
        # Measure real bone lengths from the FINAL fitted world positions
        # (after all enforcement steps) and feed them as calibration samples.
        # Once enough good samples are collected, the skeleton's bone_lengths
        # are updated to match this specific person — replacing the generic
        # anthropometric template.
        #
        # Quality gate: only accept frames with avg_error < threshold.
        # Outlier gate: reject individual bones that deviate > 30% from
        #               current template (likely fitting artifacts).
        # Lock: after N good samples, mark is_calibrated=True.
        # =====================================================================
        # Calibration threshold: after STEP 8 corrects joints toward
        # voxel_under_pixel centroids, avg_error reflects post-correction
        # residuals (cm distance from corrected position to target centroid).
        # 15 cm accepts frames where most joints are well-fitted but a few
        # distal joints (wrists, ankles) have no voxel_under_pixel hit.
        _CALIB_ERROR_THRESHOLD = 15.0
        _CALIB_REQUIRED_SAMPLES = 5    # frames needed before locking
        _CALIB_OUTLIER_TOLERANCE = 0.30  # max deviation from template (30%)

        if not self.skeleton.is_calibrated and avg_error < _CALIB_ERROR_THRESHOLD:
            # Measure bone lengths from final world positions (21-kp array)
            measured = self.skeleton._measure_bone_lengths(keypoints_world_21)

            if len(measured) >= 4:
                # Reject outlier bones: > 30% off template
                filtered = {}
                for bone_name, measured_len in measured.items():
                    template_len = self.skeleton.bone_lengths.get(bone_name)
                    if template_len and template_len > 0:
                        deviation = abs(measured_len - template_len) / template_len
                        if deviation <= _CALIB_OUTLIER_TOLERANCE:
                            filtered[bone_name] = measured_len
                        else:
                            logger.debug(
                                f"[STEP 13] Rejected {bone_name}: measured={measured_len:.1f}cm "
                                f"vs template={template_len:.1f}cm (deviation={deviation:.0%})")
                    else:
                        filtered[bone_name] = measured_len

                # ── BUG-8 FIX: Hard physiological bounds ─────────────────────
                # The 30% deviation filter above is relative to the template,
                # but the template itself can drift if earlier frames fed bad
                # samples (e.g. inflated 128cm-wide cluster → shoulder=58cm).
                # Absolute bounds are anatomy-level hard gates that cannot be
                # fooled by template drift.  Adult ranges:
                #   shoulder_width: 26–52cm  (biacromial 32–48cm ±safety)
                #   hip_width:       8–42cm  (bicristal 25–38cm ±safety)
                _PHYS_BOUNDS = {
                    'shoulder_width': (26.0, 52.0),
                    'hip_width':      ( 8.0, 42.0),
                }
                _phys_reject = []
                for _pbone, (_pmin, _pmax) in _PHYS_BOUNDS.items():
                    if _pbone in filtered:
                        _pval = filtered[_pbone]
                        if not (_pmin <= _pval <= _pmax):
                            _phys_reject.append(_pbone)
                            logger.warning(
                                f"[BUG-8 FIX] Rejected {_pbone}={_pval:.1f}cm — "
                                f"outside physiological bounds [{_pmin:.0f},{_pmax:.0f}]cm. "
                                f"Cluster geometry is likely inflated (chair leak). "
                                f"Keeping locked template value "
                                f"{self.skeleton.bone_lengths.get(_pbone, 0):.1f}cm.")
                for _pb in _phys_reject:
                    del filtered[_pb]
                # ── END BUG-8 FIX ─────────────────────────────────────────────

                if len(filtered) >= 4:
                    # Confidence inversely proportional to error
                    confidence = max(0.1, 1.0 - (avg_error / _CALIB_ERROR_THRESHOLD))

                    self.skeleton._calibration_samples.append({
                        'lengths': filtered,
                        'confidence': confidence
                    })

                    # Update bone lengths from accumulated samples
                    self.skeleton._update_bone_lengths_from_samples()

                    n_samples = len(self.skeleton._calibration_samples)
                    logger.info(
                        f"[STEP 13] Calibration sample {n_samples}/{_CALIB_REQUIRED_SAMPLES}: "
                        f"{len(filtered)} bones measured, confidence={confidence:.2f}, "
                        f"avg_error={avg_error:.2f}cm")

                    # Lock after enough good samples
                    if n_samples >= _CALIB_REQUIRED_SAMPLES:
                        self.skeleton.is_calibrated = True
                        logger.info(
                            f"[STEP 13] *** CALIBRATION LOCKED *** from {n_samples} samples. "
                            f"Bone lengths now reflect real person measurements.")
                        # Log key calibrated values
                        for bn in ['torso', 'lower_spine', 'upper_spine', 'neck',
                                    'thigh_l', 'shin_l', 'upper_arm_l', 'forearm_l',
                                    'shoulder_width', 'hip_width']:
                            if bn in self.skeleton.bone_lengths:
                                logger.info(f"  {bn}: {self.skeleton.bone_lengths[bn]:.1f}cm")
        elif self.skeleton.is_calibrated:
            logger.debug("[STEP 13] Skeleton already calibrated — using locked bone lengths")

        # =====================================================================
        # STEP PLY: PLY BALLOON MESH CONTAINMENT (Level 3 guidance)
        # shell_as_suit.docx Rule R6 + skeleton21_humanoid_mannequin_v3.docx §7 Step bbox guard
        #
        # balloon.ply = the 3D watertight envelope built from this frame's
        # depth point cloud. ALL 21 joints MUST be inside it (signed distance < 0).
        # Any joint outside is pulled to the mesh interior along the inward normal.
        # This makes the skeleton TIGHTLY fitted to the cluster shell — without it
        # joints float outside and the skeleton is visually wrong.
        # =====================================================================
        if ply_mesh is not None:
            _ply_corrections = self._enforce_ply_containment(ply_mesh, keypoints_world_21)
            if _ply_corrections > 0:
                for _ki in range(17):
                    keypoints_world[_ki] = keypoints_world_21[_ki]
                for _ki in range(NUM_KEYPOINTS):
                    self.skeleton.keypoints_iccs[_ki] = self.skeleton.iccs.world_to_iccs(
                        keypoints_world_21[_ki])
                    self.skeleton.keypoints_world[_ki] = keypoints_world_21[_ki]
                logger.info(f"[PLY] Pulled {_ply_corrections}/21 joints inside balloon mesh")

        # =====================================================================
        # FIX A9: ENFORCE PELVIS_CENTER = [0,0,0] IN ICCS
        #
        # The ICCS origin IS the pelvis by definition. After all correction
        # steps (depth correction, bone enforcement, velocity clamping) the
        # pelvis ICCS coord must be exactly [0,0,0]. Residual drift (observed
        # as [0, -12.84, +6.5] in frame 1) biases every FK-propagated child
        # joint downstream.
        #
        # Recompute ICCS origin from current pelvis world position, then reset.
        # =====================================================================
        pelvis_world_final = self.skeleton.keypoints_world[KP.PELVIS_CENTER].copy()
        if not np.allclose(pelvis_world_final, 0):
            self.skeleton.iccs.update(pelvis_world_final, self.skeleton.iccs.yaw)
        self.skeleton.keypoints_iccs[KP.PELVIS_CENTER] = np.zeros(3)
        # Recompute all ICCS coords from final world positions
        for _ki in range(NUM_KEYPOINTS):
            self.skeleton.keypoints_iccs[_ki] = self.skeleton.iccs.world_to_iccs(
                self.skeleton.keypoints_world[_ki]
            )
        self.skeleton.keypoints_iccs[KP.PELVIS_CENTER] = np.zeros(3)
        logger.debug(f"[FIX A9] PELVIS ICCS enforced = [0,0,0], world={pelvis_world_final.tolist()}")

        # =====================================================================
        # BUG 9 FIX: Write temporal state back onto the persistent skeleton.
        #
        # The skeleton instance is stored in ClusterStateBank.skeletons[uuid]
        # and lives across frames.  _store_temporal_state records:
        #   - previous_keypoints_iccs  → used by STEP 10b velocity clamping
        #   - previous_keypoints_world → available for downstream consumers
        #   - previous_pelvis_world    → used by STEP 3 pelvis clamping
        #   - frame_count              → how many frames this skeleton has seen
        #   - fitting_errors_history   → per-joint fit quality for next frame
        # =====================================================================
        self.skeleton._store_temporal_state(
            keypoints_world_21=keypoints_world_21,
            fitting_errors=self.fitting_errors
        )

        logger.info(f"[BUG9] Temporal state written: frame_count={self.skeleton.frame_count}, "
                   f"pelvis_world=[{self.skeleton.previous_pelvis_world[0]:.1f}, "
                   f"{self.skeleton.previous_pelvis_world[1]:.1f}, "
                   f"{self.skeleton.previous_pelvis_world[2]:.1f}]")

        return {
            'success': True,
            'fitted_keypoints_world': keypoints_world,                        # 17x3 (legacy)
            'fitted_keypoints_world_21': keypoints_world_21,                  # 21x3 (NEW)
            'fitted_keypoints_iccs': self.skeleton.keypoints_iccs[:17].copy(),# 17x3 (legacy)
            'fitted_keypoints_iccs_21': self.skeleton.keypoints_iccs[:NUM_KEYPOINTS].copy(),  # 21x3 (NEW)
            'cell_assignments': self.fitted_cells.copy(),
            'errors': self.fitting_errors.copy(),
            'avg_error': avg_error,
            'correction': correction_result,
            # BUG 6 FIX: Include ICCS data so visualization.py can pass
            # real world pelvis to movement_index instead of [0,0,0]
            'iccs': {
                'origin': self.skeleton.iccs.origin.copy(),        # world pelvis
                'yaw': self.skeleton.iccs.yaw,                     # facing angle
            },
            'facing_direction': facing_direction,
            'fitting_path': getattr(self, '_fitting_path', 'frontal'),
            'body_yaw_deg': body_yaw_deg,
            # Temporal metadata
            'frame_count': self.skeleton.frame_count,
            'is_calibrated': self.skeleton.is_calibrated,
            'calibration_samples': len(self.skeleton._calibration_samples),
            # Option 4 zone data for zone-colored PLY export
            'placement_method': 'opt4' if opt4_ok else ('vup' if vup_ok else 'cop3d'),
            'voxel_zones': getattr(self, '_voxel_zones', {}),
            'pose_dof_applied': bool(pose_dof) and opt4_ok,
            # Blanket algorithm: extract current spine curve for next frame
            'spine_curve': self._extract_spine_curve(),
        }
    
    # =========================================================================
    # BLANKET: Spine curve extraction for next-frame slicing prediction
    # =========================================================================

    def _extract_spine_curve(self) -> Optional[list]:
        """
        Extract the spine curve from the current fitted skeleton.

        Returns:
            List of 4 world-space (3,) points: [KP19 (pelvis), KP20 (spine_mid),
            KP18 (shoulder_center), KP17 (head_center)], bottom to top.
            Or None if insufficient joints are placed.
        """
        kw = self.skeleton.keypoints_world
        indices = [KP.PELVIS_CENTER, KP.SPINE_MID, KP.SHOULDER_CENTER, KP.HEAD_CENTER]
        curve = []
        for idx in indices:
            pt = kw[idx]
            if np.allclose(pt, 0):
                return None  # incomplete spine — can't use as prediction
            curve.append(pt.tolist())
        return curve

    # =========================================================================
    # NEW METHODS: CoP-3D-driven placement (FitStrategy §4 revised)
    # =========================================================================

    def _get_flesh_radius(self, kp_idx: int) -> float:
        """
        Return the flesh radius (cm) for the given keypoint index.

        The flesh radius is the distance from the bone axis to the cluster
        surface.  Used by Step 8 to offset voxel_under_pixel targets INWARD
        along the 3D surface normal so the skeleton sits inside the cluster
        (shell_as_suit.docx Rule R1).

        Returns 0.0 for keypoints with no flesh mapping (computed KPs 17-20).
        """
        segment = KP_TO_FLESH_SEGMENT.get(kp_idx)
        if segment is None:
            return 0.0
        ratio = FLESH_RADII.get(segment, 0.0)
        H = self.skeleton.height if hasattr(self.skeleton, 'height') else 170.0
        return H * ratio

    def _compute_flesh_inward_offset(self, surface_world: np.ndarray,
                                      cluster_centroid: np.ndarray,
                                      flesh_r: float) -> np.ndarray:
        """
        Compute a 3D offset vector that moves a surface point INWARD toward
        the cluster centroid by flesh_r centimeters.

        The inward direction is the unit vector from the surface voxel toward
        the cluster centroid.  This naturally handles all body regions:
          - Front surface voxels → offset in +Y (depth)
          - Side surface voxels  → offset in ±X (lateral)
          - Top surface voxels   → offset in -Z (downward)
          - Bottom surface voxels → offset in +Z (upward)

        Args:
            surface_world: 3D world position of the surface voxel
            cluster_centroid: 3D world centroid of the entire cluster
            flesh_r: flesh radius in cm

        Returns:
            3D offset vector (add to surface_world to get interior target)
        """
        direction = cluster_centroid - surface_world
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return np.zeros(3)
        # Clamp offset to at most half the distance to centroid
        # (prevents overshooting through the body center)
        effective_r = min(flesh_r, dist * 0.5)
        return (direction / dist) * effective_r

    def _get_cop_world_pos(self, keypoints_3d_mapping: List[Dict], kp_idx: int) -> Optional[np.ndarray]:
        """
        Extract the CoP-derived 3D world position from keypoints_3d_mapping for kp_idx.
        Returns None if unavailable or zero.
        """
        if kp_idx >= len(keypoints_3d_mapping):
            return None
        entry = keypoints_3d_mapping[kp_idx]
        if not isinstance(entry, dict):
            return None
        wp = entry.get('world_pos')
        if wp is None:
            return None
        pos = np.array(wp, dtype=float)
        if np.allclose(pos, 0):
            return None
        return pos

    def _establish_iccs_from_cop3d(self, keypoints_3d_mapping: List[Dict],
                                    cluster_y_bounds: Optional[Tuple[float, float]] = None) -> bool:
        """
        STEP 1 (FitStrategy): Establish ICCS origin and yaw directly from CoP 3D
        hip world positions — replacing the old voxel-candidate ICCS method.

        Uses world_pos for LEFT_HIP (11) and RIGHT_HIP (12), which come from
        extract_3d_from_cop() and carry sub-voxel precision.  Applies the same
        pelvis velocity clamping and yaw smoothing as the old method so temporal
        stability is preserved.

        FIX (Feb 2026): Added cluster_y_bounds grounding.
        After velocity clamping, if the computed pelvis Y lies outside the
        cluster's world-space Y extent, clamp it to the nearest cluster Y
        boundary.  Both hips travel with the pelvis by the same offset so the
        ICCS midpoint invariant is preserved.  This prevents the 18 cm gap
        seen in frame 8 (pelvis at Y=277 cm, cluster at Y=[295, 315] cm).

        Returns True if both hip world_pos values were available; False triggers
        the old fallback path.
        """
        left_hip_w  = self._get_cop_world_pos(keypoints_3d_mapping, KP.LEFT_HIP)
        right_hip_w = self._get_cop_world_pos(keypoints_3d_mapping, KP.RIGHT_HIP)

        if left_hip_w is None or right_hip_w is None:
            logger.warning("[CoP3D] world_pos missing for hips — CoP3D ICCS not possible")
            return False

        raw_pelvis = (left_hip_w + right_hip_w) / 2.0
        hip_vec    = right_hip_w - left_hip_w
        raw_yaw    = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

        # -----------------------------------------------------------------
        # Pelvis velocity clamping (preserved from old method)
        # -----------------------------------------------------------------
        PELVIS_MAX_CM  = 6.0
        YAW_MAX_DEG    = 15.0
        pelvis = raw_pelvis.copy()
        yaw    = raw_yaw

        if self.skeleton.previous_pelvis_world is not None:
            delta = pelvis - self.skeleton.previous_pelvis_world
            dist  = np.linalg.norm(delta)
            if dist > PELVIS_MAX_CM:
                pelvis = (self.skeleton.previous_pelvis_world
                          + (delta / dist) * PELVIS_MAX_CM)
                logger.info(f"[CoP3D] Pelvis clamped: {dist:.1f}cm → {PELVIS_MAX_CM}cm")
            prev_yaw = getattr(self.skeleton, '_previous_yaw', None)
            if prev_yaw is not None:
                dy = ((yaw - prev_yaw + 180) % 360) - 180
                if abs(dy) > YAW_MAX_DEG:
                    yaw = prev_yaw + np.sign(dy) * YAW_MAX_DEG
                    logger.info(f"[CoP3D] Yaw clamped: Δ{dy:.1f}° → ±{YAW_MAX_DEG}°")

        self.skeleton._previous_yaw = yaw

        # -----------------------------------------------------------------
        # CLUSTER Y-GROUNDING FIX (Feb 2026)
        # After velocity clamping the pelvis must still land INSIDE the
        # cluster's world-space Y range.  If extract_3d_from_cop placed
        # hips at zone-Y values that are outside the cluster (e.g. the
        # 18 cm gap seen in 55.txt frame 8: pelvis Y=277 cm vs cluster
        # Y=[295, 315] cm), the ICCS origin floats outside the body
        # voxels and every depth correction anchors off a wrong baseline.
        #
        # Strategy: if pelvis.Y is outside [y_lo, y_hi], shift it to the
        # nearest boundary.  Apply the SAME offset to both hip world
        # positions so midpoint(L_hip, R_hip) == pelvis is maintained
        # and the PELVIS–HIP COLLINEARITY FIX below stays correct.
        # -----------------------------------------------------------------
        if cluster_y_bounds is not None:
            y_lo, y_hi = cluster_y_bounds
            pelvis_y = pelvis[1]
            if pelvis_y < y_lo or pelvis_y > y_hi:
                clamped_y      = float(np.clip(pelvis_y, y_lo, y_hi))
                y_shift        = clamped_y - pelvis_y
                pelvis[1]      = clamped_y
                left_hip_w[1]  = left_hip_w[1]  + y_shift
                right_hip_w[1] = right_hip_w[1] + y_shift
                logger.info(
                    f"[CoP3D] CLUSTER Y-GROUNDING: pelvis.Y {pelvis_y:.1f} → {clamped_y:.1f} cm "
                    f"(cluster Y=[{y_lo:.1f}, {y_hi:.1f}], shift={y_shift:+.1f} cm)"
                )
            else:
                logger.debug(
                    f"[CoP3D] pelvis.Y={pelvis_y:.1f} inside cluster Y=[{y_lo:.1f},{y_hi:.1f}]"
                    f" — no grounding needed"
                )

        # -----------------------------------------------------------------
        # PELVIS–HIP COLLINEARITY FIX
        # The ICCS origin is set to `pelvis` (which may be clamped away from
        # raw_pelvis).  If we naïvely convert the original hip world positions
        # using this shifted origin, their ICCS midpoint is NOT [0,0,0]:
        #
        #   midpoint_iccs = R_inv @ ((L_world + R_world)/2 − clamped_pelvis)
        #                 = R_inv @ (raw_pelvis − clamped_pelvis) ≠ [0,0,0]
        #
        # This makes pelvis, left-hip and right-hip form an impossible triangle
        # (pelvis not on the hip line) — the visual artifact you observed.
        #
        # CORRECT MODEL: The pelvis is anatomically the rigid midpoint of the
        # hip bones.  When we velocity-clamp the pelvis, the hips travel WITH
        # it by the same displacement.  Apply the clamp offset to both hip
        # world positions before converting to ICCS so that:
        #   midpoint(l_iccs, r_iccs) = [0, 0, 0]  ← ICCS origin  [OK]
        # -----------------------------------------------------------------
        clamp_offset = pelvis - raw_pelvis          # zero when no clamping occurred
        left_hip_w  = left_hip_w  + clamp_offset    # hips ride with pelvis
        right_hip_w = right_hip_w + clamp_offset

        self.skeleton.iccs.update(pelvis, yaw)
        self.skeleton.keypoints_iccs[KP.PELVIS_CENTER] = np.zeros(3)  # by construction

        # Place hips in ICCS — keep CoP Y (depth) and Z (height),
        # enforce locked hip width in X (±hip_width/2).
        # After the clamp_offset shift above, midpoint(l_iccs, r_iccs) = [0,0,0] [OK]
        l_iccs = self.skeleton.iccs.world_to_iccs(left_hip_w)
        r_iccs = self.skeleton.iccs.world_to_iccs(right_hip_w)
        half_hip = self.skeleton.bone_lengths.get('hip_width', 31.67) / 2.0
        l_iccs[0] = -half_hip
        r_iccs[0] = +half_hip
        self.skeleton.keypoints_iccs[KP.LEFT_HIP]  = l_iccs
        self.skeleton.keypoints_iccs[KP.RIGHT_HIP] = r_iccs
        self.skeleton.keypoints_world[KP.LEFT_HIP]  = self.skeleton.iccs.iccs_to_world(l_iccs)
        self.skeleton.keypoints_world[KP.RIGHT_HIP] = self.skeleton.iccs.iccs_to_world(r_iccs)

        # Verify invariant (should always pass after the fix above)
        midpoint_iccs = (l_iccs + r_iccs) / 2.0
        midpoint_err  = np.linalg.norm(midpoint_iccs)
        if midpoint_err > 0.5:
            logger.warning(f"[CoP3D] PELVIS invariant drift={midpoint_err:.2f}cm — "
                           f"pelvis≠midpoint(hips). clamp_offset={clamp_offset.tolist()}")
        else:
            logger.debug(f"[CoP3D] PELVIS invariant OK: midpoint_err={midpoint_err:.3f}cm")

        logger.info(f"[CoP3D] ICCS established: origin={pelvis.tolist()}, yaw={yaw:.1f}°, "
                    f"hip_width={half_hip*2:.1f}cm, clamp_offset={np.linalg.norm(clamp_offset):.1f}cm")
        return True

    def _sanitize_limb_dofs_world(self, kpw):
        """
        Correct splayed legs / jutting arms on the final fitted skeleton.

        Works in ICCS (Z = vertical up, Y = depth/facing) where the limb rest
        directions are simple: thigh and upper-arm hang straight down [0,0,-1],
        the shin/forearm are clamped relative to their parent segment.  The
        unreliable ICCS-Y depth component is damped (harder in profile views),
        and each limb's tilt off its reference is clamped.  Bone lengths are
        preserved — only directions are corrected, at the existing length.

        Operates on the REAL fitter output so it actually reaches the rendered
        fitted_keypoints_world_21 (the earlier edit was on the unused
        update_from_detection path and never ran).
        """
        iccs = self.skeleton.iccs
        kp = np.zeros((NUM_KEYPOINTS, 3))
        valid = np.zeros(NUM_KEYPOINTS, dtype=bool)
        for i in range(NUM_KEYPOINTS):
            if not np.allclose(kpw[i], 0):
                kp[i] = iccs.world_to_iccs(kpw[i])
                valid[i] = True

        depth_damp = _limb_depth_damping_for_view(kp, valid)
        out = kp.copy()

        # Arms: shoulder->elbow (vs down), elbow->wrist (vs upper arm)
        for shoulder, elbow, wrist in [
            (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST),
            (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST)
        ]:
            if valid[shoulder] and valid[elbow]:
                ua_len = np.linalg.norm(out[elbow] - out[shoulder])
                d = _limb_dir_sanitize(out[elbow] - out[shoulder],
                                       depth_damp, ARM_FIT_MAX_TILT_FROM_DOWN_DEG)
                out[elbow] = out[shoulder] + d * ua_len
            if valid[elbow] and valid[wrist]:
                fa_len = np.linalg.norm(out[wrist] - out[elbow])
                upper = out[elbow] - out[shoulder]
                d = _limb_dir_sanitize(out[wrist] - out[elbow], depth_damp,
                                       ARM_FIT_MAX_TILT_FROM_DOWN_DEG,
                                       reference_vec=upper)
                out[wrist] = out[elbow] + d * fa_len

        # Legs: hip->knee (vs down), knee->ankle (vs thigh)
        for hip, knee, ankle in [
            (KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE),
            (KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE)
        ]:
            if valid[hip] and valid[knee]:
                th_len = np.linalg.norm(out[knee] - out[hip])
                d = _limb_dir_sanitize(out[knee] - out[hip],
                                       depth_damp, LEG_FIT_MAX_TILT_FROM_DOWN_DEG)
                out[knee] = out[hip] + d * th_len
            if valid[knee] and valid[ankle]:
                sh_len = np.linalg.norm(out[ankle] - out[knee])
                thigh = out[knee] - out[hip]
                d = _limb_dir_sanitize(out[ankle] - out[knee], depth_damp,
                                       LEG_FIT_MAX_TILT_FROM_DOWN_DEG,
                                       reference_vec=thigh)
                out[ankle] = out[knee] + d * sh_len

        result = kpw.copy()
        for i in range(NUM_KEYPOINTS):
            if valid[i]:
                result[i] = iccs.iccs_to_world(out[i])
        return result

    def _place_all_keypoints_from_cop3d(self, keypoints_3d_mapping: List[Dict]) -> int:
        """
        STEP 2 (FitStrategy): Place all 17 COCO keypoints directly from CoP 3D
        world_pos — no voxel snapping, no IK, no candidate hunting.

        Hips (11, 12) and PELVIS_CENTER (19) are already set by
        _establish_iccs_from_cop3d(), so they are skipped here.

        Shoulder width is enforced in ICCS-X just as hip width is — CoP depth (Y)
        and height (Z) are preserved untouched.

        Returns the count of keypoints successfully placed.
        """
        SKIP = {KP.LEFT_HIP, KP.RIGHT_HIP, KP.PELVIS_CENTER,
                # Face keypoints (0-4) are part of the RIGID HEAD TRAPEZOID.
                # They must NOT be placed individually from CoP — doing so
                # would override the rigid body constraint with noisy per-joint
                # positions from MiDaS 2.5D surface snapping.
                # STEP 4 (_fit_face_keypoints_cop3d) uses CoP ear positions
                # to determine HEAD_CENTER + orientation, then rotates the
                # rigid template to place all face keypoints consistently.
                KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR}
        placed = 0

        for kp_idx in range(17):
            if kp_idx in SKIP:
                continue
            world_pos = self._get_cop_world_pos(keypoints_3d_mapping, kp_idx)
            if world_pos is None:
                continue
            iccs_pos = self.skeleton.iccs.world_to_iccs(world_pos)

            # Do NOT override shoulder X with the template ratio.
            # CoP 3D already ran bone-length enforcement; replacing ICCS-X with
            # ±half_shoulder forces the template shoulder width onto every frame
            # regardless of the actual person, producing the "gorilla" appearance
            # (shoulders extending far outside the silhouette).
            # Hips are enforced in _establish_iccs_from_cop3d because they define
            # the ICCS yaw and must be symmetric around X=0. Shoulders are limb
            # roots, not the coordinate system anchor — trust CoP here.

            self.skeleton.keypoints_iccs[kp_idx]  = iccs_pos
            self.skeleton.keypoints_world[kp_idx] = world_pos
            placed += 1

        # Log actual shoulder width from placed positions for diagnostics
        l_sh = self.skeleton.keypoints_iccs[KP.LEFT_SHOULDER]
        r_sh = self.skeleton.keypoints_iccs[KP.RIGHT_SHOULDER]
        if not np.allclose(l_sh, 0) and not np.allclose(r_sh, 0):
            actual_sw = abs(r_sh[0] - l_sh[0])
        else:
            actual_sw = 0.0
        logger.info(f"[CoP3D] Placed {placed} keypoints from world_pos — "
                    f"actual shoulder_width={actual_sw:.1f}cm (template={self.skeleton.bone_lengths.get('shoulder_width',0):.1f}cm)")
        return placed

    # =========================================================================
    # PLY BALLOON MESH CONTAINMENT (Level 3 guidance — shell_as_suit R6)
    # =========================================================================

    def _enforce_ply_containment(self, ply_mesh, keypoints_world_21: np.ndarray) -> int:
        """
        Pull any joint outside the PLY balloon mesh to the mesh interior.

        The balloon.ply is built from this frame's depth point cloud and is
        the 3D ground-truth envelope of the person's body volume. All 21
        joint world positions must satisfy signed_distance < 0 (inside mesh).

        For joints outside:
          1. Find nearest point on mesh surface.
          2. Pull joint to surface - flesh_radius along inward mesh normal.
          3. Iterate up to 3 times if still outside.

        keypoints_world_21: ndarray (21, 3) — modified IN PLACE.
        Returns: count of joints that were corrected.
        """
        try:
            import trimesh as _trimesh_mod
            import numpy as _np
        except ImportError:
            return 0

        corrected = 0

        try:
            # Build a trimesh from the input mesh object.
            # ply_mesh may already be a trimesh.Trimesh (from build_frame_balloon)
            # or an open3d.geometry.TriangleMesh (from file load).
            if hasattr(ply_mesh, 'vertices') and hasattr(ply_mesh, 'faces'):
                # Already a trimesh.Trimesh
                tm = ply_mesh
            elif hasattr(ply_mesh, 'vertices') and hasattr(ply_mesh, 'triangles'):
                # open3d TriangleMesh → convert
                verts = np.asarray(ply_mesh.vertices)
                tris  = np.asarray(ply_mesh.triangles)
                tm = _trimesh_mod.Trimesh(vertices=verts, faces=tris, process=False)
            else:
                return 0

            # Ensure consistent winding for reliable signed-distance
            if not tm.is_watertight:
                tm.fix_normals()

            for kp_idx in range(NUM_KEYPOINTS):
                pt = keypoints_world_21[kp_idx].copy()
                if np.allclose(pt, 0):
                    continue

                for _iter in range(3):
                    # Check containment
                    pt_arr = pt.reshape(1, 3)
                    inside = tm.contains(pt_arr)[0]
                    if inside:
                        break

                    # Not inside — find nearest surface point
                    closest_pts, dists, face_ids = _trimesh_mod.proximity.closest_point(tm, pt_arr)
                    closest = closest_pts[0]
                    dist    = dists[0]

                    if dist < 0.01:   # already on surface
                        break

                    # Inward normal: direction from surface toward mesh interior
                    face_normal = tm.face_normals[face_ids[0]]   # outward normal
                    inward      = -face_normal / (np.linalg.norm(face_normal) + 1e-10)

                    # Flesh radius for this joint
                    flesh_r = self._get_flesh_radius(kp_idx) if hasattr(self, '_get_flesh_radius') else 2.0
                    flesh_r = max(flesh_r, 1.5)   # minimum 1.5 cm so we're clearly inside

                    # Move to surface, then step inward by flesh_radius
                    pt = closest + inward * flesh_r

                    if _iter == 0:
                        corrected += 1

                keypoints_world_21[kp_idx] = pt

        except Exception as e:
            logger.debug(f"[PLY] _enforce_ply_containment error: {e}")

        return corrected

    # =========================================================================
    # POISSON PLY DEPTH CORRECTION (FitStrategy v3 Step 4-6)
    # =========================================================================

    def _apply_poisson_depth_correction(self,
                                         poisson_ply_path: str,
                                         facing_direction: str = 'away_from_camera') -> int:
        """
        FitStrategy v3 Step 4: Correct joint Y (depth) using the per-frame
        Poisson half-shell PLY.

        For each non-face joint (KP 5-16):
          1. Find nearest Poisson vertex by XZ only.
          2. Read surface_Y from that vertex.
          3. Set joint_Y = surface_Y + flesh_radius (push bone behind skin).
          4. X and Z are unchanged.
          5. Sanity clamp: never move Y more than 30cm from current position.

        Face keypoints (0-4) are excluded — handled by rigid head block.
        Spine/extended joints (17-20) are excluded — derived from COCO joints.

        Args:
            poisson_ply_path: path to frames/frame_NNN/poisson.ply
            facing_direction: 'away_from_camera' or 'toward_camera'

        Returns count of joints corrected.
        """
        import os
        if not poisson_ply_path or not os.path.exists(poisson_ply_path):
            logger.debug(f"[PoissonDepth] PLY not found at {poisson_ply_path} — skipping")
            return 0

        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(poisson_ply_path, enable_post_processing=False)
            verts_ply = np.asarray(mesh.vertices)
            if len(verts_ply) == 0:
                return 0

            # Reverse PLY→world transform:
            #   PLY_X=-world_X, PLY_Y=world_Z, PLY_Z=world_Y
            # So: world_X=-PLY_X, world_Y=PLY_Z, world_Z=PLY_Y
            verts_world = np.zeros_like(verts_ply)
            verts_world[:, 0] = -verts_ply[:, 0]   # world_X
            verts_world[:, 1] =  verts_ply[:, 2]   # world_Y (depth)
            verts_world[:, 2] =  verts_ply[:, 1]   # world_Z (height)

        except Exception as e:
            logger.warning(f"[PoissonDepth] Failed to load PLY: {e}")
            return 0

        EXCLUDE = {KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR,
                   KP.HEAD_CENTER, KP.SHOULDER_CENTER, KP.PELVIS_CENTER, KP.SPINE_MID}
        corrected = 0

        for kp_idx in range(17):
            if kp_idx in EXCLUDE:
                continue

            world_pos = self.skeleton.keypoints_world[kp_idx].copy()
            if np.allclose(world_pos, 0):
                continue

            # Find nearest PLY vertex by XZ only
            xz_dists = np.sqrt((verts_world[:, 0] - world_pos[0])**2 +
                               (verts_world[:, 2] - world_pos[2])**2)
            nearest_idx = int(np.argmin(xz_dists))

            if xz_dists[nearest_idx] > 25.0:
                # No close XZ match — joint likely outside PLY extent, skip
                logger.debug(f"[PoissonDepth] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                             f"nearest XZ dist={xz_dists[nearest_idx]:.1f}cm > 25cm, skipping")
                continue

            # Use MEDIAN Y of all PLY vertices within 5cm XZ radius.
            # Single-vertex lookup is noise-sensitive on 2.5D MiDaS shells
            # where L and R vertices at similar XZ can have 20cm Y spread.
            _xz_radius = 5.0
            _nearby_mask = xz_dists <= _xz_radius
            if np.count_nonzero(_nearby_mask) >= 3:
                surface_y = float(np.median(verts_world[_nearby_mask, 1]))
            else:
                surface_y = verts_world[nearest_idx, 1]
            flesh_r   = self._get_flesh_radius(kp_idx)
            new_y     = surface_y + flesh_r
            old_y     = world_pos[1]

            # Sanity clamp: never move more than 30cm in depth
            if abs(new_y - old_y) > 30.0:
                logger.debug(f"[PoissonDepth] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                             f"Y move {old_y:.1f}→{new_y:.1f}cm exceeds 30cm clamp, skipping")
                continue

            corrected_world = world_pos.copy()
            corrected_world[1] = new_y
            corrected_iccs = self.skeleton.iccs.world_to_iccs(corrected_world)
            self.skeleton.keypoints_world[kp_idx] = corrected_world
            self.skeleton.keypoints_iccs[kp_idx]  = corrected_iccs
            corrected += 1

            logger.debug(f"[PoissonDepth] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                         f"Y {old_y:.1f}→{new_y:.1f}cm "
                         f"(surface={surface_y:.1f} + flesh={flesh_r:.1f}cm)")

        logger.info(f"[PoissonDepth] Corrected Y depth for {corrected} joints from Poisson PLY")
        return corrected

    def _refine_limbs_from_2d_rays(self,
                                    keypoints_2d_mapping: List[Dict],
                                    keypoints_3d_mapping: List[Dict],
                                    camera_params: Dict) -> int:
        """
        STEP 3 (FitStrategy): For distal limb joints (elbows, wrists, knees, ankles),
        replace the CoP 3D Z/X position with one derived from the 2D pixel→ray
        intersection at the correct bone length from the parent joint.

        Pixel→ray unprojection is depth-invariant: the angular direction from
        shoulder pixel to elbow pixel encodes the true arm direction regardless
        of depth noise.  We compute the child joint world position as the
        intersection of this ray with the sphere of radius bone_length centred
        on the parent.  This is more reliable than the raw CoP position for
        distal joints, which suffer from compounded bone-chain depth noise.

        Only runs when camera_params is provided.  Each refinement is accepted
        only when: (a) parent world_pos is valid, (b) child 2D pixel is valid,
        (c) the ray intersects the bone-length sphere (disc ≥ 0).

        Returns count of joints refined.
        """
        camera_pos    = np.array(camera_params.get('camera_position', [-47.0,28.0,-20.0]), dtype=float)
        camera_target = np.array(camera_params.get('camera_target', [-25.1,123.8,-28.3]),    dtype=float)
        focal_length  = camera_params.get('focal_length', 27.5)
        panel_w       = camera_params.get('panel_width', 480)
        panel_h       = camera_params.get('panel_height', 864)
        # FOCAL_SCALE FIX: use FOV-based formula matching opencv_integration.project_3d_to_2d.
        # Old focal_length*10=275 vs correct (864/2)/tan(33deg)=665 => 2.4x mismatch => 67cm errors.
        _fov = camera_params.get('field_of_view')
        if _fov is not None and panel_h > 0:
            import math; focal_scale = (panel_h / 2.0) / math.tan(math.radians(_fov / 2.0))
        else:
            focal_scale = focal_length * 10.0  # legacy fallback

        # Build camera basis (same as mmpose_integration.py)
        fwd = camera_target - camera_pos
        fwd_len = np.linalg.norm(fwd)
        fwd = fwd / fwd_len if fwd_len > 1e-6 else np.array([0., 1., 0.])
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            world_up = np.array([0., 1., 0.])
            right = np.cross(fwd, world_up)
            right_len = np.linalg.norm(right)
        right = right / right_len
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)

        def _pixel_to_ray(u, v):
            """Return unit ray direction from camera through pixel (u, v)."""
            dx = (u - panel_w / 2.0) / focal_scale
            dy = (panel_h / 2.0 - v) / focal_scale
            ray = fwd + right * dx + up * dy
            ray_len = np.linalg.norm(ray)
            return ray / ray_len if ray_len > 1e-6 else fwd.copy()

        def _ray_sphere_intersect(ray_dir, parent_world, bone_len):
            """
            Find child world position on the ray such that
            ||child - parent||  = bone_len.

            Solves: t² - 2t·(P-C)·d + ||P-C||² - L² = 0
            where C = camera_pos, d = ray_dir, P = parent_world, L = bone_len.
            Returns the intersection with t > 0, or None if no real solution.
            """
            v   = camera_pos - parent_world
            b   = np.dot(v, ray_dir)
            c   = np.dot(v, v) - bone_len ** 2
            disc = b * b - c
            if disc < 0:
                return None                     # ray misses sphere
            # Defect 3c FIX: choose NEAR-SIDE root first (t = -b - sqrt).
            # Far-side root placed joint behind/through torso — wrong for elbows/knees.
            t = -b - np.sqrt(disc)             # near side (closest intersection)
            if t <= 0:
                t = -b + np.sqrt(disc)         # try far side if near is behind camera
            if t <= 0:
                return None
            return camera_pos + t * ray_dir

        # Limb pairs per RO_NOUS_21Joint_Pipeline.docx §2:
        # KP 7,8 (elbows), KP 9,10 (wrists), KP 13,14 (knees) → 2D ray only.
        # KP 15,16 (ankles) = end-effectors → CoP 3D only, NO 2D ray.
        LIMB_CHAINS = [
            # Arms (proximal → distal)
            (KP.LEFT_SHOULDER,  KP.LEFT_ELBOW,   'upper_arm_l'),
            (KP.LEFT_ELBOW,     KP.LEFT_WRIST,   'forearm_l'),
            (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW,  'upper_arm_r'),
            (KP.RIGHT_ELBOW,    KP.RIGHT_WRIST,  'forearm_r'),
            # Legs — knees only; ankles are end-effectors driven by CoP 3D
            (KP.LEFT_HIP,       KP.LEFT_KNEE,    'thigh_l'),
            (KP.RIGHT_HIP,      KP.RIGHT_KNEE,   'thigh_r'),
        ]

        refined = 0
        for parent_kp, child_kp, bone_key in LIMB_CHAINS:
            # 1. Parent world position must be known
            parent_world = self.skeleton.keypoints_world[parent_kp]
            if parent_world is None or np.allclose(parent_world, 0):
                continue

            # 2. Child must have a valid 2D pixel
            if child_kp >= len(keypoints_2d_mapping):
                continue
            kp2d_entry = keypoints_2d_mapping[child_kp]
            if not isinstance(kp2d_entry, dict):
                continue
            # Defect 3a FIX: use middle_panel_pixel (overlay space, matches camera_params
            # projection matrix), NOT left_panel_pixel (raw camera frame).
            # Three-level fallback matching visualization.py's corrected_2d_mapping:
            #   Level 1: middle_panel_pixel (best — flat-transform from Step 6c)
            #   Level 2: middle_panel_pixel_from_2d (mmpose reverse-transform)
            #   Level 3: left_panel_pixel (last resort, known to produce ~10cm error)
            pixel = (kp2d_entry.get('middle_panel_pixel')
                     or kp2d_entry.get('middle_panel_pixel_from_2d')
                     or kp2d_entry.get('left_panel_pixel'))
            if pixel is None:
                continue
            u, v = float(pixel[0]), float(pixel[1])
            if u <= 0 and v <= 0:
                continue

            # FACING-AWAY LEFT/RIGHT MIRROR FIX
            #
            # When the person faces AWAY from the camera (back view), their
            # anatomical LEFT side appears on the camera's RIGHT in the image
            # and their anatomical RIGHT appears on the camera's LEFT.
            # MMPose labels keypoints anatomically (left_knee = person's left
            # knee), but the pixel it reports is correctly on image-right for
            # a left-limb joint.  When we convert pixel→ray, a pixel at
            # image-right produces a ray pointing to +camera_right, which maps
            # to positive ICCS_X.  But person's left should be NEGATIVE ICCS_X.
            # Result: both knees get positive ICCS_X → legs cross.
            #
            # Fix: for away-facing, mirror the X-pixel across the panel
            # centre before computing the ray.  This gives the anatomically
            # correct ray direction (toward -X for left joints).
            _facing_dir = getattr(self, '_facing_direction', 'toward_camera')
            _LEFT_KPS  = {KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST,
                          KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE}
            _RIGHT_KPS = {KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST,
                          KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE}
            if _facing_is_away(_facing_dir) and child_kp in (_LEFT_KPS | _RIGHT_KPS):
                u = panel_w - u   # mirror pixel X across panel centre

            # 3. Bone length
            bone_len = self.skeleton.bone_lengths.get(bone_key)
            if bone_len is None or bone_len <= 0:
                continue

            # 4. Intersect ray with bone-length sphere
            ray = _pixel_to_ray(u, v)
            child_world = _ray_sphere_intersect(ray, parent_world, bone_len)
            if child_world is None:
                continue

            # 5. Sanity: child must be within 2.5× bone_length of parent
            actual_dist = np.linalg.norm(child_world - parent_world)
            if actual_dist > bone_len * 2.5 or actual_dist < bone_len * 0.1:
                continue

            # 6. Accept: update ICCS and world positions
            child_iccs = self.skeleton.iccs.world_to_iccs(child_world)
            self.skeleton.keypoints_iccs[child_kp]  = child_iccs
            self.skeleton.keypoints_world[child_kp] = child_world
            refined += 1

            logger.debug(f"[2DRay] KP{child_kp} ({KEYPOINT_NAMES[child_kp]}): "
                         f"refined via 2D ray, dist={actual_dist:.1f}cm (bone={bone_len:.1f}cm)")

        return refined

    # =========================================================================
    # OPTION 4: Zone assignment — label each cluster voxel with a body zone
    # =========================================================================

    def _assign_voxel_zones(self,
                            voxel_grid,
                            cluster_voxel_indices: Set[Tuple[int, int, int]],
                            mannequin_world_21: np.ndarray,
                            ) -> Dict[Tuple[int, int, int], int]:
        """
        Assign each cluster voxel to the nearest body zone (0-5) using
        the yaw-rotated mannequin as spatial reference.

        Zone assignment uses representative keypoints per zone:
          head(0):      KP 0,17     (nose, head_center)
          torso(1):     KP 18,19,20 (shoulder_center, pelvis_center, spine_mid)
          left_arm(2):  KP 5,7,9    (L_shoulder, L_elbow, L_wrist)
          right_arm(3): KP 6,8,10   (R_shoulder, R_elbow, R_wrist)
          left_leg(4):  KP 11,13,15 (L_hip, L_knee, L_ankle)
          right_leg(5): KP 12,14,16 (R_hip, R_knee, R_ankle)

        Returns dict: voxel_tuple → zone_id
        """
        zone_representatives = {
            0: [0, 17],          # head
            1: [18, 19, 20],     # torso
            2: [5, 7, 9],        # left_arm
            3: [6, 8, 10],       # right_arm
            4: [11, 13, 15],     # left_leg
            5: [12, 14, 16],     # right_leg
        }

        # Build per-zone centroid from mannequin (skip zero-valued keypoints)
        zone_centroids = {}
        for zone_id, kp_list in zone_representatives.items():
            pts = []
            for kp_idx in kp_list:
                if kp_idx < len(mannequin_world_21) and not np.allclose(mannequin_world_21[kp_idx], 0):
                    pts.append(mannequin_world_21[kp_idx])
            if pts:
                zone_centroids[zone_id] = np.mean(pts, axis=0)

        if not zone_centroids:
            logger.warning("[ZONE] No valid zone centroids from mannequin — all voxels unzoned")
            return {}

        # Pre-build array of zone centroids for vectorised distance
        zone_ids_arr = sorted(zone_centroids.keys())
        zone_pts_arr = np.array([zone_centroids[z] for z in zone_ids_arr])  # (N_zones, 3)

        _voxel_size = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
        _bounds_min = np.array(voxel_grid.bounds[0], dtype=float)

        voxel_zones: Dict[Tuple[int, int, int], int] = {}
        for vt in cluster_voxel_indices:
            # Voxel centroid in world space
            vc = _bounds_min + (np.array(vt, dtype=float) + 0.5) * _voxel_size
            dists = np.linalg.norm(zone_pts_arr - vc, axis=1)
            best_idx = int(np.argmin(dists))
            voxel_zones[vt] = zone_ids_arr[best_idx]

        # Log zone distribution
        from collections import Counter
        _counts = Counter(voxel_zones.values())
        _zone_names = {0: 'head', 1: 'torso', 2: 'L_arm', 3: 'R_arm', 4: 'L_leg', 5: 'R_leg'}
        _summary = ", ".join(f"{_zone_names.get(z, '?')}={c}" for z, c in sorted(_counts.items()))
        logger.info(f"[ZONE] Assigned {len(voxel_zones)} voxels: {_summary}")

        return voxel_zones

    # =========================================================================
    # OPTION 4: Ray + Bone-Length Sphere — one joint at a time
    # =========================================================================

    def _fit_via_ray_sphere_chain(self,
                                   voxel_grid,
                                   keypoints_2d_mapping: List[Dict],
                                   keypoints_3d_mapping: List[Dict],
                                   cluster_voxel_indices: Set[Tuple[int, int, int]],
                                   facing_direction: str,
                                   camera_params: Dict,
                                   mannequin_world_21: np.ndarray = None,
                                   mp33_arm_extended: bool = False,
                                   pose_dof: Optional[Dict] = None,
                                   ) -> int:
        """
        OPTION 4 PIPELINE — Place all joints via ray+sphere intersection,
        one joint at a time in strict kinematic chain order.

        Each joint's 3D position satisfies TWO hard constraints:
          1. Lies on the camera ray through its MMPose 2D pixel (reprojection ≤0.5px)
          2. At exact bone_length distance from its already-placed parent

        The intersection of a ray and a sphere gives 0, 1, or 2 solutions.
        The nearest populated voxel disambiguates when 2 solutions exist.

        Returns count of joints successfully placed.
        """
        # ------------------------------------------------------------------
        # Camera basis (same construction as _refine_limbs_from_2d_rays)
        # ------------------------------------------------------------------
        camera_pos    = np.array(camera_params.get('camera_position', [-47.0, 28.0, -20.0]), dtype=float)
        camera_target = np.array(camera_params.get('camera_target', [-25.1, 123.8, -28.3]), dtype=float)
        panel_w       = camera_params.get('panel_width', 480)
        panel_h       = camera_params.get('panel_height', 864)
        _fov = camera_params.get('field_of_view')
        if _fov is not None and panel_h > 0:
            focal_scale = (panel_h / 2.0) / math.tan(math.radians(_fov / 2.0))
        else:
            focal_scale = camera_params.get('focal_length', 27.5) * 10.0

        fwd = camera_target - camera_pos
        fwd_len = np.linalg.norm(fwd)
        fwd = fwd / fwd_len if fwd_len > 1e-6 else np.array([0., 1., 0.])
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            world_up = np.array([0., 1., 0.])
            right = np.cross(fwd, world_up)
            right_len = np.linalg.norm(right)
        right = right / right_len
        up = np.cross(right, fwd)
        up = up / np.linalg.norm(up)

        def pixel_to_ray(u, v):
            """Return unit ray direction from camera through pixel (u, v)."""
            dx = (u - panel_w / 2.0) / focal_scale
            dy = (panel_h / 2.0 - v) / focal_scale
            ray = fwd + right * dx + up * dy
            ray_len = np.linalg.norm(ray)
            return ray / ray_len if ray_len > 1e-6 else fwd.copy()

        def ray_sphere_intersect(ray_dir, sphere_center, radius):
            """
            Intersect ray (camera_pos + t*ray_dir) with sphere.
            Returns (point_near, point_far) or (None, None).
            """
            v   = camera_pos - sphere_center
            b   = np.dot(v, ray_dir)
            c   = np.dot(v, v) - radius ** 2
            disc = b * b - c
            if disc < 0:
                return None, None
            sqrt_disc = np.sqrt(disc)
            t_near = -b - sqrt_disc
            t_far  = -b + sqrt_disc
            p_near = camera_pos + t_near * ray_dir if t_near > 0 else None
            p_far  = camera_pos + t_far  * ray_dir if t_far  > 0 else None
            return p_near, p_far

        def closest_point_on_ray(ray_dir, target):
            """Find point on ray closest to target (for fallback)."""
            v = target - camera_pos
            t = np.dot(v, ray_dir)
            if t <= 0:
                t = 1.0  # at least 1cm in front of camera
            return camera_pos + t * ray_dir

        # ------------------------------------------------------------------
        # Voxel grid helpers
        # ------------------------------------------------------------------
        _voxel_size = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
        _bounds_min = np.array(voxel_grid.bounds[0], dtype=float)

        def voxel_centroid(vt):
            return _bounds_min + (np.array(vt, dtype=float) + 0.5) * _voxel_size

        # Build KD-tree of cluster voxel centroids for fast nearest-voxel lookup
        _voxel_list = list(cluster_voxel_indices)
        if not _voxel_list:
            logger.warning("[OPT4] Empty cluster — cannot fit")
            return 0
        _voxel_centroids = np.array([voxel_centroid(vt) for vt in _voxel_list])

        # Zone assignment (Preliminary Step 2)
        voxel_zones = {}
        if mannequin_world_21 is not None:
            voxel_zones = self._assign_voxel_zones(
                voxel_grid, cluster_voxel_indices, mannequin_world_21)
        # Store on self so fit_to_cluster_shell can include it in the result dict
        self._voxel_zones = voxel_zones

        def find_nearest_voxel(world_pos, zone_filter=None):
            """
            Find the nearest cluster voxel centroid to world_pos.
            If zone_filter is not None, restrict to voxels in that zone.
            Returns (centroid_world, voxel_tuple) or (None, None).
            """
            if zone_filter is not None and voxel_zones:
                mask = np.array([voxel_zones.get(_voxel_list[i], -1) == zone_filter
                                 for i in range(len(_voxel_list))])
                if mask.any():
                    filtered_centroids = _voxel_centroids[mask]
                    filtered_indices = np.where(mask)[0]
                    dists = np.linalg.norm(filtered_centroids - world_pos, axis=1)
                    best = int(np.argmin(dists))
                    return filtered_centroids[best].copy(), _voxel_list[filtered_indices[best]]

            # Unfiltered fallback
            dists = np.linalg.norm(_voxel_centroids - world_pos, axis=1)
            best = int(np.argmin(dists))
            return _voxel_centroids[best].copy(), _voxel_list[best]

        def nudge_to_ray(pos, ray_dir):
            """
            Nudge pos along the ray so its projection matches the ray pixel exactly.
            Returns adjusted position on the ray closest to pos.
            """
            v = pos - camera_pos
            t = np.dot(v, ray_dir) / np.dot(ray_dir, ray_dir)
            if t <= 0:
                t = np.linalg.norm(v)
            return camera_pos + t * ray_dir

        # ------------------------------------------------------------------
        # Get 2D pixel for a keypoint (3-level fallback)
        # ------------------------------------------------------------------
        def get_mmpose_pixel(kp_idx):
            """Return (u, v) for kp_idx or None."""
            if kp_idx >= len(keypoints_2d_mapping):
                return None
            entry = keypoints_2d_mapping[kp_idx]
            if not isinstance(entry, dict):
                return None
            pixel = (entry.get('middle_panel_pixel')
                     or entry.get('middle_panel_pixel_from_2d')
                     or entry.get('left_panel_pixel'))
            if pixel is None:
                return None
            u, v = float(pixel[0]), float(pixel[1])
            if u <= 0 and v <= 0:
                return None
            return (u, v)

        def get_mmpose_confidence(kp_idx):
            """Return confidence for kp_idx or 0."""
            if kp_idx >= len(keypoints_2d_mapping):
                return 0.0
            entry = keypoints_2d_mapping[kp_idx]
            if not isinstance(entry, dict):
                return 0.0
            return float(entry.get('confidence', entry.get('score', 0.0)))

        # ------------------------------------------------------------------
        # NOTE: NO pixel mirroring for away-from-camera.
        # MMPose 2D keypoints are in image-pixel space — they describe
        # where each joint APPEARS in the image, regardless of facing.
        # The camera ray through that pixel is already correct.
        # Mirroring would shift the ray 50+cm laterally (confirmed:
        # dX=54cm at typical depth), placing anchors OUTSIDE the cluster.
        # ------------------------------------------------------------------

        # ==================================================================
        # STAGE 1: Select anchor joint (highest-confidence ankle)
        # ==================================================================
        conf_15 = get_mmpose_confidence(KP.LEFT_ANKLE)
        conf_16 = get_mmpose_confidence(KP.RIGHT_ANKLE)
        if conf_15 >= conf_16:
            anchor_kp = KP.LEFT_ANKLE
        else:
            anchor_kp = KP.RIGHT_ANKLE

        logger.info(f"[OPT4] Anchor: KP{int(anchor_kp)} ({KEYPOINT_NAMES[anchor_kp]}), "
                    f"conf L={conf_15:.2f} R={conf_16:.2f}")

        # ==================================================================
        # STAGE 2+3: Place joints in kinematic chain order
        # ==================================================================
        placed_world = {}   # kp_idx → world position (3,)
        placed_count = 0
        fallback_count = 0

        for child_kp, parent_kp, bone_key in KINEMATIC_CHAIN_ORDER:
            # Skip computed joints — they'll be derived from placed COCO joints
            if child_kp in COMPUTED_KEYPOINTS:
                continue

            # ---- Get child 2D pixel ----
            pixel = get_mmpose_pixel(child_kp)
            if pixel is None:
                logger.debug(f"[OPT4] KP{int(child_kp)} ({KEYPOINT_NAMES[child_kp]}): no 2D pixel — skip")
                continue

            u, v = pixel[0], pixel[1]
            ray_dir = pixel_to_ray(u, v)

            zone_id = KP_TO_ZONE.get(int(child_kp))

            # ---- ANCHOR PLACEMENT (no parent constraint) ----
            if parent_kp is None:
                # Use MiDaS depth from voxel_under_pixel if available
                initial_world = None

                if child_kp < len(keypoints_2d_mapping):
                    entry = keypoints_2d_mapping[child_kp]
                    if isinstance(entry, dict):
                        vup = entry.get('voxel_under_pixel')
                        if vup is not None and len(vup) == 3:
                            vt = (int(vup[0]), int(vup[1]), int(vup[2]))
                            if vt in cluster_voxel_indices:
                                initial_world = voxel_centroid(vt)

                # Fallback: world_pos from CoP
                if initial_world is None and child_kp < len(keypoints_3d_mapping):
                    wp = keypoints_3d_mapping[child_kp].get('world_pos')
                    if wp is not None and len(wp) == 3 and not all(x == 0 for x in wp):
                        initial_world = np.array(wp, dtype=float)

                if initial_world is None:
                    # Last resort: place 150cm along ray (approximate body depth)
                    initial_world = camera_pos + ray_dir * 150.0
                    logger.warning(f"[OPT4] KP{int(child_kp)}: no depth data, using 150cm default")

                # Snap to nearest voxel, then nudge onto ray
                snap_pos, snap_vt = find_nearest_voxel(initial_world, zone_filter=zone_id)
                if snap_pos is not None:
                    # Apply flesh offset inward
                    flesh_r = self._get_flesh_radius(int(child_kp))
                    if flesh_r > 0:
                        # Compute cluster centroid for inward direction
                        _cluster_centroid = np.mean(_voxel_centroids, axis=0)
                        offset = self._compute_flesh_inward_offset(
                            snap_pos, _cluster_centroid, flesh_r)
                        snap_pos = snap_pos + offset

                    final_pos = nudge_to_ray(snap_pos, ray_dir)
                else:
                    final_pos = nudge_to_ray(initial_world, ray_dir)

                placed_world[int(child_kp)] = final_pos
                self.skeleton.keypoints_world[child_kp] = final_pos.copy()
                placed_count += 1

                logger.info(f"[OPT4] ANCHOR KP{int(child_kp)} ({KEYPOINT_NAMES[child_kp]}): "
                           f"[{final_pos[0]:.1f},{final_pos[1]:.1f},{final_pos[2]:.1f}]")
                continue

            # ---- Compute spine chain after both hips placed ----
            if (int(parent_kp) == int(KP.SHOULDER_CENTER)
                    and KP.SHOULDER_CENTER not in placed_world):
                # Derive computed joints from hips → spine → shoulders
                lh = placed_world.get(int(KP.LEFT_HIP))
                rh = placed_world.get(int(KP.RIGHT_HIP))
                ls = placed_world.get(int(KP.LEFT_SHOULDER))
                rs = placed_world.get(int(KP.RIGHT_SHOULDER))

                if lh is not None and rh is not None:
                    pelvis_c = (lh + rh) / 2.0
                    placed_world[int(KP.PELVIS_CENTER)] = pelvis_c
                    self.skeleton.keypoints_world[KP.PELVIS_CENTER] = pelvis_c.copy()

                    # Shoulder center: if both shoulders already placed use them,
                    # else derive from pelvis + torso bone length
                    if ls is not None and rs is not None:
                        shoulder_c = (ls + rs) / 2.0
                    else:
                        torso_len = self.skeleton.bone_lengths.get('torso',
                                        self.skeleton.height * ANTHROPOMETRIC_RATIOS['torso'])
                        shoulder_c = pelvis_c.copy()
                        shoulder_c[2] += torso_len  # straight up
                    placed_world[int(KP.SHOULDER_CENTER)] = shoulder_c
                    self.skeleton.keypoints_world[KP.SHOULDER_CENTER] = shoulder_c.copy()

                    spine_mid = (pelvis_c + shoulder_c) / 2.0
                    placed_world[int(KP.SPINE_MID)] = spine_mid
                    self.skeleton.keypoints_world[KP.SPINE_MID] = spine_mid.copy()

                    # Head center: derive from shoulder_center + neck+head length
                    neck_len = self.skeleton.bone_lengths.get('neck',
                                   self.skeleton.height * ANTHROPOMETRIC_RATIOS['head'] *
                                   ANTHROPOMETRIC_RATIOS['neck_ratio'])
                    head_len = self.skeleton.bone_lengths.get('head',
                                   self.skeleton.height * ANTHROPOMETRIC_RATIOS['head'] *
                                   (1 - ANTHROPOMETRIC_RATIOS['neck_ratio']))
                    head_c = shoulder_c.copy()
                    head_c[2] += neck_len + head_len * 0.5
                    placed_world[int(KP.HEAD_CENTER)] = head_c
                    self.skeleton.keypoints_world[KP.HEAD_CENTER] = head_c.copy()

                    logger.info(f"[OPT4] Computed spine chain: pelvis→spine_mid→shoulder_c→head_c")
                else:
                    logger.warning(f"[OPT4] Cannot compute spine — hips not placed")

            # ---- NON-ANCHOR: Ray + bone-length sphere intersection ----
            parent_world = placed_world.get(int(parent_kp))
            if parent_world is None:
                # Parent wasn't placed — use skeleton's current position
                parent_world = self.skeleton.keypoints_world[parent_kp].copy()
                if np.allclose(parent_world, 0):
                    logger.debug(f"[OPT4] KP{int(child_kp)}: parent KP{int(parent_kp)} not placed — skip")
                    continue

            bone_len = self.skeleton.bone_lengths.get(bone_key, 0)
            if bone_len <= 0:
                logger.debug(f"[OPT4] KP{int(child_kp)}: bone '{bone_key}' length=0 — skip")
                continue

            # For shoulders, bone_len is HALF the shoulder_width (from center to one side)
            if bone_key == 'shoulder_width':
                bone_len = bone_len / 2.0

            # Intersect ray with bone-length sphere centered on parent
            p_near, p_far = ray_sphere_intersect(ray_dir, parent_world, bone_len)

            child_world = None

            if p_near is not None or p_far is not None:
                # Pick the intersection closest to a populated voxel
                candidates = [p for p in (p_near, p_far) if p is not None]
                best_pos = None
                best_dist = float('inf')
                for cand in candidates:
                    vox_pos, _ = find_nearest_voxel(cand, zone_filter=zone_id)
                    if vox_pos is not None:
                        d = np.linalg.norm(cand - vox_pos)
                        if d < best_dist:
                            best_dist = d
                            best_pos = cand

                if best_pos is not None:
                    child_world = best_pos
            else:
                # FALLBACK (Stage 4): ray misses sphere
                # Find closest point on ray to parent
                closest_on_ray = closest_point_on_ray(ray_dir, parent_world)
                dist_to_parent = np.linalg.norm(closest_on_ray - parent_world)

                if dist_to_parent <= bone_len * 1.2:
                    # Relax: place on ray, enforce bone length along ray→parent direction
                    direction = closest_on_ray - parent_world
                    dir_len = np.linalg.norm(direction)
                    if dir_len > 1e-6:
                        child_world = parent_world + (direction / dir_len) * bone_len
                    else:
                        child_world = closest_on_ray
                    logger.debug(f"[OPT4] KP{int(child_kp)}: ray missed sphere, "
                                f"relaxed (dist={dist_to_parent:.1f} vs bone={bone_len:.1f})")
                else:
                    # Irreconcilable — use previous frame position
                    prev = self.skeleton.keypoints_world[child_kp].copy()
                    if not np.allclose(prev, 0):
                        child_world = prev
                        fallback_count += 1
                        logger.warning(f"[OPT4] KP{int(child_kp)} ({KEYPOINT_NAMES[child_kp]}): "
                                      f"FALLBACK to previous frame (dist={dist_to_parent:.1f}cm)")
                    else:
                        # Absolute last resort: place at bone_len along ray from parent
                        child_world = parent_world + ray_dir * bone_len
                        fallback_count += 1

            if child_world is None:
                continue

            # Apply flesh offset inward for surface joints
            flesh_r = self._get_flesh_radius(int(child_kp))
            if flesh_r > 0:
                _cluster_centroid = np.mean(_voxel_centroids, axis=0)
                offset = self._compute_flesh_inward_offset(
                    child_world, _cluster_centroid, flesh_r)
                child_world = child_world + offset

            # Nudge onto ray for exact reprojection
            final_pos = nudge_to_ray(child_world, ray_dir)

            # Re-enforce bone length after nudge (nudge can change distance)
            direction = final_pos - parent_world
            dir_len = np.linalg.norm(direction)
            if dir_len > 1e-6 and abs(dir_len - bone_len) > 0.5:
                final_pos = parent_world + (direction / dir_len) * bone_len

            placed_world[int(child_kp)] = final_pos
            self.skeleton.keypoints_world[child_kp] = final_pos.copy()
            placed_count += 1

            actual_dist = np.linalg.norm(final_pos - parent_world)
            logger.debug(f"[OPT4] KP{int(child_kp)} ({KEYPOINT_NAMES[child_kp]}): "
                        f"dist={actual_dist:.1f}cm (bone={bone_len:.1f}cm)")

        # ==================================================================
        # STAGE 2 COMPLETION: Derive computed joints if not yet done
        # ==================================================================
        lh = placed_world.get(int(KP.LEFT_HIP))
        rh = placed_world.get(int(KP.RIGHT_HIP))
        if lh is not None and rh is not None:
            if int(KP.PELVIS_CENTER) not in placed_world:
                pc = (lh + rh) / 2.0
                placed_world[int(KP.PELVIS_CENTER)] = pc
                self.skeleton.keypoints_world[KP.PELVIS_CENTER] = pc.copy()

            ls = placed_world.get(int(KP.LEFT_SHOULDER))
            rs = placed_world.get(int(KP.RIGHT_SHOULDER))
            if ls is not None and rs is not None:
                sc = (ls + rs) / 2.0
                if int(KP.SHOULDER_CENTER) not in placed_world:
                    placed_world[int(KP.SHOULDER_CENTER)] = sc
                    self.skeleton.keypoints_world[KP.SHOULDER_CENTER] = sc.copy()

                pc = placed_world[int(KP.PELVIS_CENTER)]
                sm = (pc + sc) / 2.0
                placed_world[int(KP.SPINE_MID)] = sm
                self.skeleton.keypoints_world[KP.SPINE_MID] = sm.copy()

                sc_final = placed_world.get(int(KP.SHOULDER_CENTER), sc)
                neck_len = self.skeleton.bone_lengths.get('neck',
                               self.skeleton.height * ANTHROPOMETRIC_RATIOS['head'] *
                               ANTHROPOMETRIC_RATIOS['neck_ratio'])
                head_len = self.skeleton.bone_lengths.get('head',
                               self.skeleton.height * ANTHROPOMETRIC_RATIOS['head'] *
                               (1 - ANTHROPOMETRIC_RATIOS['neck_ratio']))
                hc = sc_final.copy()
                hc[2] += neck_len + head_len * 0.5
                placed_world[int(KP.HEAD_CENTER)] = hc
                self.skeleton.keypoints_world[KP.HEAD_CENTER] = hc.copy()

        # ==================================================================
        # STAGE 5: Bilateral enforcement (hips, shoulders)
        # ==================================================================
        _BILATERAL = [
            (KP.LEFT_HIP, KP.RIGHT_HIP, 'hips'),
            (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, 'shoulders'),
            (KP.LEFT_KNEE, KP.RIGHT_KNEE, 'knees'),
            (KP.LEFT_ANKLE, KP.RIGHT_ANKLE, 'ankles'),
        ]
        bilateral_fixed = 0
        for l_kp, r_kp, label in _BILATERAL:
            lw = self.skeleton.keypoints_world[l_kp]
            rw = self.skeleton.keypoints_world[r_kp]
            if np.allclose(lw, 0) or np.allclose(rw, 0):
                continue
            y_diff = abs(float(lw[1]) - float(rw[1]))
            if y_diff > 3.0:
                avg_y = (float(lw[1]) + float(rw[1])) / 2.0
                self.skeleton.keypoints_world[l_kp][1] = avg_y
                self.skeleton.keypoints_world[r_kp][1] = avg_y
                bilateral_fixed += 1
                logger.debug(f"[OPT4-BILATERAL] {label}: Y-diff={y_diff:.1f}cm → equalized")

        logger.info(f"[OPT4] Placed {placed_count}/17 joints, "
                   f"{fallback_count} fallbacks, "
                   f"{bilateral_fixed} bilateral fixes")

        # ==================================================================
        # STAGE 6: FLEXIBLE MANNEQUIN — apply POSE_DB DoF angles
        #
        # The 57-pose database implies specific joint angles for each
        # matched pose.  Apply them via the DoF system:
        #   1. set_angles() per segment (with anatomical limit clamping)
        #   2. propagate_fk() cascades the kinematic chain
        #   3. keypoints_world updated to the posed positions
        #
        # This is the 1950s stencil: the pose tells the mannequin how
        # to stand, and the shell drapes over the result.
        # ==================================================================
        if pose_dof and isinstance(pose_dof, dict) and placed_count >= 6:
            try:
                _kin = self.skeleton.get_kinematics()
                _segs = self.skeleton.segments
                _dof_applied = 0
                _dof_details = []

                for _seg_name, _ang in pose_dof.items():
                    _seg = _segs.get(_seg_name)
                    if _seg is None or not isinstance(_ang, dict):
                        continue
                    _rx = float(_ang.get('rx', 0.0))
                    _ry = float(_ang.get('ry', 0.0))
                    _rz = float(_ang.get('rz', 0.0))
                    if abs(_rx) < 0.1 and abs(_ry) < 0.1 and abs(_rz) < 0.1:
                        continue
                    _seg.set_angles(_rx, _ry, _rz)
                    _dof_applied += 1
                    _dof_details.append(
                        f"{_seg_name}(rx={_seg.rx:.0f} ry={_seg.ry:.0f})")

                if _dof_applied > 0:
                    # Sync ICCS from world positions set by OPT4
                    # so propagate_fk reads correct parent positions
                    if hasattr(self.skeleton, 'iccs') and self.skeleton.iccs is not None:
                        for _kp_i in range(len(self.skeleton.keypoints_world)):
                            _wpos = self.skeleton.keypoints_world[_kp_i]
                            if not np.allclose(_wpos, 0):
                                try:
                                    self.skeleton.keypoints_iccs[_kp_i] = \
                                        self.skeleton.iccs.world_to_iccs(_wpos)
                                except Exception:
                                    pass

                    # Propagate FK PER CHAIN from shoulders — NOT from
                    # pelvis, which would overwrite legs/spine with
                    # rest-pose (zero-angle) positions.
                    _fk_roots = [
                        int(KP.LEFT_SHOULDER),
                        int(KP.RIGHT_SHOULDER),
                    ]
                    for _fk_root in _fk_roots:
                        try:
                            _kin.propagate_fk(_fk_root)
                        except Exception as _fk_e:
                            logger.debug(f"[POSE-DOF] FK from KP{_fk_root}: {_fk_e}")

                    logger.info(
                        f"[OPT4] [POSE-DOF] Applied {_dof_applied} "
                        f"segment angles + FK from shoulders: "
                        f"{', '.join(_dof_details)}")
            except Exception as _dof_e:
                logger.warning(
                    f"[OPT4] [POSE-DOF] Failed: {_dof_e}")

        return placed_count

    # =========================================================================
    # NEW: Safe 2D angle-driven limb direction refinement (Bug #2 replacement)
    # =========================================================================

    def _refine_limb_directions_from_2d(self,
                                         keypoints_2d_mapping: List[Dict],
                                         camera_params: Dict) -> int:
        """
        BUG #5 FIX — Mecanim FK implementation.

        For each limb bone (parent→child):
          1. Get parent and child 'middle_panel_pixel' (overlay-space, matching
             camera_params — correct key, fixed by Bug V1 in visualization.py).
          2. Compute the 2D screen-space direction (child_px − parent_px).
          3. Unproject to approximate world direction via camera ray difference.
             The difference (child_ray − parent_ray) is depth-invariant: it encodes
             the true angular direction of the limb regardless of CoP depth noise.
          4. Convert world direction → ICCS direction (R_inv @ world_dir).
          5. Compute rotation from segment REST DIRECTION (ICCS) to current ICCS dir.
             This is the angle the joint is actually at in this frame.
          6. Extract Euler (rx, ry, rz) from that rotation (XYZ order, degrees).
          7. Call kinematics.set_joint_angles(parent_kp, rx, ry, rz, clamp=True).
             ROM clamping is applied here — impossible poses cannot enter the pipeline.
          8. Apply FK for this single bone: kinematics._apply_fk_to_child(parent_kp, child_kp).
             This writes the anatomically-correct child position into keypoints_iccs/world.

        Processing is done bone-by-bone in kinematic order (proximal → distal)
        so that the distal bone's FK uses the already-updated proximal position.

        Bones where either pixel is missing are silently skipped — the CoP-3D
        position from Step 2 is preserved for those joints (best available fallback).

        Returns: count of bones whose direction was refined via FK.
        """
        # ── Camera setup ──────────────────────────────────────────────────────
        panel_w = camera_params.get('panel_width')
        panel_h = camera_params.get('panel_height')
        if panel_w is None or panel_h is None:
            logger.debug("[FK] panel_width/panel_height missing from camera_params — skipping")
            return 0

        focal_length = camera_params.get('focal_length', 27.5)
        # FOCAL_SCALE FIX: FOV-based formula matching opencv_integration.project_3d_to_2d
        _fov = camera_params.get('field_of_view')
        if _fov is not None and panel_h is not None and panel_h > 0:
            focal_scale = (panel_h / 2.0) / math.tan(math.radians(_fov / 2.0))
        else:
            focal_scale = focal_length * 10.0  # legacy fallback (2.4x less accurate)
        camera_pos    = np.array(camera_params.get('camera_position', [0, -100, 50]), dtype=float)
        camera_target = np.array(camera_params.get('camera_target', [0, 60, 50]),    dtype=float)

        fwd = camera_target - camera_pos
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-6:
            return 0
        fwd = fwd / fwd_len
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            world_up = np.array([0., 1., 0.])
            right = np.cross(fwd, world_up)
            right_len = np.linalg.norm(right)
            if right_len < 1e-6:
                return 0
        right = right / right_len
        up_cam = np.cross(right, fwd)
        up_cam = up_cam / (np.linalg.norm(up_cam) + 1e-10)

        # ICCS inverse rotation (world → ICCS for direction vectors, no translation)
        R_inv = self.skeleton.iccs.get_inverse_rotation_matrix()

        def _get_px(kp_idx):
            """Return (u, v) from middle_panel_pixel, None if unavailable."""
            if kp_idx >= len(keypoints_2d_mapping):
                return None
            e = keypoints_2d_mapping[kp_idx]
            if not isinstance(e, dict):
                return None
            px = e.get('middle_panel_pixel')
            if px is None:
                return None
            u, v = float(px[0]), float(px[1])
            if u <= 0 and v <= 0:
                return None
            return (u, v)

        def _px_to_ray(u, v):
            """Unit ray from camera through pixel (u, v) in world space."""
            dx = (u - panel_w / 2.0) / focal_scale
            dy = (panel_h / 2.0 - v) / focal_scale
            ray = fwd + right * dx + up_cam * dy
            return ray / (np.linalg.norm(ray) + 1e-10)

        # Process bones in kinematic order (proximal → distal within each chain)
        LIMB_CHAINS = [
            (KP.LEFT_SHOULDER,  KP.LEFT_ELBOW,  'upper_arm_l'),
            (KP.LEFT_ELBOW,     KP.LEFT_WRIST,  'forearm_l'),
            (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, 'upper_arm_r'),
            (KP.RIGHT_ELBOW,    KP.RIGHT_WRIST, 'forearm_r'),
            (KP.LEFT_HIP,       KP.LEFT_KNEE,   'thigh_l'),
            (KP.LEFT_KNEE,      KP.LEFT_ANKLE,  'shin_l'),
            (KP.RIGHT_HIP,      KP.RIGHT_KNEE,  'thigh_r'),
            (KP.RIGHT_KNEE,     KP.RIGHT_ANKLE, 'shin_r'),
        ]

        refined = 0
        for parent_kp, child_kp, bone_key in LIMB_CHAINS:
            # ── 1. Parent must be validly placed ────────────────────────────
            parent_world = self.skeleton.keypoints_world[parent_kp]
            if parent_world is None or np.allclose(parent_world, 0):
                continue

            # ── 2. Get 2D pixels ─────────────────────────────────────────────
            parent_px = _get_px(parent_kp)
            child_px  = _get_px(child_kp)
            if parent_px is None or child_px is None:
                continue

            # ── 3. 2D pixel distance guard (angle unreliable if too close) ──
            du = child_px[0] - parent_px[0]
            dv = child_px[1] - parent_px[1]
            if np.sqrt(du*du + dv*dv) < 2.0:
                continue

            # ── 4. Approximate world limb direction from ray difference ──────
            # depth-invariant: encodes true angular direction of the limb
            parent_ray = _px_to_ray(parent_px[0], parent_px[1])
            child_ray  = _px_to_ray(child_px[0],  child_px[1])
            world_dir  = child_ray - parent_ray
            w_len = np.linalg.norm(world_dir)
            if w_len < 1e-6:
                continue
            world_dir = world_dir / w_len

            # ── 5. Convert world direction → ICCS direction ──────────────────
            # For direction vectors: iccs_dir = R_inv @ world_dir (no translation)
            iccs_dir = R_inv @ world_dir
            iccs_len = np.linalg.norm(iccs_dir)
            if iccs_len < 1e-6:
                continue
            iccs_dir = iccs_dir / iccs_len

            # ── 6. Get segment rest direction (ICCS space) ───────────────────
            rest_dir = self.kinematics._get_rest_direction(parent_kp, child_kp)
            rest_dir = rest_dir / (np.linalg.norm(rest_dir) + 1e-10)

            # ── 7. Compute rotation: rest_dir → iccs_dir ────────────────────
            cross = np.cross(rest_dir, iccs_dir)
            dot   = float(np.dot(rest_dir, iccs_dir))
            cross_len = np.linalg.norm(cross)

            if cross_len < 1e-6:
                if dot > 0:
                    # Parallel — no rotation needed; keep CoP position
                    # Still apply FK using existing angles so child is consistent
                    self.kinematics._apply_fk_to_child(parent_kp, child_kp)
                    refined += 1
                    continue
                else:
                    # Anti-parallel — 180° rotation around any perpendicular axis
                    perp = np.array([1., 0., 0.]) if abs(rest_dir[0]) < 0.9 else np.array([0., 1., 0.])
                    axis = np.cross(rest_dir, perp)
                    axis = axis / (np.linalg.norm(axis) + 1e-10)
                    R_bone = Rotation.from_rotvec(np.pi * axis)
            else:
                angle = np.arctan2(cross_len, dot)
                axis  = cross / cross_len
                R_bone = Rotation.from_rotvec(angle * axis)

            rx, ry, rz = R_bone.as_euler('xyz', degrees=True)

            # ── 8. Set joint angles (ROM clamping applied automatically) ─────
            self.kinematics.set_joint_angles(parent_kp, rx=rx, ry=ry, rz=rz, clamp=True)

            # ── 9. Apply FK for this single bone ────────────────────────────
            # Writes FK-derived child position into keypoints_iccs and keypoints_world.
            # Proximal-first ordering ensures each bone uses the already-updated
            # parent position (e.g. forearm uses FK-placed elbow, not CoP elbow).
            self.kinematics._apply_fk_to_child(parent_kp, child_kp)

            # ── 10. Sanity: FK child must be within 2× bone_len of CoP child ─
            bone_len = self.skeleton.bone_lengths.get(bone_key, 0.0)
            if bone_len > 0:
                cop_world = None
                # Check against original CoP position (before FK overwrite)
                # We compare against the parent + bone_len in the CoP direction
                # as a proxy (cop_world was just overwritten by _apply_fk_to_child)
                new_child_world = self.skeleton.keypoints_world[child_kp]
                parent_updated  = self.skeleton.keypoints_world[parent_kp]
                dist = np.linalg.norm(new_child_world - parent_updated)
                if abs(dist - bone_len) > bone_len * 0.3:
                    logger.debug(f"[FK] KP{child_kp} ({KEYPOINT_NAMES[child_kp]}): "
                                 f"FK dist={dist:.1f}cm vs bone_len={bone_len:.1f}cm — anomaly")

            refined += 1
            logger.debug(f"[FK] KP{child_kp} ({KEYPOINT_NAMES[child_kp]}): "
                         f"angle-driven FK: rx={rx:.1f}° ry={ry:.1f}° rz={rz:.1f}°")

        if refined > 0:
            logger.info(f"[FK] Mecanim FK refined {refined} limb bones from 2D pixel angles")
        return refined

    # =========================================================================
    # NEW: Reprojection error tracking (Bug #3 fix)
    # =========================================================================

    def _compute_reprojection_errors(self,
                                      keypoints_2d_mapping: List[Dict],
                                      camera_params: Dict) -> None:
        """
        BUG #3 FIX: Populate self.fitting_errors by reprojecting each placed
        3D joint back to 2D and measuring pixel distance to MMPose's
        'middle_panel_pixel'. Converts pixel distance to centimetres using
        the projected torso height as scale reference.

        This re-enables three downstream quality gates that were completely
        dead because fitting_errors was always empty in the CoP-3D path:

          1. Calibration guard: avg_fitting_error_cm < 5.0 threshold.
             Previously ALWAYS passed → badly fitted frames corrupted
             bone-length calibration.

          2. Per-joint bad-fit fallback in _apply_velocity_clamping_iccs:
             fit_err > 12.0 → revert to previous frame position.
             Previously NEVER fired → wild positions entered temporal history.

          3. Velocity clamping quality check: same threshold.

        Method:
          For each COCO keypoint (0-16) that has a world position:
            a. Project world → camera → image pixel using camera_params.
            b. Compare to middle_panel_pixel from keypoints_2d_mapping.
            c. Convert pixel distance to cm using torso pixel height ÷ torso cm.
            d. Write result to self.fitting_errors[kp_idx].

        Joints with no middle_panel_pixel get a default error of 0.0
        (neutral — no false positives from missing detections).
        """
        panel_w = camera_params.get('panel_width')
        panel_h = camera_params.get('panel_height')
        if panel_w is None or panel_h is None:
            logger.debug("[ReproErr] panel_width/panel_height missing — skipping")
            return

        focal_length = camera_params.get('focal_length', 27.5)
        # FOCAL_SCALE FIX: FOV-based formula matching opencv_integration.project_3d_to_2d
        _fov = camera_params.get('field_of_view')
        if _fov is not None and panel_h is not None and panel_h > 0:
            focal_scale = (panel_h / 2.0) / math.tan(math.radians(_fov / 2.0))
        else:
            focal_scale = focal_length * 10.0  # legacy fallback (2.4x less accurate)
        camera_pos    = np.array(camera_params.get('camera_position', [0, -100, 50]), dtype=float)
        camera_target = np.array(camera_params.get('camera_target', [0, 60, 50]),    dtype=float)

        fwd = camera_target - camera_pos
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-6:
            return
        fwd = fwd / fwd_len
        world_up = np.array([0., 0., 1.])
        right = np.cross(fwd, world_up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-6:
            world_up = np.array([0., 1., 0.])
            right = np.cross(fwd, world_up)
            right_len = np.linalg.norm(right)
            if right_len < 1e-6:
                return
        right = right / right_len
        up = np.cross(right, fwd)
        up = up / (np.linalg.norm(up) + 1e-10)

        def _world_to_pixel(world_pt):
            """Project a 3D world point to (u, v) pixel. Returns None if behind camera."""
            rel = world_pt - camera_pos
            depth = np.dot(rel, fwd)
            if depth < 1.0:
                return None
            # Perspective divide
            x_cam = np.dot(rel, right)
            y_cam = np.dot(rel, up)
            u = panel_w / 2.0 + (x_cam / depth) * focal_scale
            v = panel_h / 2.0 - (y_cam / depth) * focal_scale
            return (u, v)

        # Compute pixel-to-cm scale factor from torso (most reliable known length)
        torso_cm = self.skeleton.bone_lengths.get('torso', 0.0)
        px_per_cm = None
        if torso_cm > 1.0:
            hip_world = self.skeleton.keypoints_world[KP.LEFT_HIP]
            sh_world  = self.skeleton.keypoints_world[KP.LEFT_SHOULDER]
            if not np.allclose(hip_world, 0) and not np.allclose(sh_world, 0):
                hip_px = _world_to_pixel(hip_world)
                sh_px  = _world_to_pixel(sh_world)
                if hip_px is not None and sh_px is not None:
                    torso_px = np.sqrt((sh_px[0]-hip_px[0])**2 + (sh_px[1]-hip_px[1])**2)
                    if torso_px > 5.0:
                        px_per_cm = torso_px / torso_cm

        if px_per_cm is None or px_per_cm < 0.1:
            # Fallback: rough estimate — 1 cm ≈ 3 pixels for typical frame size
            px_per_cm = 3.0
            logger.debug("[ReproErr] Using fallback px_per_cm=3.0 (torso scale unavailable)")

        errors_populated = 0
        for kp_idx in range(17):
            world_pos = self.skeleton.keypoints_world[kp_idx]
            if world_pos is None or np.allclose(world_pos, 0):
                continue

            # Project placed 3D position to 2D
            proj_px = _world_to_pixel(world_pos)
            if proj_px is None:
                continue

            # Get MMPose's middle_panel_pixel for this keypoint
            if kp_idx >= len(keypoints_2d_mapping):
                continue
            entry = keypoints_2d_mapping[kp_idx]
            if not isinstance(entry, dict):
                continue
            gt_px = entry.get('middle_panel_pixel')
            if gt_px is None:
                continue

            gt_u, gt_v = float(gt_px[0]), float(gt_px[1])
            if gt_u <= 0 and gt_v <= 0:
                continue

            # Pixel distance → cm
            px_dist = np.sqrt((proj_px[0]-gt_u)**2 + (proj_px[1]-gt_v)**2)
            err_cm  = px_dist / px_per_cm

            self.fitting_errors[kp_idx] = err_cm
            errors_populated += 1

            if err_cm > 5.0:
                logger.debug(f"[ReproErr] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                             f"reprojection error={err_cm:.1f}cm "
                             f"(px_dist={px_dist:.1f}, px_per_cm={px_per_cm:.2f})")

        # Compute and log avg error for diagnostics
        if errors_populated > 0:
            avg_err = sum(self.fitting_errors.values()) / errors_populated
            logger.info(f"[ReproErr] Populated {errors_populated} keypoint errors, "
                        f"avg_fitting_error_cm={avg_err:.2f}")
        else:
            logger.warning("[ReproErr] No reprojection errors could be computed "
                           "(no middle_panel_pixel available)")

    # =========================================================================
    # FACE GEOMETRY ENFORCEMENT — three mandatory anatomical constraints
    # =========================================================================

    def _enforce_face_geometry(self, facing_direction: str = 'toward_camera') -> None:
        """
        Place all 5 face keypoints by ROTATING the RIGID head trapezoid.

        The head is a regular isosceles trapezoid with FIXED dimensions
        stored in self.skeleton.rigid_head_template (built once in
        _build_rigid_head_template, never changed).

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        RIGID BODY RULE — head dimensions NEVER change between frames.
        Only HEAD_CENTER's position and the head's 3D orientation change.
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        HEAD_CENTER is the ONLY flexible joint (neck attachment).
        Its position is taken as-is from ICCS (placed by earlier steps).

        HEAD ORIENTATION is determined from ear ICCS positions:
          • ear_axis  = R_EAR − L_EAR, projected to XZ (depth-free)
          • head_up   = world Z orthogonalised against ear_axis
          • face_norm = cross(head_up, ear_axis)  →  +Y (FORWARD)

        The rotation matrix R maps template-local axes [X, Y, Z] to
        ICCS-space axes [ear_axis, face_norm, head_up].

        All 5 face keypoints are then placed as:
            kp[i] = HEAD_CENTER + R @ template_offset[i]

        At most 3 of the 6 head keypoints can touch the MiDaS shell
        surface; the rest are positioned purely by the rigid constraint.

        Args:
            facing_direction: 'toward_camera' or 'away_from_camera'.
                Flips face_norm sign so nose protrudes away from camera
                when person faces away.
        """
        kp = self.skeleton.keypoints_iccs
        template = self.skeleton.rigid_head_template

        # ── HEAD_CENTER must already be placed ──────────────────────────
        head_ctr = kp[KP.HEAD_CENTER].copy()

        # If HEAD_CENTER not yet placed, attempt to set from ears
        if np.allclose(head_ctr, 0):
            l_ear = kp[KP.LEFT_EAR]
            r_ear = kp[KP.RIGHT_EAR]
            l_ok = not np.allclose(l_ear, 0)
            r_ok = not np.allclose(r_ear, 0)
            if l_ok and r_ok:
                head_ctr = (l_ear + r_ear) / 2.0
                kp[KP.HEAD_CENTER] = head_ctr
            elif l_ok or r_ok:
                head_ctr = l_ear.copy() if l_ok else r_ear.copy()
                kp[KP.HEAD_CENTER] = head_ctr
            else:
                logger.debug("[RigidHead] HEAD_CENTER and ears at origin — cannot place face")
                return

        # ── Determine head orientation from ear axis ────────────────────
        l_ear = kp[KP.LEFT_EAR]
        r_ear = kp[KP.RIGHT_EAR]
        l_valid = not np.allclose(l_ear, 0)
        r_valid = not np.allclose(r_ear, 0)

        if l_valid and r_valid:
            # Use XZ components only — Y (depth) is noisy from MiDaS 2.5D
            ear_vec_xz = np.array([
                r_ear[0] - l_ear[0],
                0.0,
                r_ear[2] - l_ear[2]
            ])
            ear_dist_xz = np.linalg.norm(ear_vec_xz)
            if ear_dist_xz < 1.0:
                logger.debug(f"[RigidHead] Ear XZ distance too small ({ear_dist_xz:.2f}cm)")
                ear_axis = np.array([1.0, 0.0, 0.0])
            else:
                ear_axis = ear_vec_xz / ear_dist_xz
        elif l_valid or r_valid:
            # Only one ear — assume head faces forward, ear_axis = +X
            ear_axis = np.array([1.0, 0.0, 0.0])
        else:
            # No ears — use ICCS X as lateral
            ear_axis = np.array([1.0, 0.0, 0.0])

        # ── Build orthonormal head frame ────────────────────────────────
        # Template axes: X = ear_axis, Y = face_norm (forward), Z = head_up
        world_up = np.array([0.0, 0.0, 1.0])

        head_up = world_up - np.dot(world_up, ear_axis) * ear_axis
        head_up_len = np.linalg.norm(head_up)
        if head_up_len < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            head_up = world_up - np.dot(world_up, ear_axis) * ear_axis
            head_up_len = np.linalg.norm(head_up)
        head_up = head_up / (head_up_len + 1e-10)

        # face_normal = cross(head_up, ear_axis)
        # Right-hand rule: cross(+Z, +X) = +Y = FORWARD in ICCS
        face_norm = np.cross(head_up, ear_axis)
        face_norm_len = np.linalg.norm(face_norm)
        if face_norm_len < 1e-6:
            logger.debug("[RigidHead] Degenerate face frame — skipping")
            return
        face_norm = face_norm / face_norm_len

        # Flip for facing away
        if _facing_is_away(facing_direction):
            face_norm = -face_norm

        # ── Rotation matrix: template-local → ICCS ──────────────────────
        # Template local:  X=lateral, Y=forward, Z=up
        # ICCS target:     X=ear_axis, Y=face_norm, Z=head_up
        # R maps [1,0,0]→ear_axis, [0,1,0]→face_norm, [0,0,1]→head_up
        R = np.column_stack([ear_axis, face_norm, head_up])

        # ── Place all face keypoints rigidly ────────────────────────────
        for kp_idx, local_offset in template.items():
            rotated = R @ local_offset
            iccs_pos = head_ctr + rotated
            kp[kp_idx] = iccs_pos
            self.skeleton.keypoints_world[kp_idx] = (
                self.skeleton.iccs.iccs_to_world(iccs_pos)
            )

        # Update HEAD_CENTER world
        self.skeleton.keypoints_world[KP.HEAD_CENTER] = (
            self.skeleton.iccs.iccs_to_world(head_ctr)
        )

        logger.debug(
            f"[RigidHead] Placed: ear_axis={np.round(ear_axis, 2).tolist()}, "
            f"face_norm={np.round(face_norm, 2).tolist()}, "
            f"facing={facing_direction}"
        )

    def _fit_face_keypoints_cop3d(self, keypoints_3d_mapping: List[Dict],
                                   facing_direction: str) -> None:
        """
        STEP 4 (FitStrategy): Determine HEAD_CENTER from CoP ear positions,
        then place all face keypoints via RIGID head template rotation.

        DESIGN RATIONALE
        ────────────────
        The head is a rigid isosceles trapezoid.  Only HEAD_CENTER (neck
        attachment) moves between frames; internal dimensions are LOCKED.

        CoP-derived ear positions provide two pieces of information:
          • HEAD_CENTER position  = midpoint(LEFT_EAR, RIGHT_EAR)
          • Head lateral orientation = ear_axis from XZ positions (Y ignored)

        _enforce_face_geometry() then rotates the rigid template to match
        this orientation and places ALL 6 head keypoints (ears, eyes, nose,
        HEAD_CENTER) from the template.  No face keypoint is ever placed
        independently from CoP — the rigid body constraint guarantees
        anatomically correct relative positions every frame.
        """
        EAR_KPS = [KP.LEFT_EAR, KP.RIGHT_EAR]

        for kp_idx in EAR_KPS:
            world_pos = self._get_cop_world_pos(keypoints_3d_mapping, kp_idx)
            if world_pos is not None:
                iccs_pos = self.skeleton.iccs.world_to_iccs(world_pos)
                self.skeleton.keypoints_iccs[kp_idx]  = iccs_pos
                self.skeleton.keypoints_world[kp_idx] = world_pos
                logger.debug(f"[FaceCoP3D] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                             f"CoP world_pos placed (used for orientation only)")
            else:
                logger.debug(f"[FaceCoP3D] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                             f"no CoP world_pos — keeping existing ICCS position")

        # HEAD_CENTER from ear midpoint (if ears are available)
        l_ear = self.skeleton.keypoints_iccs[KP.LEFT_EAR]
        r_ear = self.skeleton.keypoints_iccs[KP.RIGHT_EAR]
        l_ok = not np.allclose(l_ear, 0)
        r_ok = not np.allclose(r_ear, 0)
        if l_ok and r_ok:
            head_ctr = (l_ear + r_ear) / 2.0
            self.skeleton.keypoints_iccs[KP.HEAD_CENTER] = head_ctr
            self.skeleton.keypoints_world[KP.HEAD_CENTER] = (
                self.skeleton.iccs.iccs_to_world(head_ctr)
            )
        elif l_ok:
            self.skeleton.keypoints_iccs[KP.HEAD_CENTER] = l_ear.copy()
        elif r_ok:
            self.skeleton.keypoints_iccs[KP.HEAD_CENTER] = r_ear.copy()

        # Rotate rigid template → place all face keypoints
        self._enforce_face_geometry(facing_direction)

    def _derive_anatomical_keypoints(self) -> None:
        """
        STEP 5 (FitStrategy): Derive the 4 extra anatomical keypoints (17–20)
        from the already-placed 17 COCO keypoints.

        SHOULDER_CENTER (18) = midpoint of shoulders
        SPINE_MID       (20) = proportional point on the spine between pelvis and shoulder_center
        HEAD_CENTER     (17) = midpoint of ears (or offset from nose if ears unavailable)
        NECK_BASE       (19/17?) = above shoulder_center toward head

        These replace the old approach of using rest-pose values or searching
        voxel candidates for computed keypoints.
        """
        l_sh = self.skeleton.keypoints_iccs[KP.LEFT_SHOULDER]
        r_sh = self.skeleton.keypoints_iccs[KP.RIGHT_SHOULDER]
        l_hip = self.skeleton.keypoints_iccs[KP.LEFT_HIP]
        r_hip = self.skeleton.keypoints_iccs[KP.RIGHT_HIP]

        # SHOULDER_CENTER (18): midpoint between shoulders
        sh_ctr = (l_sh + r_sh) / 2.0
        self.skeleton.keypoints_iccs[KP.SHOULDER_CENTER]  = sh_ctr
        self.skeleton.keypoints_world[KP.SHOULDER_CENTER] = self.skeleton.iccs.iccs_to_world(sh_ctr)

        # SPINE_MID (20): proportional between PELVIS_CENTER and SHOULDER_CENTER.
        # Must use actual pelvis ICCS position, not assume [0,0,0]:
        # spine_mid = pelvis + (shoulder_center - pelvis) * lower_spine_ratio
        lower_spine = self.skeleton.bone_lengths.get('lower_spine', 22.0)
        upper_spine = self.skeleton.bone_lengths.get('upper_spine', 22.0)
        spine_ratio = lower_spine / (lower_spine + upper_spine)  # ≈ 0.5
        pelvis_iccs = self.skeleton.keypoints_iccs[KP.PELVIS_CENTER]
        spine_mid   = pelvis_iccs + (sh_ctr - pelvis_iccs) * spine_ratio
        self.skeleton.keypoints_iccs[KP.SPINE_MID]  = spine_mid
        self.skeleton.keypoints_world[KP.SPINE_MID] = self.skeleton.iccs.iccs_to_world(spine_mid)

        # HEAD_CENTER, eyes, nose — the head is a RIGID isosceles trapezoid.
        # _enforce_face_geometry rotates the stored rigid_head_template
        # based on ear XZ orientation and places all face keypoints as one
        # rigid body.  No dimensions are recomputed — only orientation changes.
        facing = getattr(self, '_facing_direction', 'toward_camera')
        self._enforce_face_geometry(facing_direction=facing)

        logger.debug(f"[Anatomy] SHOULDER_CENTER={sh_ctr.tolist()}, "
                     f"SPINE_MID={spine_mid.tolist()}")

    def _correct_depth_from_shell(self,
                                   voxel_grid,
                                   cluster_voxel_indices: Set[Tuple[int, int, int]]) -> Dict:
        """
        STEP 7 (FitStrategy): Correct the Y-axis (depth in ICCS) of each joint
        using the voxel shell surface — while preserving X (lateral) and Z (height)
        that were placed from CoP 3D or 2D rays.

        For each non-face joint, we find the cluster voxel whose X and Z are
        closest to the joint's current world X and Z, then set the joint's
        world Y to that voxel's Y centroid.  This places the joint on the
        shell surface at the correct depth without disturbing height or lateral
        position.

        Face keypoints are excluded — they were depth-adjusted by
        _fit_face_keypoints_cop3d() already.

        Returns {'corrected_count': int}.
        """
        if not cluster_voxel_indices:
            return {'corrected_count': 0}

        # Build array of cluster centroid world positions
        voxel_size = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
        if hasattr(voxel_grid, 'bounds') and voxel_grid.bounds is not None:
            x_min = voxel_grid.bounds[0][0]
            y_min = voxel_grid.bounds[0][1]
            z_min = voxel_grid.bounds[0][2]
        else:
            x_min, y_min, z_min = -10.0, 24.0, -17.0

        centroids = []
        for vox in cluster_voxel_indices:
            cx = x_min + (vox[0] + 0.5) * voxel_size
            cy = y_min + (vox[1] + 0.5) * voxel_size
            cz = z_min + (vox[2] + 0.5) * voxel_size
            centroids.append([cx, cy, cz])
        centroids = np.array(centroids)

        EXCLUDE = {KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR}
        corrected = 0
        # FIX (Feb 2026): Widened from 8.0 → 20.0 cm.
        # After the adjust_zones_for_facing label-routing fix, CoP world_pos
        # values are correct in X and Z but may still carry residual depth
        # (Y) error up to ~20 cm in the worst frames.  The shell correction
        # closes this gap — but it only fires when the joint's XZ position
        # is within MAX_XZ_DIST_CM of the nearest cluster voxel centroid.
        # The old 8 cm threshold was too tight: joints placed 10-18 cm off
        # in XZ (zone boundary cases) never triggered the correction at all.
        # 20 cm captures all valid joints while the existing 30 cm Y-change
        # sanity guard prevents spine/neck joints pulling to distant voxels.
        MAX_XZ_DIST_CM = 20.0  # only apply if a close-enough XZ match exists

        for kp_idx in range(17):
            if kp_idx in EXCLUDE:
                continue
            world_pos = self.skeleton.keypoints_world[kp_idx]
            if world_pos is None or np.allclose(world_pos, 0):
                continue

            # Find cluster voxel with smallest XZ distance to this joint
            xz_dists = np.sqrt((centroids[:, 0] - world_pos[0])**2 +
                               (centroids[:, 2] - world_pos[2])**2)
            nearest_idx = int(np.argmin(xz_dists))
            if xz_dists[nearest_idx] > MAX_XZ_DIST_CM:
                continue   # no close match — keep CoP depth as-is

            # Set world Y from nearest cluster voxel + flesh radius offset.
            # FIX B3: Previously placed joint ON the surface (no flesh offset),
            # causing Step 8 vs Step 10 fight (15-40cm errors).
            # Now: joint_Y = surface_Y + flesh_radius (bone is inside the skin).
            flesh_r = self._get_flesh_radius(kp_idx)
            new_y = centroids[nearest_idx, 1] + flesh_r
            old_y = world_pos[1]
            if abs(new_y - old_y) > 30.0:
                continue   # sanity: never move more than 30cm in depth

            corrected_world = world_pos.copy()
            corrected_world[1] = new_y
            corrected_iccs = self.skeleton.iccs.world_to_iccs(corrected_world)
            self.skeleton.keypoints_world[kp_idx] = corrected_world
            self.skeleton.keypoints_iccs[kp_idx]  = corrected_iccs
            corrected += 1
            logger.debug(f"[DepthFix] KP{kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                         f"Y {old_y:.1f} → {new_y:.1f}cm (Δ={new_y-old_y:+.1f}cm)")

        logger.info(f"[DepthFix] Corrected depth for {corrected} keypoints from shell")
        return {'corrected_count': corrected}

    # =========================================================================
    # STEP 1: Build hint targets from mappings (legacy fallback)
    # =========================================================================

    def _build_hint_targets(self,
                            voxel_grid,
                            keypoints_2d_mapping: List[Dict],
                            keypoints_3d_mapping: List[Dict],
                            cluster_voxel_indices: Set[Tuple[int, int, int]] = None) -> Dict[int, Dict]:
        """
        Build hint targets from 2D and 3D keypoint mappings.
        
        Priority:
        1. voxel_under_pixel (2D ray-cast) - BUT only if near cluster!
        2. nearest_cluster_voxel_index - guaranteed on surface
        3. grid_index - geometry-derived, least reliable
        
        Returns:
            Dict[kp_idx, {'hint_voxel': tuple, 'hint_world': array, 'source': str}]
        """
        hints = {}
        
        # DIAGNOSTIC: Log cluster voxel range
        if cluster_voxel_indices:
            x_vals = [v[0] for v in cluster_voxel_indices]
            y_vals = [v[1] for v in cluster_voxel_indices]
            z_vals = [v[2] for v in cluster_voxel_indices]
            logger.info(f"[SHELL_FIT] Cluster voxel ranges: "
                       f"X=[{min(x_vals)},{max(x_vals)}], "
                       f"Y=[{min(y_vals)},{max(y_vals)}], "
                       f"Z=[{min(z_vals)},{max(z_vals)}]")
        
        for kp_idx in range(17):
            hint = {'hint_voxel': None, 'hint_world': None, 'source': None}
            
            # Get 2D mapping
            kp_2d = keypoints_2d_mapping[kp_idx] if kp_idx < len(keypoints_2d_mapping) else {}
            kp_3d = keypoints_3d_mapping[kp_idx] if kp_idx < len(keypoints_3d_mapping) else {}
            
            # Priority 1: voxel_under_pixel (2D ray-cast)
            # Validate it's near the cluster to catch coordinate system mismatches
            voxel_under = kp_2d.get('voxel_under_pixel')
            if voxel_under is not None:
                voxel_tuple = tuple(voxel_under)
                is_valid = True
                
                if cluster_voxel_indices:
                    if voxel_tuple not in cluster_voxel_indices:
                        # Check neighborhood (5 voxel radius)
                        is_valid = False
                        for dx in range(-5, 6):
                            if is_valid:
                                break
                            for dy in range(-5, 6):
                                if is_valid:
                                    break
                                for dz in range(-5, 6):
                                    test = (voxel_tuple[0]+dx, voxel_tuple[1]+dy, voxel_tuple[2]+dz)
                                    if test in cluster_voxel_indices:
                                        is_valid = True
                                        break
                
                if is_valid:
                    hint['hint_voxel'] = voxel_tuple
                    hint['hint_world'] = self._voxel_to_world(voxel_grid, hint['hint_voxel'])
                    hint['source'] = 'ray_cast_2d'
                else:
                    logger.debug(f"[SHELL_FIT] KP {kp_idx}: voxel_under_pixel {voxel_tuple} "
                               f"NOT near cluster, falling through")
            
            # Priority 2: nearest_cluster_voxel_index (guaranteed on cluster surface)
            if hint['hint_voxel'] is None and kp_3d.get('nearest_cluster_voxel_index') is not None:
                nearest = kp_3d['nearest_cluster_voxel_index']
                hint['hint_voxel'] = tuple(nearest)
                hint['hint_world'] = self._voxel_to_world(voxel_grid, hint['hint_voxel'])
                hint['source'] = 'cluster_shell'
            
            # Priority 3: grid_index (least reliable)
            if hint['hint_voxel'] is None and kp_3d.get('grid_index') is not None:
                grid_idx = kp_3d['grid_index']
                hint['hint_voxel'] = tuple(grid_idx)
                hint['hint_world'] = self._voxel_to_world(voxel_grid, hint['hint_voxel'])
                hint['source'] = 'geometry_3d'
            
            if hint['hint_voxel'] is not None:
                hints[kp_idx] = hint
                logger.debug(f"[SHELL_FIT] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                           f"hint={hint['hint_voxel']}, source={hint['source']}")
        
        return hints
    
    # =========================================================================
    # STEP 2: Find candidate cells near hints
    # =========================================================================
    
    def _find_candidate_cells(self,
                              voxel_grid,
                              hint_targets: Dict[int, Dict],
                              cluster_voxels: Set[Tuple[int, int, int]]) -> Dict[int, List[Dict]]:
        """
        Find candidate Y-plane cells near each keypoint hint.
        
        For each keypoint:
        1. Get hint position
        2. Search nearby cluster voxels
        3. Rank by distance and surface quality
        
        Returns:
            Dict[kp_idx, List[{'voxel': tuple, 'centroid': array, 'score': float}]]
        """
        candidates = {}
        voxel_size = voxel_grid.resolution
        
        for kp_idx, hint in hint_targets.items():
            hint_voxel = hint['hint_voxel']
            hint_world = hint['hint_world']
            
            if hint_world is None:
                candidates[kp_idx] = []
                continue
            
            # Search radius in voxel units
            search_voxels_xz = int(self.search_radius_xy / voxel_size) + 1
            search_voxels_y = int(self.search_radius_y / voxel_size) + 1
            
            kp_candidates = []
            
            # Search nearby voxels
            for dx in range(-search_voxels_xz, search_voxels_xz + 1):
                for dy in range(-search_voxels_y, search_voxels_y + 1):
                    for dz in range(-search_voxels_xz, search_voxels_xz + 1):
                        test_voxel = (
                            hint_voxel[0] + dx,
                            hint_voxel[1] + dy,
                            hint_voxel[2] + dz
                        )
                        
                        # Must be in cluster
                        if test_voxel not in cluster_voxels:
                            continue
                        
                        # Get cell centroid
                        centroid = self._get_cell_centroid(voxel_grid, test_voxel)
                        if centroid is None:
                            continue
                        
                        # Calculate distance from hint
                        distance = np.linalg.norm(centroid - hint_world)
                        
                        # Score: prefer closer, with bonus for same XZ column as hint
                        xz_match = (dx == 0 and dz == 0)
                        score = 1.0 / (distance + 0.1)
                        if xz_match:
                            score *= 2.0  # Bonus for same column
                        
                        kp_candidates.append({
                            'voxel': test_voxel,
                            'centroid': centroid,
                            'distance': distance,
                            'score': score
                        })
            
            # Sort by score (descending) and keep top candidates
            kp_candidates.sort(key=lambda c: c['score'], reverse=True)
            candidates[kp_idx] = kp_candidates[:self.max_candidates]
            
            if kp_candidates:
                best = kp_candidates[0]
                logger.debug(f"[SHELL_FIT] KP {kp_idx}: {len(kp_candidates)} candidates, "
                           f"best={best['voxel']}, dist={best['distance']:.1f}cm")
        
        return candidates
    
    # =========================================================================
    # STEP 3: Establish ICCS from hip candidates
    # =========================================================================
    
    def _establish_iccs_from_candidates(self,
                                        voxel_grid,
                                        candidate_cells: Dict[int, List[Dict]]) -> bool:
        """
        Establish ICCS (origin and yaw) from hip keypoint candidates.
        Uses LEFT_HIP and RIGHT_HIP candidates to determine:
        - Origin: midpoint between hips
        - Yaw: direction from left hip to right hip
        """
        left_hip_candidates = candidate_cells.get(KP.LEFT_HIP, [])
        right_hip_candidates = candidate_cells.get(KP.RIGHT_HIP, [])

        if not left_hip_candidates or not right_hip_candidates:
            logger.warning("[SHELL_FIT] Cannot establish ICCS: missing hip candidates")
            return False

        # Use best candidates
        left_hip_pos = left_hip_candidates[0]['centroid']
        right_hip_pos = right_hip_candidates[0]['centroid']

        # Raw candidate values (same as original)
        raw_pelvis_center = (left_hip_pos + right_hip_pos) / 2
        hip_vec = right_hip_pos - left_hip_pos
        raw_yaw = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

        # =================================================================
        # BUG 2 FIX: Pelvis velocity clamping
        # MiDaS depth noise causes the pelvis to drift ~35cm across 5
        # frames on a stationary person.  If the skeleton has a previous
        # world pelvis position, clamp the displacement to a physically
        # plausible per-frame maximum.
        #
        # Threshold: 6 cm/frame ≈ fast walking at 12 fps (~2.6 km/h).
        # Direction is preserved; only magnitude is capped.
        # =================================================================
        PELVIS_MAX_DISPLACEMENT_CM = 6.0   # per frame
        YAW_MAX_CHANGE_DEG = 15.0          # per frame

        pelvis_center = raw_pelvis_center.copy()
        yaw = raw_yaw

        if self.skeleton.previous_pelvis_world is not None:
            delta = pelvis_center - self.skeleton.previous_pelvis_world
            dist = np.linalg.norm(delta)
            if dist > PELVIS_MAX_DISPLACEMENT_CM:
                pelvis_center = (self.skeleton.previous_pelvis_world
                                 + (delta / dist) * PELVIS_MAX_DISPLACEMENT_CM)
                logger.info(f"[STEP 3] Pelvis clamped: raw_delta={dist:.1f}cm "
                           f"→ capped at {PELVIS_MAX_DISPLACEMENT_CM}cm")
            # Smooth yaw to prevent heading whiplash
            prev_yaw = getattr(self.skeleton, '_previous_yaw', None)
            if prev_yaw is not None:
                yaw_delta = ((yaw - prev_yaw + 180) % 360) - 180  # signed diff
                if abs(yaw_delta) > YAW_MAX_CHANGE_DEG:
                    yaw = prev_yaw + np.sign(yaw_delta) * YAW_MAX_CHANGE_DEG
                    logger.info(f"[STEP 3] Yaw clamped: delta={yaw_delta:.1f}° "
                               f"→ capped at ±{YAW_MAX_CHANGE_DEG}°")

        # Store yaw for next-frame clamping (on the skeleton, not the fitter)
        self.skeleton._previous_yaw = yaw

        # -----------------------------------------------------------------
        # PELVIS–HIP COLLINEARITY FIX (same as _establish_iccs_from_cop3d)
        # When pelvis is velocity-clamped, shift hip candidate positions by
        # the same offset so they remain symmetric around the new ICCS origin.
        # This preserves: midpoint(l_iccs, r_iccs) = [0,0,0] = PELVIS_CENTER [OK]
        # -----------------------------------------------------------------
        clamp_offset = pelvis_center - raw_pelvis_center   # zero when no clamping
        left_hip_pos  = left_hip_pos  + clamp_offset
        right_hip_pos = right_hip_pos + clamp_offset

        # Update ICCS
        self.skeleton.iccs.update(pelvis_center, yaw)

        # Update pelvis keypoint — by construction, midpoint(hips) = [0,0,0]
        self.skeleton.keypoints_iccs[KP.PELVIS_CENTER] = np.array([0, 0, 0])

        # Store fitted cells for hips
        self.fitted_cells[KP.LEFT_HIP] = left_hip_candidates[0]['voxel']
        self.fitted_cells[KP.RIGHT_HIP] = right_hip_candidates[0]['voxel']

        # Transform hip positions to ICCS
        left_hip_iccs = self.skeleton.iccs.world_to_iccs(left_hip_pos)
        right_hip_iccs = self.skeleton.iccs.world_to_iccs(right_hip_pos)

        self.skeleton.keypoints_iccs[KP.LEFT_HIP] = left_hip_iccs
        self.skeleton.keypoints_iccs[KP.RIGHT_HIP] = right_hip_iccs

        logger.info(
            f"[ICCS] Established: origin=[{pelvis_center[0]:.1f}, {pelvis_center[1]:.1f}, "
            f"{pelvis_center[2]:.1f}], yaw={yaw:.1f}Ã‚Â°"
        )

        # -----------------------------------------------------------------
        # Enforce locked hip width in ICCS (BUG 5 FIX)
        # Candidates give us the Y (depth) and Z (height) but X must be
        # exactly ±hip_width/2 to maintain skeleton proportions.
        # -----------------------------------------------------------------
        half_hip = self.skeleton.bone_lengths['hip_width'] / 2
        l_hip_iccs = self.skeleton.keypoints_iccs[KP.LEFT_HIP]
        r_hip_iccs = self.skeleton.keypoints_iccs[KP.RIGHT_HIP]

        l_hip_iccs[0] = -half_hip   # Left = negative X in ICCS
        r_hip_iccs[0] = +half_hip   # Right = positive X in ICCS

        self.skeleton.keypoints_iccs[KP.LEFT_HIP] = l_hip_iccs
        self.skeleton.keypoints_iccs[KP.RIGHT_HIP] = r_hip_iccs
        self.skeleton.keypoints_world[KP.LEFT_HIP] = self.skeleton.iccs.iccs_to_world(l_hip_iccs)
        self.skeleton.keypoints_world[KP.RIGHT_HIP] = self.skeleton.iccs.iccs_to_world(r_hip_iccs)

        logger.info(
            f"[STEP 3] Hip width enforced: L_x={l_hip_iccs[0]:.1f}, "
            f"R_x={r_hip_iccs[0]:.1f} (locked={half_hip*2:.1f}cm)"
        )
#
# WHY: ICCS X values were whatever the candidate cell provided — logs showed
#      hip widths of 19.38, 21.46, 24.68cm instead of locked 31.67cm.
#      This fix keeps the candidate's Y (depth) and Z (height) but forces
#      X to the anatomically locked value.

        return True
    
    def _place_limb_roots_from_candidates(self,
                                          voxel_grid,
                                          candidate_cells: Dict[int, List[Dict]]):
        """
        STEP 3b: Place limb root joints (shoulders, hips) directly from
        candidate cell centroids BEFORE running limb IK.

        WHY: If shoulders/hips are still at rest-pose positions when
        analytical IK runs, the IK root is wrong and the elbow/knee
        solution is off-cluster.  By placing roots from actual surface
        data first, the 3-joint analytical solver starts from the right
        spot and only needs to solve elbow+wrist or knee+ankle.

        Hips are already placed by _establish_iccs_from_candidates (STEP 3).
        This method places SHOULDERS from candidates, and validates/refines
        hip positions if better candidates exist.
        """
        # -----------------------------------------------------------------
        # SHOULDERS Ã¢â‚¬â€ place from candidates (critical for arm IK)
        # -----------------------------------------------------------------
        for kp_idx in [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER]:
            candidates = candidate_cells.get(kp_idx, [])
            if not candidates:
                logger.debug(f"[STEP 3b] No candidates for {KEYPOINT_NAMES[kp_idx]}, "
                             f"keeping rest-pose position")
                continue

            best = candidates[0]
            target_world = best['centroid']
            target_iccs = self.skeleton.iccs.world_to_iccs(target_world)

            # Validate: shoulder must be above pelvis (positive Z in ICCS)
            if target_iccs[2] < 0:
                logger.warning(f"[STEP 3b] {KEYPOINT_NAMES[kp_idx]} candidate below pelvis "
                               f"(Z={target_iccs[2]:.1f}cm), skipping")
                continue

            # FIX B2: Do NOT override shoulder X with ±shoulder_width/2.
            # The candidate voxel already has a valid XZ position from the
            # cluster surface. Forcing X = ±half_shoulder_width discards that
            # measurement and produces the "gorilla" appearance.
            # Only depth (Y) and height (Z) are kept from the candidate; X is
            # now trusted from the voxel hit, same as _place_all_keypoints_from_cop3d.

            # Place the joint
            self.skeleton.keypoints_iccs[kp_idx] = target_iccs
            self.skeleton.keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(target_iccs)

            # Record fitted cell
            self.fitted_cells[kp_idx] = best['voxel']
            self.fitting_errors[kp_idx] = np.linalg.norm(
                self.skeleton.iccs.iccs_to_world(target_iccs) - target_world
            )

            logger.info(
                f"[STEP 3b] Placed {KEYPOINT_NAMES[kp_idx]} from candidate: "
                f"iccs=[{target_iccs[0]:.1f}, {target_iccs[1]:.1f}, {target_iccs[2]:.1f}], "
                f"error={self.fitting_errors[kp_idx]:.1f}cm"
            )

        # -----------------------------------------------------------------
        # FIX 4: Enforce symmetric shoulder height (Z) AND depth (Y) in ICCS
        # After placing both shoulders from candidates, average their Z and Y
        # to prevent asymmetric positions from MiDaS depth noise.
        # Shoulders in ICCS must be symmetric: same Z (height), same Y (depth),
        # differing only in X (lateral, enforced by shoulder_width lock).
        # -----------------------------------------------------------------
        l_sh_iccs = self.skeleton.keypoints_iccs[KP.LEFT_SHOULDER]
        r_sh_iccs = self.skeleton.keypoints_iccs[KP.RIGHT_SHOULDER]
        if not np.allclose(l_sh_iccs, 0) and not np.allclose(r_sh_iccs, 0):
            # Z symmetry (height)
            avg_z = (l_sh_iccs[2] + r_sh_iccs[2]) / 2
            z_diff = abs(l_sh_iccs[2] - r_sh_iccs[2])
            if z_diff > 2.0:
                l_sh_iccs[2] = avg_z
                r_sh_iccs[2] = avg_z
                logger.info(
                    f"[STEP 3b] Shoulder Z symmetry enforced: "
                    f"diff={z_diff:.1f}cm, avg_z={avg_z:.1f}cm"
                )
            # Y symmetry (depth) — shoulders must be at same depth from body center
            avg_y = (l_sh_iccs[1] + r_sh_iccs[1]) / 2
            y_diff = abs(l_sh_iccs[1] - r_sh_iccs[1])
            if y_diff > 2.0:
                l_sh_iccs[1] = avg_y
                r_sh_iccs[1] = avg_y
                logger.info(
                    f"[STEP 3b] Shoulder Y symmetry enforced: "
                    f"diff={y_diff:.1f}cm, avg_y={avg_y:.1f}cm"
                )
            self.skeleton.keypoints_iccs[KP.LEFT_SHOULDER] = l_sh_iccs
            self.skeleton.keypoints_iccs[KP.RIGHT_SHOULDER] = r_sh_iccs
            self.skeleton.keypoints_world[KP.LEFT_SHOULDER] = self.skeleton.iccs.iccs_to_world(l_sh_iccs)
            self.skeleton.keypoints_world[KP.RIGHT_SHOULDER] = self.skeleton.iccs.iccs_to_world(r_sh_iccs)

        # -----------------------------------------------------------------
        # Also place SHOULDER_CENTER (kp 18) and SPINE_MID (kp 20) so the
        # spine chain is consistent.  These are COMPUTED keypoints, not
        # detected, so derive from the placed shoulders + pelvis.
        # -----------------------------------------------------------------
        l_sh = self.skeleton.keypoints_iccs[KP.LEFT_SHOULDER]
        r_sh = self.skeleton.keypoints_iccs[KP.RIGHT_SHOULDER]
        shoulder_center_iccs = (l_sh + r_sh) / 2
        self.skeleton.keypoints_iccs[KP.SHOULDER_CENTER] = shoulder_center_iccs

        pelvis_iccs = self.skeleton.keypoints_iccs[KP.PELVIS_CENTER]  # [0,0,0]
        # PROPORTIONAL split: spine_mid at lower_spine/(lower_spine+upper_spine)
        # = 55% of the way from pelvis to shoulder_center (NOT 50% midpoint).
        _ls = self.skeleton.bone_lengths.get('lower_spine', 0)
        _us = self.skeleton.bone_lengths.get('upper_spine', 0)
        _spine_ratio = _ls / (_ls + _us) if (_ls + _us) > 0 else 0.55
        spine_dir = shoulder_center_iccs - pelvis_iccs
        spine_mid_iccs = pelvis_iccs + spine_dir * _spine_ratio
        self.skeleton.keypoints_iccs[KP.SPINE_MID] = spine_mid_iccs

        logger.info(
            f"[STEP 3b] SHOULDER_CENTER iccs=[{shoulder_center_iccs[0]:.1f}, "
            f"{shoulder_center_iccs[1]:.1f}, {shoulder_center_iccs[2]:.1f}]"
        )

        # -----------------------------------------------------------------
        # Place HEAD_CENTER (kp 17) from DEFINING KEYPOINTS
        #
        # HEAD_CENTER is DEFINED as midpoint between ears (base of skull).
        # Previous code walked a direction from shoulder_center toward a
        # nose/ear candidate by neck_length — this created a "goose neck"
        # because nose is forward/above head_center, and the chain walked
        # the WRONG direction for 12cm.
        #
        # Priority:
        #   1. midpoint(left_ear, right_ear)  — the definition
        #   2. midpoint(left_eye, right_eye)  — close approximation
        #   3. nose position (with backward offset)
        #   4. chain direction from shoulder_center (last resort)
        # -----------------------------------------------------------------
        sc_iccs = self.skeleton.keypoints_iccs[KP.SHOULDER_CENTER]
        neck_len = self.skeleton.bone_lengths.get('neck', 10.0)
        head_center_iccs = None

        # METHOD 1: midpoint(ears) — the DEFINITION of HEAD_CENTER
        l_ear_cands = candidate_cells.get(KP.LEFT_EAR, [])
        r_ear_cands = candidate_cells.get(KP.RIGHT_EAR, [])
        if l_ear_cands and r_ear_cands:
            l_ear_world = l_ear_cands[0]['centroid']
            r_ear_world = r_ear_cands[0]['centroid']
            hc_world = (l_ear_world + r_ear_world) / 2
            head_center_iccs = self.skeleton.iccs.world_to_iccs(hc_world)
            logger.info(f"[STEP 3b] HEAD_CENTER from midpoint(ears)")

        # METHOD 2: midpoint(eyes) — close to ear midpoint
        if head_center_iccs is None:
            l_eye_cands = candidate_cells.get(KP.LEFT_EYE, [])
            r_eye_cands = candidate_cells.get(KP.RIGHT_EYE, [])
            if l_eye_cands and r_eye_cands:
                l_eye_world = l_eye_cands[0]['centroid']
                r_eye_world = r_eye_cands[0]['centroid']
                hc_world = (l_eye_world + r_eye_world) / 2
                head_center_iccs = self.skeleton.iccs.world_to_iccs(hc_world)
                logger.info(f"[STEP 3b] HEAD_CENTER from midpoint(eyes) [fallback]")

        # METHOD 3: nose with backward offset
        if head_center_iccs is None:
            nose_cands = candidate_cells.get(KP.NOSE, [])
            if nose_cands:
                nose_world = nose_cands[0]['centroid']
                nose_iccs = self.skeleton.iccs.world_to_iccs(nose_world)
                # Nose is forward (Y+) and slightly above head_center
                # Pull back in Y and down in Z
                head_center_iccs = nose_iccs.copy()
                head_center_iccs[1] -= 8.0   # pull backward (Y- in ICCS)
                head_center_iccs[2] -= 3.0   # pull down slightly
                logger.info(f"[STEP 3b] HEAD_CENTER from nose with offset [fallback 2]")

        # METHOD 4: chain propagation (last resort)
        if head_center_iccs is None:
            head_center_iccs = sc_iccs + np.array([0, 0, 1]) * neck_len
            logger.warning(f"[STEP 3b] HEAD_CENTER from chain direction [last resort]")

        self.skeleton.keypoints_iccs[KP.HEAD_CENTER] = head_center_iccs
        self.skeleton.keypoints_world[KP.HEAD_CENTER] = \
            self.skeleton.iccs.iccs_to_world(head_center_iccs)

        logger.info(
            f"[STEP 3b] HEAD_CENTER iccs="
            f"[{head_center_iccs[0]:.1f}, {head_center_iccs[1]:.1f}, {head_center_iccs[2]:.1f}]"
        )

        # -----------------------------------------------------------------
        # HIPS Ã¢â‚¬â€ already placed by STEP 3, but validate world coords match
        # -----------------------------------------------------------------
        for kp_idx in [KP.LEFT_HIP, KP.RIGHT_HIP]:
            iccs_pos = self.skeleton.keypoints_iccs[kp_idx]
            self.skeleton.keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(iccs_pos)

    
    # =========================================================================
    # STEP 4: Fit end-effectors using IK
    # =========================================================================
    
    def _fit_end_effectors(self,
                           voxel_grid,
                           candidate_cells: Dict[int, List[Dict]],
                           facing_direction: str):
        """
        Fit end-effector keypoints (ankles, wrists) using IK.

        End-effectors are fitted first because they define the reach of the limbs.
        """
        # Map end-effector [OK] (chain_name, root_kp) so we can log root status
        end_effectors = [
            (KP.LEFT_ANKLE,  'left_leg',  KP.LEFT_HIP),
            (KP.RIGHT_ANKLE, 'right_leg', KP.RIGHT_HIP),
            (KP.LEFT_WRIST,  'left_arm',  KP.LEFT_SHOULDER),
            (KP.RIGHT_WRIST, 'right_arm', KP.RIGHT_SHOULDER),
        ]

        for kp_idx, chain_name, root_kp in end_effectors:
            candidates = candidate_cells.get(kp_idx, [])
            if not candidates:
                logger.debug(f"[SHELL_FIT] No candidates for {KEYPOINT_NAMES[kp_idx]}")
                continue

            # Use best candidate as IK target
            target_world = candidates[0]['centroid']
            target_iccs = self.skeleton.iccs.world_to_iccs(target_world)

            # Log: confirm root joint position (placed by STEP 3 / 3b)
            root_iccs = self.skeleton.keypoints_iccs[root_kp]
            logger.info(
                f"[IK] {chain_name}: root {KEYPOINT_NAMES[root_kp]} "
                f"iccs=[{root_iccs[0]:.1f},{root_iccs[1]:.1f},{root_iccs[2]:.1f}] [OK] "
                f"target {KEYPOINT_NAMES[kp_idx]} "
                f"iccs=[{target_iccs[0]:.1f},{target_iccs[1]:.1f},{target_iccs[2]:.1f}]"
            )

            # Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€[OK] BUG 1 FIX: use 'analytical' (3-joint sub-chain) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€[OK]
            # analytical IK uses ONLY:
            #   arm:  LEFT_SHOULDER [OK] LEFT_ELBOW [OK] LEFT_WRIST
            #   leg:  LEFT_HIP [OK] LEFT_KNEE [OK] LEFT_ANKLE
            # It does NOT walk back to PELVIS_CENTER.
            success = self.kinematics.solve_ik(
                end_effector_kp=kp_idx,
                target_pos=target_iccs,
                method='analytical',          # was: 'fabrik'
                max_iterations=30,
                tolerance=1.0,
                constrain_angles=True
            )

            if success:
                self.fitted_cells[kp_idx] = candidates[0]['voxel']
                # Calculate actual error
                actual_pos = self.skeleton.keypoints_iccs[kp_idx]
                error = np.linalg.norm(actual_pos - target_iccs)
                self.fitting_errors[kp_idx] = error
                logger.info(
                    f"[IK] {KEYPOINT_NAMES[kp_idx]}: "
                    f"target_iccs=[{target_iccs[0]:.1f}, {target_iccs[1]:.1f}, {target_iccs[2]:.1f}], "
                    f"error={error:.2f}cm"
                )
            else:
                logger.warning(f"[IK] Failed for {KEYPOINT_NAMES[kp_idx]}")
    
    # =========================================================================
    # STEP 5: Fit intermediate joints
    # =========================================================================
    
    def _fit_intermediate_joints(self,
                                 voxel_grid,
                                 candidate_cells: Dict[int, List[Dict]]):
        """
        Fit intermediate joints (knees, elbows, shoulders) to nearby cells.
        
        These are constrained by the end-effector fitting, so we just
        snap to the nearest valid cell within the kinematic solution.
        """
        intermediate_joints = [
            KP.LEFT_KNEE, KP.RIGHT_KNEE,
            KP.LEFT_ELBOW, KP.RIGHT_ELBOW,
            KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER,
        ]
        
        for kp_idx in intermediate_joints:
            candidates = candidate_cells.get(kp_idx, [])
            if not candidates:
                continue
            
            # Get current position from kinematic solution
            current_iccs = self.skeleton.keypoints_iccs[kp_idx]
            current_world = self.skeleton.iccs.iccs_to_world(current_iccs)
            
            # Find closest candidate to current kinematic position
            best_candidate = None
            best_distance = float('inf')
            
            for candidate in candidates:
                dist = np.linalg.norm(candidate['centroid'] - current_world)
                if dist < best_distance:
                    best_distance = dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                self.fitted_cells[kp_idx] = best_candidate['voxel']
                self.fitting_errors[kp_idx] = best_distance
                logger.debug(f"[SHELL_FIT] Snap {KEYPOINT_NAMES[kp_idx]}: dist={best_distance:.2f}cm")
    
    # =========================================================================
    # STEP 6: Fit face keypoints with depth adjustment
    # =========================================================================
    
    def _fit_face_keypoints(self,
                            voxel_grid,
                            candidate_cells: Dict[int, List[Dict]],
                            facing_direction: str):
        """
        Fit face keypoints (fallback voxel-candidate path).

        REVISED: Only places EAR positions from voxel candidates (as the
        primary face-plane anchors), then calls _enforce_face_geometry() to
        derive HEAD_CENTER, eyes, and nose with proper geometric constraints:
          1. HEAD_CENTER = midpoint(ears)
          2. Eyes coplanar with ears (frontal plane)
          3. Nose perpendicular from eye midpoint (mid-sagittal normal)

        The old approach of applying ad-hoc depth offsets per keypoint
        independently produced faces where eyes were not in the ear plane
        and nose was not on the mid-sagittal line.
        """
        logger.info(f"[FACE] Fitting face keypoints from candidates, facing={facing_direction}")

        EAR_KPS = [KP.LEFT_EAR, KP.RIGHT_EAR]

        for kp_idx in EAR_KPS:
            candidates = candidate_cells.get(kp_idx, [])
            if not candidates:
                continue

            target_world = candidates[0]['centroid'].copy()
            target_iccs  = self.skeleton.iccs.world_to_iccs(target_world)
            self.skeleton.keypoints_iccs[kp_idx] = target_iccs

            self.fitted_cells[kp_idx]   = candidates[0]['voxel']
            self.fitting_errors[kp_idx] = 0.0  # placed from candidate directly

        # Derive HEAD_CENTER, eyes, nose with correct geometric constraints
        self._enforce_face_geometry(facing_direction)
    
    # =========================================================================
    # VOXEL-SURFACE PRIMARY PLACEMENT (new Step 0)
    # =========================================================================

    def _place_all_from_voxel_surface(
        self,
        voxel_grid,
        keypoints_2d_mapping: List[Dict],
        keypoints_3d_mapping: List[Dict],
        cluster_voxel_indices: Set[Tuple[int, int, int]],
        facing_direction: str,
    ) -> int:
        """
        Place ALL body joints (KP 5-16) directly from voxel_under_pixel + flesh offset.

        This is the PRIMARY placement method.  Each joint's 2D MMPose pixel was
        ray-cast through the voxel grid during Phase 1 and the hit voxel stored
        in keypoints_2d_mapping[kp_idx]['voxel_under_pixel'].

        Fallback chain per joint:
          voxel_under_pixel (in cluster) → nearest_cluster_voxel_index → world_pos

        Returns count of body joints successfully placed from voxel surface data.
        """
        if voxel_grid is None or not hasattr(voxel_grid, 'bounds') or voxel_grid.bounds is None:
            logger.warning("[VUP] No voxel_grid — cannot place from voxel surface")
            return 0

        _voxel_size = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
        _cluster_sum = np.zeros(3)
        _cluster_n = 0
        for _cv in cluster_voxel_indices:
            _cluster_sum[0] += voxel_grid.bounds[0][0] + (_cv[0] + 0.5) * _voxel_size
            _cluster_sum[1] += voxel_grid.bounds[0][1] + (_cv[1] + 0.5) * _voxel_size
            _cluster_sum[2] += voxel_grid.bounds[0][2] + (_cv[2] + 0.5) * _voxel_size
            _cluster_n += 1
        if _cluster_n == 0:
            logger.warning("[VUP] Empty cluster — cannot compute centroid")
            return 0
        cluster_centroid = _cluster_sum / _cluster_n

        body_joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        placed = 0

        for kp_idx in body_joints:
            surface_world = None
            source = None

            # Priority 1: voxel_under_pixel from 2D ray-cast
            if kp_idx < len(keypoints_2d_mapping):
                vup = keypoints_2d_mapping[kp_idx].get('voxel_under_pixel')
                if vup is not None and len(vup) == 3:
                    voxel_tuple = (int(vup[0]), int(vup[1]), int(vup[2]))
                    if voxel_tuple in cluster_voxel_indices:
                        centroid = self._get_cell_centroid(voxel_grid, voxel_tuple)
                        if centroid is not None:
                            surface_world = centroid
                            source = 'voxel_under_pixel'

            # Priority 2: nearest_cluster_voxel_index
            if surface_world is None and kp_idx < len(keypoints_3d_mapping):
                ncvi = keypoints_3d_mapping[kp_idx].get('nearest_cluster_voxel_index')
                if ncvi is not None and len(ncvi) == 3:
                    voxel_tuple = (int(ncvi[0]), int(ncvi[1]), int(ncvi[2]))
                    if voxel_tuple in cluster_voxel_indices:
                        centroid = self._get_cell_centroid(voxel_grid, voxel_tuple)
                        if centroid is not None:
                            surface_world = centroid
                            source = 'nearest_cluster_voxel'

            # Priority 3: world_pos from CoP (last resort)
            if surface_world is None and kp_idx < len(keypoints_3d_mapping):
                wp = keypoints_3d_mapping[kp_idx].get('world_pos')
                if wp is not None and len(wp) == 3 and not all(v == 0 for v in wp):
                    surface_world = np.array(wp, dtype=float)
                    source = 'cop_world_pos'

            if surface_world is None:
                logger.debug(f"[VUP] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): no surface data")
                continue

            # Apply flesh offset inward toward cluster centroid
            flesh_r = self._get_flesh_radius(kp_idx)
            if flesh_r > 0 and source != 'cop_world_pos':
                offset = self._compute_flesh_inward_offset(
                    surface_world, cluster_centroid, flesh_r)
                joint_world = surface_world.copy() + offset
            else:
                joint_world = surface_world.copy()

            self.skeleton.keypoints_world[kp_idx] = joint_world
            placed += 1

            logger.info(
                f"[VUP] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                f"[{joint_world[0]:.1f},{joint_world[1]:.1f},{joint_world[2]:.1f}] "
                f"flesh_r={flesh_r:.1f}cm ({source})"
            )

        logger.info(f"[VUP] Placed {placed}/12 body joints from voxel surface")
        return placed

    def _solve_two_bone_ik(
        self,
        root_world: np.ndarray,
        end_world: np.ndarray,
        L1: float,
        L2: float,
        hint_mid_world: np.ndarray,
    ) -> np.ndarray:
        """
        Two-bone IK with BOTH endpoints fixed.  Find knee/elbow that satisfies
        bone lengths L1 and L2.

        The locus of valid points is a circle perpendicular to the root-end axis.
        We pick the point closest to hint_mid_world (preserves bend direction).
        """
        axis = end_world - root_world
        base = np.linalg.norm(axis)

        if base < 1e-6:
            d = hint_mid_world - root_world
            dn = np.linalg.norm(d)
            if dn > 1e-6:
                return root_world + (d / dn) * L1
            return root_world + np.array([0, 0, L1])

        axis_n = axis / base

        if base >= L1 + L2 - 0.01:
            t = L1 / (L1 + L2)
            return root_world + axis_n * (base * t)

        if base <= abs(L1 - L2) + 0.01:
            return root_world + axis_n * L1

        d = (L1 * L1 - L2 * L2 + base * base) / (2.0 * base)
        h = np.sqrt(max(L1 * L1 - d * d, 0.0))

        proj = root_world + axis_n * d
        hint_vec = hint_mid_world - proj
        hint_vec = hint_vec - np.dot(hint_vec, axis_n) * axis_n
        hint_len = np.linalg.norm(hint_vec)

        if hint_len > 1e-6:
            perp = hint_vec / hint_len
        else:
            if abs(axis_n[0]) < 0.9:
                perp = np.cross(axis_n, np.array([1, 0, 0]))
            else:
                perp = np.cross(axis_n, np.array([0, 1, 0]))
            perp = perp / np.linalg.norm(perp)

        return proj + perp * h

    def _establish_iccs_from_placed_joints(self, facing_direction: str) -> bool:
        """
        Establish ICCS from joints placed by _place_all_from_voxel_surface().
        Uses placed hips → pelvis center → ICCS origin, hip-to-hip → yaw.
        """
        left_hip_w = self.skeleton.keypoints_world[KP.LEFT_HIP].copy()
        right_hip_w = self.skeleton.keypoints_world[KP.RIGHT_HIP].copy()

        if np.allclose(left_hip_w, 0) or np.allclose(right_hip_w, 0):
            logger.warning("[VUP-ICCS] Hips not placed — cannot establish ICCS")
            return False

        pelvis = (left_hip_w + right_hip_w) / 2.0
        hip_vec = right_hip_w - left_hip_w
        yaw = np.degrees(np.arctan2(hip_vec[1], hip_vec[0]))

        if self.skeleton.previous_pelvis_world is not None:
            prev = self.skeleton.previous_pelvis_world
            delta = np.linalg.norm(pelvis - prev)
            if delta > 15.0:
                direction = (pelvis - prev) / delta
                pelvis = prev + direction * 15.0

        self.skeleton.iccs.update(pelvis, yaw)
        self.skeleton.keypoints_iccs[KP.PELVIS_CENTER] = np.zeros(3)
        self.skeleton.keypoints_world[KP.PELVIS_CENTER] = pelvis.copy()

        for kp_idx in range(17):
            w = self.skeleton.keypoints_world[kp_idx]
            if not np.allclose(w, 0):
                self.skeleton.keypoints_iccs[kp_idx] = self.skeleton.iccs.world_to_iccs(w)

        half_hip = self.skeleton.bone_lengths.get('hip_width', 31.67) / 2.0
        l_iccs = self.skeleton.keypoints_iccs[KP.LEFT_HIP].copy()
        r_iccs = self.skeleton.keypoints_iccs[KP.RIGHT_HIP].copy()
        l_iccs[0] = -half_hip
        r_iccs[0] = +half_hip
        self.skeleton.keypoints_iccs[KP.LEFT_HIP] = l_iccs
        self.skeleton.keypoints_iccs[KP.RIGHT_HIP] = r_iccs
        self.skeleton.keypoints_world[KP.LEFT_HIP] = self.skeleton.iccs.iccs_to_world(l_iccs)
        self.skeleton.keypoints_world[KP.RIGHT_HIP] = self.skeleton.iccs.iccs_to_world(r_iccs)

        logger.info(f"[VUP-ICCS] origin=[{pelvis[0]:.1f},{pelvis[1]:.1f},{pelvis[2]:.1f}], "
                    f"yaw={yaw:.1f}°, hip_width={half_hip*2:.1f}cm")
        return True

    def _enforce_bone_lengths_two_bone_ik(self) -> int:
        """
        Enforce bone lengths using two-bone IK: both ROOT and END stay at
        their voxel_under_pixel positions, only INTERMEDIATE (knee/elbow) moves.
        """
        limb_chains = [
            (KP.LEFT_WRIST,  KP.LEFT_ELBOW,  KP.LEFT_SHOULDER,  'left_arm',  'upper_arm_l', 'forearm_l'),
            (KP.RIGHT_WRIST, KP.RIGHT_ELBOW, KP.RIGHT_SHOULDER, 'right_arm', 'upper_arm_r', 'forearm_r'),
            (KP.LEFT_ANKLE,  KP.LEFT_KNEE,   KP.LEFT_HIP,       'left_leg',  'thigh_l',     'shin_l'),
            (KP.RIGHT_ANKLE, KP.RIGHT_KNEE,  KP.RIGHT_HIP,      'right_leg', 'thigh_r',     'shin_r'),
        ]

        adjusted = 0
        for end_kp, mid_kp, root_kp, chain_name, bone1_name, bone2_name in limb_chains:
            root_w = self.skeleton.keypoints_world[root_kp].copy()
            mid_w = self.skeleton.keypoints_world[mid_kp].copy()
            end_w = self.skeleton.keypoints_world[end_kp].copy()

            if np.allclose(root_w, 0) or np.allclose(end_w, 0):
                continue

            L1 = self.skeleton.bone_lengths.get(bone1_name, 0)
            L2 = self.skeleton.bone_lengths.get(bone2_name, 0)
            if L1 <= 0 or L2 <= 0:
                continue

            hint_mid = mid_w if not np.allclose(mid_w, 0) else (root_w + end_w) / 2.0
            new_mid_w = self._solve_two_bone_ik(root_w, end_w, L1, L2, hint_mid)

            d1 = np.linalg.norm(new_mid_w - root_w)
            d2 = np.linalg.norm(end_w - new_mid_w)

            self.skeleton.keypoints_world[mid_kp] = new_mid_w
            self.skeleton.keypoints_iccs[mid_kp] = self.skeleton.iccs.world_to_iccs(new_mid_w)
            self.skeleton.keypoints_iccs[root_kp] = self.skeleton.iccs.world_to_iccs(root_w)
            self.skeleton.keypoints_iccs[end_kp] = self.skeleton.iccs.world_to_iccs(end_w)

            logger.info(
                f"[VUP-IK] {chain_name}: L1={L1:.1f}cm (actual={d1:.1f}), "
                f"L2={L2:.1f}cm (actual={d2:.1f})")
            adjusted += 1

        return adjusted

    def _compute_extended_joints_from_placed(self, facing_direction: str):
        """
        Compute KP 17-20 from placed body joints.
        KP19 PELVIS_CENTER already set by ICCS establishment.
        KP18 SHOULDER_CENTER = midpoint(KP5, KP6)
        KP20 SPINE_MID = midpoint(KP19, KP18)
        KP17 HEAD_CENTER from ears or offset above shoulders.
        """
        l_sh = self.skeleton.keypoints_world[KP.LEFT_SHOULDER]
        r_sh = self.skeleton.keypoints_world[KP.RIGHT_SHOULDER]
        if not np.allclose(l_sh, 0) and not np.allclose(r_sh, 0):
            sc = (l_sh + r_sh) / 2.0
            self.skeleton.keypoints_world[KP.SHOULDER_CENTER] = sc
            self.skeleton.keypoints_iccs[KP.SHOULDER_CENTER] = self.skeleton.iccs.world_to_iccs(sc)

        pelvis_w = self.skeleton.keypoints_world[KP.PELVIS_CENTER]
        shoulder_c = self.skeleton.keypoints_world[KP.SHOULDER_CENTER]
        if not np.allclose(pelvis_w, 0) and not np.allclose(shoulder_c, 0):
            sm = (pelvis_w + shoulder_c) / 2.0
            self.skeleton.keypoints_world[KP.SPINE_MID] = sm
            self.skeleton.keypoints_iccs[KP.SPINE_MID] = self.skeleton.iccs.world_to_iccs(sm)

        l_ear = self.skeleton.keypoints_world[KP.LEFT_EAR]
        r_ear = self.skeleton.keypoints_world[KP.RIGHT_EAR]
        if not np.allclose(l_ear, 0) and not np.allclose(r_ear, 0):
            hc = (l_ear + r_ear) / 2.0
        elif not np.allclose(shoulder_c, 0):
            neck_len = self.skeleton.bone_lengths.get('neck', 10.0)
            hc = shoulder_c.copy()
            hc[2] += neck_len
        else:
            hc = np.zeros(3)
        if not np.allclose(hc, 0):
            self.skeleton.keypoints_world[KP.HEAD_CENTER] = hc
            self.skeleton.keypoints_iccs[KP.HEAD_CENTER] = self.skeleton.iccs.world_to_iccs(hc)

        self._enforce_face_geometry(facing_direction)

    # =========================================================================
    # STEP 8: Validate and correct skeleton from 2D ray-cast anchors
    # =========================================================================

    def _validate_and_correct_from_2d_anchors(
        self,
        voxel_grid,
        keypoints_2d_mapping: List[Dict],
        cluster_voxel_indices: Set[Tuple[int, int, int]]
    ) -> Dict[str, Any]:
        """
        Validate skeleton joint positions against voxel_under_pixel (2D ray-cast)
        and cell_assignments (shell fitter). Correct joints that deviate beyond
        threshold, then re-solve kinematics to maintain bone-length consistency.

        Uses a 3-phase approach:
          Phase 1 Ã¢â‚¬â€ Detect: compare skeleton position vs ray-cast centroid
          Phase 2 Ã¢â‚¬â€ Correct: move flagged joints to ground-truth centroids
          Phase 3 Ã¢â‚¬â€ Re-solve: FABRIK per limb from corrected roots, then FK

        Args:
            voxel_grid: OccupancyGrid with bounds and cell_metadata
            keypoints_2d_mapping: List of dicts with 'voxel_under_pixel' per keypoint
            cluster_voxel_indices: Set of (x,y,z) voxel tuples in person cluster

        Returns:
            Dict with correction stats:
              - 'corrected_count': int
              - 'corrections': Dict[int, dict] per-joint detail
              - 'avg_error_before': float
              - 'avg_error_after': float
        """
        CORRECTION_THRESHOLD = 5.0  # cm Ã¢â‚¬â€ correct if error exceeds this
        
        logger.info("=" * 60)
        logger.info("[STEP 8] Validate & correct from 2D ray-cast anchors")

        # -----------------------------------------------------------------
        # Guard: need grid bounds to convert voxel index [OK] world centroid
        # -----------------------------------------------------------------
        if voxel_grid is None or not hasattr(voxel_grid, 'bounds') or voxel_grid.bounds is None:
            logger.warning("[STEP 8] No voxel_grid bounds Ã¢â‚¬â€ skipping correction")
            return {'corrected_count': 0, 'corrections': {}}

        # -----------------------------------------------------------------
        # Compute cluster centroid for 3D flesh inward-normal offset
        # (shell_as_suit.docx Rule R1: bone = surface + flesh_r * inward_normal)
        # -----------------------------------------------------------------
        _voxel_size = voxel_grid.resolution if hasattr(voxel_grid, 'resolution') else 2.0
        _x0 = voxel_grid.bounds[0][0]
        _y0 = voxel_grid.bounds[0][1]
        _z0 = voxel_grid.bounds[0][2]
        _cluster_sum = np.zeros(3)
        _cluster_n = 0
        for _cv in cluster_voxel_indices:
            _cluster_sum[0] += _x0 + (_cv[0] + 0.5) * _voxel_size
            _cluster_sum[1] += _y0 + (_cv[1] + 0.5) * _voxel_size
            _cluster_sum[2] += _z0 + (_cv[2] + 0.5) * _voxel_size
            _cluster_n += 1
        cluster_centroid = _cluster_sum / max(_cluster_n, 1) if _cluster_n > 0 else None
        if cluster_centroid is not None:
            logger.info(f"[STEP 8] Cluster centroid: [{cluster_centroid[0]:.1f}, "
                       f"{cluster_centroid[1]:.1f}, {cluster_centroid[2]:.1f}] "
                       f"from {_cluster_n} voxels")

        # -----------------------------------------------------------------
        # PHASE 1 Ã¢â‚¬â€ Detect disagreement
        # Body joints only (skip face 0-4, they have special depth handling)
        # -----------------------------------------------------------------
        # Joints to validate: shoulders(5,6), elbows(7,8), wrists(9,10),
        #                     hips(11,12), knees(13,14), ankles(15,16)
        body_joints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        corrections = {}  # kp_idx -> {target_world, source, error_before}

        for kp_idx in body_joints:
            current_world = self.skeleton.keypoints_world[kp_idx].copy()
            if np.allclose(current_world, 0):
                continue

            target_world = None
            target_source = None

            # Priority 1: voxel_under_pixel from ray-cast (independent 2D anchor)
            if kp_idx < len(keypoints_2d_mapping):
                vup = keypoints_2d_mapping[kp_idx].get('voxel_under_pixel')
                if vup is not None and len(vup) == 3:
                    voxel_tuple = (int(vup[0]), int(vup[1]), int(vup[2]))
                    # Only use if voxel belongs to the person cluster
                    if voxel_tuple in cluster_voxel_indices:
                        centroid = self._get_cell_centroid(voxel_grid, voxel_tuple)
                        if centroid is not None:
                            target_world = centroid
                            target_source = 'voxel_under_pixel'

            # Priority 2: cell_assignments from shell fitter Steps 4-6
            if target_world is None and kp_idx in self.fitted_cells:
                cell = self.fitted_cells[kp_idx]
                voxel_tuple = tuple(cell) if not isinstance(cell, tuple) else cell
                if voxel_tuple in cluster_voxel_indices:
                    centroid = self._get_cell_centroid(voxel_grid, voxel_tuple)
                    if centroid is not None:
                        target_world = centroid
                        target_source = 'cell_assignment'

            if target_world is None:
                continue

            # ---------------------------------------------------------
            # FIX 5: Depth validation for knee/ankle ray-casts
            # If knee/ankle Y exceeds parent hip Y by more than bone
            # length, the ray-cast hit floor/background. Keep X and Z,
            # but replace Y with the hip's Y.
            # ---------------------------------------------------------
            if target_world is not None and target_source == 'voxel_under_pixel':
                parent_hip_kp = None
                if kp_idx in (KP.LEFT_KNEE, KP.LEFT_ANKLE):
                    parent_hip_kp = KP.LEFT_HIP
                elif kp_idx in (KP.RIGHT_KNEE, KP.RIGHT_ANKLE):
                    parent_hip_kp = KP.RIGHT_HIP

                if parent_hip_kp is not None:
                    hip_world = self.skeleton.keypoints_world[parent_hip_kp]
                    if not np.allclose(hip_world, 0):
                        side = 'l' if kp_idx in (KP.LEFT_KNEE, KP.LEFT_ANKLE) else 'r'
                        thigh_len = self.skeleton.bone_lengths.get(f'thigh_{side}', 41.0)
                        shin_len = self.skeleton.bone_lengths.get(f'shin_{side}', 41.0)
                        max_depth = thigh_len + shin_len

                        depth_diff = abs(target_world[1] - hip_world[1])
                        if depth_diff > max_depth:
                            logger.warning(
                                f"[STEP 8] DEPTH REJECT {KEYPOINT_NAMES[kp_idx]}: "
                                f"depth_diff={depth_diff:.1f}cm > max={max_depth:.1f}cm, "
                                f"replacing Y with hip Y={hip_world[1]:.1f}"
                            )
                            target_world = target_world.copy()
                            target_world[1] = hip_world[1]

            # ---------------------------------------------------------
            # FLESH-SPACER 3D INWARD OFFSET (shell_as_suit Rule R1)
            #
            # The voxel_under_pixel centroid is on the SURFACE of the
            # cluster.  The bone joint lives INSIDE the body by the
            # flesh radius.  Offset the target INWARD along the 3D
            # vector from surface → cluster centroid.  This handles
            # ALL body surfaces: front (Y), sides (X), top/bottom (Z).
            # ---------------------------------------------------------
            surface_target = target_world  # save for logging
            flesh_r = self._get_flesh_radius(kp_idx)
            if flesh_r > 0 and cluster_centroid is not None:
                offset = self._compute_flesh_inward_offset(
                    target_world, cluster_centroid, flesh_r)
                target_world = target_world.copy() + offset

            # Measure error
            error = np.linalg.norm(current_world - target_world)

            if error > CORRECTION_THRESHOLD:
                corrections[kp_idx] = {
                    'target_world': target_world,
                    'source': target_source,
                    'error_before': error
                }
                logger.info(
                    f"[STEP 8] FLAGGED {KEYPOINT_NAMES[kp_idx]}: "
                    f"error={error:.1f}cm from {target_source} "
                    f"flesh_r={flesh_r:.1f}cm "
                    f"current=[{current_world[0]:.1f},{current_world[1]:.1f},{current_world[2]:.1f}] "
                    f"surface=[{surface_target[0]:.1f},{surface_target[1]:.1f},{surface_target[2]:.1f}] "
                    f"target=[{target_world[0]:.1f},{target_world[1]:.1f},{target_world[2]:.1f}]"
                )

        if not corrections:
            logger.info("[STEP 8] All body joints within threshold Ã¢â‚¬â€ no correction needed")
            return {'corrected_count': 0, 'corrections': {}}

        avg_error_before = sum(c['error_before'] for c in corrections.values()) / len(corrections)
        logger.info(f"[STEP 8] {len(corrections)} joints flagged, avg_error_before={avg_error_before:.1f}cm")

        # -----------------------------------------------------------------
        # PHASE 2 Ã¢â‚¬â€ Move flagged joints to ground-truth centroids
        #
        # Order matters: move chain ROOTS first (shoulders, hips) so that
        # when FABRIK re-solves the limb, it starts from the correct root.
        # -----------------------------------------------------------------
        # Process order: hips [OK] shoulders [OK] elbows/knees [OK] wrists/ankles
        correction_order = [11, 12, 5, 6, 7, 8, 13, 14, 9, 10, 15, 16]

        for kp_idx in correction_order:
            if kp_idx not in corrections:
                continue
            corr = corrections[kp_idx]
            target_world = corr['target_world']
            target_iccs = self.skeleton.iccs.world_to_iccs(target_world)

            self.skeleton.keypoints_iccs[kp_idx] = target_iccs
            self.skeleton.keypoints_world[kp_idx] = target_world

            logger.info(
                f"[STEP 8] MOVED {KEYPOINT_NAMES[kp_idx]} [OK] "
                f"[{target_world[0]:.1f},{target_world[1]:.1f},{target_world[2]:.1f}] "
                f"({corr['source']})"
            )

        # -----------------------------------------------------------------
        # PHASE 3 Ã¢â‚¬â€ Enforce bone lengths on corrected limb sub-chains
        #
        # DO NOT use FABRIK/solve_ik here Ã¢â‚¬â€ it builds chains all the way
        # back to PELVIS_CENTER and its forward pass overwrites corrections.
        # Instead, directly enforce bone lengths on the 3-joint sub-chain:
        #   root(shoulder/hip) [OK] mid(elbow/knee) [OK] end(wrist/ankle)
        # Positions were set in Phase 2; here we just fix bone lengths.
        # -----------------------------------------------------------------
        limb_chains = [
            # (end_effector_kp, intermediate_kp, root_kp, chain_name)
            (KP.LEFT_WRIST,  KP.LEFT_ELBOW,  KP.LEFT_SHOULDER,  'left_arm'),
            (KP.RIGHT_WRIST, KP.RIGHT_ELBOW, KP.RIGHT_SHOULDER, 'right_arm'),
            (KP.LEFT_ANKLE,  KP.LEFT_KNEE,   KP.LEFT_HIP,       'left_leg'),
            (KP.RIGHT_ANKLE, KP.RIGHT_KNEE,  KP.RIGHT_HIP,      'right_leg'),
        ]

        for end_kp, mid_kp, root_kp, chain_name in limb_chains:
            # Check if ANY joint in this chain was corrected
            chain_joints = [root_kp, mid_kp, end_kp]
            chain_corrected = any(kp in corrections for kp in chain_joints)
            if not chain_corrected:
                continue

            # Current ICCS positions (already corrected in Phase 2)
            root_iccs = self.skeleton.keypoints_iccs[root_kp].copy()
            mid_iccs = self.skeleton.keypoints_iccs[mid_kp].copy()
            end_iccs = self.skeleton.keypoints_iccs[end_kp].copy()

            # Get bone lengths
            seg_upper = self.kinematics._get_segment_from_endpoints(root_kp, mid_kp)
            seg_lower = self.kinematics._get_segment_from_endpoints(mid_kp, end_kp)
            L1 = seg_upper.bone_length if seg_upper and seg_upper.bone_length > 0 else np.linalg.norm(mid_iccs - root_iccs)
            L2 = seg_lower.bone_length if seg_lower and seg_lower.bone_length > 0 else np.linalg.norm(end_iccs - mid_iccs)

            # Enforce bone length ONLY when actual > bone (compress, never stretch).
            # If corrected joints are closer than bone length, the surface
            # centroids are our best data Ã¢â‚¬â€ stretching would undo the correction.
            dir_root_mid = mid_iccs - root_iccs
            d1 = np.linalg.norm(dir_root_mid)
            if d1 > 0.001:
                mid_constrained = root_iccs + (dir_root_mid / d1) * L1
            else:
                mid_constrained = mid_iccs  # Degenerate — keep as-is

            dir_mid_end = end_iccs - mid_constrained
            d2 = np.linalg.norm(dir_mid_end)
            if d2 > 0.001:
                end_constrained = mid_constrained + (dir_mid_end / d2) * L2
            else:
                end_constrained = end_iccs  # Degenerate — keep as-is

            # Apply constrained positions
            self.skeleton.keypoints_iccs[mid_kp] = mid_constrained
            self.skeleton.keypoints_iccs[end_kp] = end_constrained

            # Back-calculate joint angles for this sub-chain only
            self.kinematics._apply_fabrik_positions(
                [root_kp, mid_kp, end_kp],
                [root_iccs, mid_constrained, end_constrained],
                [L1, L2],
                True  # constrain_angles
            )

            logger.info(
                f"[STEP 8] BONE-FIX {chain_name}: "
                f"L1={L1:.1f}cm (d={d1:.1f} {'compressed' if d1 > L1 else 'stretched' if d1 < L1 else 'exact'}), "
                f"L2={L2:.1f}cm (d={d2:.1f} {'compressed' if d2 > L2 else 'stretched' if d2 < L2 else 'exact'})"
            )

        # -----------------------------------------------------------------
        # Update world coords directly from corrected ICCS positions.
        # DO NOT call propagate_fk(PELVIS_CENTER) here Ã¢â‚¬â€ FK walks the
        # full kinematic tree and recomputes child positions from spine
        # angles, which OVERWRITES the corrections we just applied.
        # Instead, trust the ICCS positions set in Phase 2 + Phase 3.
        # -----------------------------------------------------------------
        for kp_idx in range(17):
            self.skeleton.keypoints_world[kp_idx] = self.skeleton.iccs.iccs_to_world(
                self.skeleton.keypoints_iccs[kp_idx]
            )

        # -----------------------------------------------------------------
        # Recalculate fitting errors after correction
        # -----------------------------------------------------------------
        corrected_count = 0
        for kp_idx, corr in corrections.items():
            new_world = self.skeleton.keypoints_world[kp_idx]
            new_error = np.linalg.norm(new_world - corr['target_world'])
            old_error = corr['error_before']
            corr['error_after'] = new_error

            # Update the shell fitter's error dict
            self.fitting_errors[kp_idx] = new_error

            if new_error < old_error:
                corrected_count += 1
                logger.info(
                    f"[STEP 8] {KEYPOINT_NAMES[kp_idx]}: "
                    f"{old_error:.1f}cm [FAIL] {new_error:.1f}cm  IMPROVED"
                )
            else:
                logger.warning(
                    f"[STEP 8] {KEYPOINT_NAMES[kp_idx]}: "
                    f"{old_error:.1f}cm [FAIL] {new_error:.1f}cm  NOT IMPROVED"
                )

        avg_error_after = (
            sum(c['error_after'] for c in corrections.values()) / len(corrections)
            if corrections else 0
        )

        logger.info(
            f"[STEP 8] Complete: {corrected_count}/{len(corrections)} improved, "
            f"avg_error {avg_error_before:.1f}cm [FAIL] {avg_error_after:.1f}cm"
        )
        logger.info("=" * 60)

        return {
            'corrected_count': corrected_count,
            'corrections': {
                kp_idx: {
                    'source': c['source'],
                    'error_before': round(c['error_before'], 2),
                    'error_after': round(c['error_after'], 2),
                }
                for kp_idx, c in corrections.items()
            },
            'avg_error_before': round(avg_error_before, 2),
            'avg_error_after': round(avg_error_after, 2),
        }
   
    # =========================================================================
    # STEP 9: Cross-body bilateral depth sanity
    # =========================================================================

    def _enforce_bilateral_depth_sanity(self):
        """
        STEP 9: Enforce bilateral symmetry in Y-depth (ICCS).
        
        In ICCS, Y is the forward/backward axis.  Left and right limb
        pairs (hips, knees, ankles, elbows, wrists) should have
        SIMILAR Y-values.  Large L-R Y-differences indicate one side
        was snapped to a wrong voxel (often a chair or background).
        
        Strategy: for each bilateral pair, if |L_Y - R_Y| exceeds a
        threshold, average their Y values (pull the outlier inward).
        Also enforce that child joints don't deviate in Y more than
        their bone length from the parent.
        """
        kp = self.skeleton.keypoints_iccs
        bl = self.skeleton.bone_lengths

        def _bilateral_y_clamp(l_kp, r_kp, max_y_diff, label):
            """Clamp bilateral Y-difference by averaging toward center."""
            l_pos = kp[l_kp]
            r_pos = kp[r_kp]
            if np.allclose(l_pos, 0) or np.allclose(r_pos, 0):
                return
            y_diff = abs(l_pos[1] - r_pos[1])
            if y_diff > max_y_diff:
                avg_y = (l_pos[1] + r_pos[1]) / 2
                old_l_y, old_r_y = l_pos[1], r_pos[1]
                kp[l_kp][1] = avg_y
                kp[r_kp][1] = avg_y
                logger.info(
                    f"[STEP 9] BILATERAL {label}: Y-diff={y_diff:.1f}cm > {max_y_diff}cm, "
                    f"L_Y={old_l_y:.1f}→{avg_y:.1f}, R_Y={old_r_y:.1f}→{avg_y:.1f}"
                )

        def _chain_y_clamp(parent_kp, child_kp, bone_name, label):
            """Ensure child Y doesn't deviate from parent more than bone length."""
            p = kp[parent_kp]
            c = kp[child_kp]
            if np.allclose(p, 0) or np.allclose(c, 0):
                return
            bone_len = bl.get(bone_name, 50.0)
            y_dev = abs(c[1] - p[1])
            if y_dev > bone_len:
                # Clamp child Y to be within bone_length of parent Y
                clamped_y = p[1] + np.sign(c[1] - p[1]) * bone_len
                old_y = c[1]
                kp[child_kp][1] = clamped_y
                logger.info(
                    f"[STEP 9] CHAIN {label}: Y-deviation={y_dev:.1f}cm > "
                    f"bone={bone_len:.1f}cm, Y={old_y:.1f}→{clamped_y:.1f}"
                )

        # ---- Bilateral Y symmetry (ICCS) ----
        # Hips: very constrained (structurally symmetric)
        _bilateral_y_clamp(KP.LEFT_HIP, KP.RIGHT_HIP, 3.0, "hips")
        # Shoulders: already enforced in STEP 3b, but re-check
        _bilateral_y_clamp(KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, 3.0, "shoulders")
        # Knees: moderate allowance (stride)
        _bilateral_y_clamp(KP.LEFT_KNEE, KP.RIGHT_KNEE, 15.0, "knees")
        # Ankles: moderate allowance (stride)
        _bilateral_y_clamp(KP.LEFT_ANKLE, KP.RIGHT_ANKLE, 15.0, "ankles")
        # Elbows: moderate
        _bilateral_y_clamp(KP.LEFT_ELBOW, KP.RIGHT_ELBOW, 15.0, "elbows")
        # Wrists: most mobile but still bounded
        _bilateral_y_clamp(KP.LEFT_WRIST, KP.RIGHT_WRIST, 20.0, "wrists")

        # ---- Chain Y-depth constraints (parent→child can't exceed bone) ----
        _chain_y_clamp(KP.LEFT_HIP, KP.LEFT_KNEE, 'thigh_l', "L_hip→L_knee")
        _chain_y_clamp(KP.LEFT_KNEE, KP.LEFT_ANKLE, 'shin_l', "L_knee→L_ankle")
        _chain_y_clamp(KP.RIGHT_HIP, KP.RIGHT_KNEE, 'thigh_r', "R_hip→R_knee")
        _chain_y_clamp(KP.RIGHT_KNEE, KP.RIGHT_ANKLE, 'shin_r', "R_knee→R_ankle")
        _chain_y_clamp(KP.LEFT_SHOULDER, KP.LEFT_ELBOW, 'upper_arm_l', "L_sho→L_elb")
        _chain_y_clamp(KP.LEFT_ELBOW, KP.LEFT_WRIST, 'forearm_l', "L_elb→L_wri")
        _chain_y_clamp(KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, 'upper_arm_r', "R_sho→R_elb")
        _chain_y_clamp(KP.RIGHT_ELBOW, KP.RIGHT_WRIST, 'forearm_r', "R_elb→R_wri")

    # =========================================================================
    # STEP 10b: Per-joint velocity clamping in ICCS  (Bug 3 + Bug 7 fix)
    # =========================================================================

    # Velocity limits in cm/frame (tuned for 10-15 fps)
    # Core joints move slower; extremities get more freedom
    _VELOCITY_LIMITS = {
        # Spine chain — very slow, controlled (stationary/seated person)
        KP.PELVIS_CENTER:    0.0,   # pelvis is DERIVED = midpoint(hips), never directly clamped
        KP.SPINE_MID:        1.5,   # spine barely moves between frames
        KP.SHOULDER_CENTER:  1.5,   # same — structural joint
        KP.HEAD_CENTER:      2.0,   # slight nod/turn allowed
        # Hips — stable structural
        KP.LEFT_HIP:         1.5,
        KP.RIGHT_HIP:        1.5,
        # Shoulders — moderate (shrugs, leaning)
        KP.LEFT_SHOULDER:    3.0,
        KP.RIGHT_SHOULDER:   3.0,
        # Elbows — more mobile
        KP.LEFT_ELBOW:       8.0,
        KP.RIGHT_ELBOW:      8.0,
        # Wrists — most mobile
        KP.LEFT_WRIST:       12.0,
        KP.RIGHT_WRIST:      12.0,
        # Knees — moderate
        KP.LEFT_KNEE:        5.0,
        KP.RIGHT_KNEE:       5.0,
        # Ankles — moderate
        KP.LEFT_ANKLE:       6.0,
        KP.RIGHT_ANKLE:      6.0,
        # Face — controlled
        KP.NOSE:             3.0,
        KP.LEFT_EYE:         2.0,
        KP.RIGHT_EYE:        2.0,
        KP.LEFT_EAR:         2.0,
        KP.RIGHT_EAR:        2.0,
    }

    # Fitting error threshold: joints above this fall back to previous frame.
    #
    # FIT_ERROR_FALLBACK: after STEP 8 corrects joints toward voxel_under_pixel
    # centroids, fitting_errors contains post-correction residuals (distance from
    # corrected world position to target centroid).  Well-fitted joints have
    # errors < 3–5 cm; joints with no valid voxel_under_pixel keep their
    # reprojection error which may be 20–60 cm.
    #
    # 30 cm threshold: fires only for joints that STEP 8 could not correct
    # (no valid voxel_under_pixel in cluster) AND whose reprojection error is
    # large enough to indicate a genuinely wrong placement.  For those joints,
    # the previous frame position is more reliable than the current bad fit.
    _FIT_ERROR_FALLBACK_CM = 30.0

    def _apply_velocity_clamping_iccs(self) -> int:
        """
        Compare each joint's ICCS position to the previous frame and clamp
        displacements that exceed physical plausibility.
        
        Also applies per-joint fallback: if fitting_error > threshold AND
        previous ICCS position is available, use the previous position
        entirely (the fit was too unreliable to trust).
        
        FRAME 1 FIX: When no previous frame exists, joints with fitting
        errors above threshold are pulled toward the rest-pose position
        (the T-pose skeleton), which provides a reasonable structural
        baseline even without temporal history.
        
        Returns:
            Number of joints that were clamped or replaced.
        """
        prev = self.skeleton.previous_keypoints_iccs
        kp = self.skeleton.keypoints_iccs
        clamped_count = 0

        if prev is None:
            # Frame 1 — no previous data: apply REST-POSE fallback for
            # joints with high fitting error.  The rest pose (T-pose) gives
            # anatomically plausible positions that prevent wild outliers
            # from polluting the temporal history.
            rest_pose = np.zeros((NUM_KEYPOINTS, 3))
            # Rebuild rest pose from bone lengths (same as _init_rest_pose_iccs)
            _bl = self.skeleton.bone_lengths
            _ls = _bl.get('lower_spine', 0)
            _us = _bl.get('upper_spine', 0)
            _nk = _bl.get('neck', 0)
            torso_h = _ls + _us
            rest_pose[KP.PELVIS_CENTER] = [0, 0, 0]
            rest_pose[KP.SPINE_MID] = [0, 0, _ls]
            rest_pose[KP.SHOULDER_CENTER] = [0, 0, torso_h]
            rest_pose[KP.HEAD_CENTER] = [0, 0, torso_h + _nk]
            hw = _bl.get('hip_width', 0) / 2
            sw = _bl.get('shoulder_width', 0) / 2
            rest_pose[KP.LEFT_HIP] = [-hw, 0, 0]
            rest_pose[KP.RIGHT_HIP] = [+hw, 0, 0]
            rest_pose[KP.LEFT_SHOULDER] = [-sw, 0, torso_h]
            rest_pose[KP.RIGHT_SHOULDER] = [+sw, 0, torso_h]

            for kp_idx in range(NUM_KEYPOINTS):
                if kp_idx == KP.PELVIS_CENTER:
                    continue
                # Face keypoints = rigid head, not individually clamped
                if kp_idx in FACE_KEYPOINTS:
                    continue
                fit_err = self.fitting_errors.get(kp_idx, 0.0)
                if fit_err > self._FIT_ERROR_FALLBACK_CM:
                    if not np.allclose(rest_pose[kp_idx], 0):
                        # Blend: 70% rest pose, 30% fitted (soften the snap)
                        kp[kp_idx] = rest_pose[kp_idx] * 0.7 + kp[kp_idx] * 0.3
                        clamped_count += 1
                        logger.info(
                            f"[STEP 10b] FRAME 1 KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                            f"fit_error={fit_err:.1f}cm > {self._FIT_ERROR_FALLBACK_CM}cm "
                            f"→ blended toward rest pose"
                        )
            return clamped_count

        kp = self.skeleton.keypoints_iccs
        clamped_count = 0

        # Retrieve previous-frame velocity vectors for direction consistency
        prev_vel = getattr(self.skeleton, 'previous_velocities_iccs', None)

        for kp_idx in range(NUM_KEYPOINTS):
            if kp_idx == KP.PELVIS_CENTER:
                continue  # pelvis is DERIVED from hips in _reanchor_computed_keypoints — not clamped directly

            # Face keypoints (0-4) are part of the RIGID HEAD TRAPEZOID.
            # They move as a rigid body with HEAD_CENTER — clamping them
            # individually would break the rigid constraint.  HEAD_CENTER
            # (KP=17) already has its own velocity limit (2.0 cm/frame).
            # After clamping, _reanchor_computed_keypoints will call
            # _enforce_face_geometry to re-place the rigid head.
            if kp_idx in FACE_KEYPOINTS:
                continue

            prev_pos = prev[kp_idx]
            curr_pos = kp[kp_idx]

            # Skip uninitialized joints
            if np.allclose(prev_pos, 0) and np.allclose(curr_pos, 0):
                continue

            # ---- Bug 7: Bad-fit fallback ----
            fit_err = self.fitting_errors.get(kp_idx, 0.0)
            if fit_err > self._FIT_ERROR_FALLBACK_CM and not np.allclose(prev_pos, 0):
                kp[kp_idx] = prev_pos.copy()
                clamped_count += 1
                logger.info(
                    f"[STEP 10b] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                    f"fit_error={fit_err:.1f}cm > {self._FIT_ERROR_FALLBACK_CM}cm "
                    f"→ fallback to previous ICCS position"
                )
                continue

            # ---- Velocity vector ----
            delta = curr_pos - prev_pos
            dist = np.linalg.norm(delta)

            if dist < 0.001:
                continue  # No movement — nothing to clamp

            # ---- Direction consistency check ----
            # If previous velocity exists, compare directions.
            # Large direction reversals (>120°) indicate jitter, not real motion.
            # Dampen: reduce magnitude to 30% and blend direction toward previous.
            if prev_vel is not None:
                prev_v = prev_vel[kp_idx]
                prev_v_norm = np.linalg.norm(prev_v)
                if prev_v_norm > 0.5:  # Only check if previous had meaningful velocity
                    curr_dir = delta / dist
                    prev_dir = prev_v / prev_v_norm
                    cos_angle = np.clip(np.dot(curr_dir, prev_dir), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))

                    if angle > 90.0:
                        # Strong reversal: dampen to 30% magnitude, blend direction
                        damped_dist = dist * 0.3
                        blended_dir = 0.4 * curr_dir + 0.6 * prev_dir
                        blended_norm = np.linalg.norm(blended_dir)
                        if blended_norm > 0.001:
                            blended_dir = blended_dir / blended_norm
                        delta = blended_dir * damped_dist
                        dist = damped_dist
                        clamped_count += 1
                        logger.info(
                            f"[STEP 10b] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                            f"direction reversal {angle:.0f}° → dampened to 30%"
                        )
                    elif angle > 60.0:
                        # Moderate reversal: dampen to 50% magnitude
                        damped_dist = dist * 0.5
                        delta = (delta / dist) * damped_dist
                        dist = damped_dist
                        clamped_count += 1
                        logger.debug(
                            f"[STEP 10b] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                            f"direction change {angle:.0f}° → dampened to 50%"
                        )

            # ---- Magnitude clamping ----
            limit = self._VELOCITY_LIMITS.get(kp_idx, 10.0)
            if dist > limit and limit > 0:
                delta = (delta / dist) * limit
                clamped_count += 1
                logger.debug(
                    f"[STEP 10b] KP {kp_idx} ({KEYPOINT_NAMES[kp_idx]}): "
                    f"velocity {dist:.1f}cm/frame → clamped to {limit}cm/frame"
                )

            kp[kp_idx] = prev_pos + delta

        return clamped_count

    # =========================================================================
    # STEP 10: Enforce locked bone lengths (FINAL PASS)
    # =========================================================================

    def _enforce_bone_lengths(self):
        """
        Walk every kinematic chain from root to tip in ICCS space.
        For each bone segment, preserve the DIRECTION (DoF / joint angle)
        but normalize the LENGTH to the locked proportional value derived
        from skeleton height.

        Processing order (parent must be finalised before child):
          1. Spine:  PELVIS_CENTER → SPINE_MID → SHOULDER_CENTER → HEAD_CENTER
          2. Hips:   PELVIS_CENTER → LEFT_HIP / RIGHT_HIP  (half hip_width)
          3. Shoulders: SHOULDER_CENTER → LEFT_SHOULDER / RIGHT_SHOULDER
          4. Arms:   shoulder → elbow → wrist  (each side)
          5. Legs:   hip → knee → ankle  (each side)
        """
        kp  = self.skeleton.keypoints_iccs
        bl  = self.skeleton.bone_lengths

        def _enforce(parent_kp, child_kp, bone_name):
            """Set child = parent + direction * locked_length."""
            L = bl.get(bone_name, 0.0)
            if L <= 0:
                return
            p = kp[parent_kp]
            c = kp[child_kp]
            # BUG 6 FIX: Allow parent to be [0,0,0] (pelvis IS the ICCS origin).
            # Only skip if CHILD is uninitialized (all zeros AND not pelvis).
            if np.allclose(c, 0) and child_kp != KP.PELVIS_CENTER:
                return
            d = c - p
            dist = np.linalg.norm(d)
            if dist < 0.001:
                return  # Degenerate — can't determine direction
            kp[child_kp] = p + (d / dist) * L

        # ---- 1. Spine chain: SKIP ----
        # SHOULDER_CENTER, SPINE_MID, HEAD_CENTER are DEFINED as midpoints
        # of their respective keypoint pairs (shoulders, hips, ears).
        # Chain propagation overrides these data-driven positions with
        # direction + length walks that create "goose neck" artifacts.
        # Instead, _reanchor_computed_keypoints() is called after enforce
        # to place them from their definitions.
        # Spine BONE LENGTHS are measured (not enforced) and learned by
        # calibration (STEP 13).

        # ---- 2. Hips + Pelvis collinearity (BUG #1 + COLLINEARITY FIX) ----
        #
        # PELVIS_CENTER is anatomically DEFINED as midpoint(LEFT_HIP, RIGHT_HIP).
        # Step 1 (establish_iccs) guarantees this by applying clamp_offset to
        # both hip world positions before ICCS conversion.  But downstream steps
        # (depth correction, bilateral clamping, velocity clamping) may shift
        # individual hip ICCS coordinates, drifting pelvis from the midpoint.
        #
        # TWO-PART FIX:
        #   Part A: Re-enforce X placement relative to pelvis (not hardcoded 0)
        #   Part B: Re-anchor PELVIS_CENTER = midpoint(hips) as definitive guard
        #
        # X-only rule (Bug #1 fix preserved): Y and Z from CoP are not touched.
        half_hip  = bl.get('hip_width', 0.0) / 2
        pelvis_x  = kp[KP.PELVIS_CENTER][0]   # robust: use actual pelvis X, not 0
        if half_hip > 0:
            for hip_kp, sign in [(KP.LEFT_HIP, -1), (KP.RIGHT_HIP, +1)]:
                kp[hip_kp][0] = pelvis_x + sign * half_hip  # X relative to pelvis

        # Part B: Re-anchor PELVIS_CENTER = midpoint(hips) — the anatomical definition.
        # Corrects any residual drift introduced by steps 7–10b between X enforcement
        # above and the downstream world-coord rebuild.
        l_hip_iccs = kp[KP.LEFT_HIP]
        r_hip_iccs = kp[KP.RIGHT_HIP]
        kp[KP.PELVIS_CENTER] = (l_hip_iccs + r_hip_iccs) / 2.0

        # Verify collinearity (log only — should be sub-mm after fix)
        _mid = (l_hip_iccs + r_hip_iccs) / 2.0
        _drift = np.linalg.norm(_mid - kp[KP.PELVIS_CENTER])
        if _drift > 0.5:
            logger.warning(f"[STEP 10] Pelvis drift after enforce: {_drift:.2f}cm")

        # ---- 3. Shoulders: lateral X from CoP measurement, not template ----
        #
        # GORILLA BUG FIX: The template shoulder_width (0.235 × H ≈ 40cm) was
        # unconditionally written over the CoP-measured shoulder width every frame,
        # making every skeleton look artificially broad (gorilla shoulders).
        #
        # CORRECT POLICY:
        #   • Before calibration (frame_count < 3): allow template as a seed.
        #   • After calibration (is_calibrated): the MEASURED shoulder width is
        #     locked in bl['shoulder_width'].  Trust it — do NOT re-enforce with
        #     the template.  Only enforce DIRECTION (left < right laterally).
        #
        # Additionally: ICCS LEFT = negative X, ICCS RIGHT = positive X.
        # LEFT_SHOULDER.X must be < SHOULDER_CENTER.X and
        # RIGHT_SHOULDER.X must be > SHOULDER_CENTER.X — always.
        half_shoulder = bl.get('shoulder_width', 0.0) / 2
        sc = kp[KP.SHOULDER_CENTER]
        is_calibrated = getattr(self.skeleton, 'is_calibrated', False)
        frame_count   = getattr(self.skeleton, 'frame_count', 0)

        if half_shoulder > 0 and np.linalg.norm(sc) > 0.001:
            if not is_calibrated and frame_count < 3:
                # Pre-calibration seed: use template proportions to place shoulders
                # symmetrically.  CoP data for early frames is unreliable.
                for sh_kp, sign in [(KP.LEFT_SHOULDER, -1), (KP.RIGHT_SHOULDER, +1)]:
                    old_pos = kp[sh_kp].copy()
                    y_offset = old_pos[1] - sc[1]
                    kp[sh_kp] = np.array([
                        sc[0] + sign * half_shoulder,
                        sc[1] + y_offset,
                        sc[2]
                    ])
                logger.debug("[STEP 10] Shoulder X: template seed (pre-calibration)")
            else:
                # Post-calibration: trust measured width.
                # Only enforce the SIGN rule (left X < SC X < right X) to
                # prevent the anti-crossing condition without overriding width.
                l_x = kp[KP.LEFT_SHOULDER][0]
                r_x = kp[KP.RIGHT_SHOULDER][0]
                sc_x = sc[0]
                # If lateral ordering is violated, swap X only
                if l_x > r_x:
                    kp[KP.LEFT_SHOULDER][0], kp[KP.RIGHT_SHOULDER][0] = r_x, l_x
                    logger.warning("[STEP 10] ANTI-CROSS shoulder: swapped L/R X")
                # Clamp: left must be ≤ sc_x, right must be ≥ sc_x
                _min_sep = max(1.0, half_shoulder * 0.5)   # at least half of target half-width
                if kp[KP.LEFT_SHOULDER][0] > sc_x:
                    kp[KP.LEFT_SHOULDER][0] = sc_x - _min_sep
                if kp[KP.RIGHT_SHOULDER][0] < sc_x:
                    kp[KP.RIGHT_SHOULDER][0] = sc_x + _min_sep
                logger.debug("[STEP 10] Shoulder X: direction guard only (post-calibration)")

        # ---- 4. Arms ----
        _enforce(KP.LEFT_SHOULDER,  KP.LEFT_ELBOW,   'upper_arm_l')
        _enforce(KP.LEFT_ELBOW,     KP.LEFT_WRIST,   'forearm_l')
        _enforce(KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW,  'upper_arm_r')
        _enforce(KP.RIGHT_ELBOW,    KP.RIGHT_WRIST,  'forearm_r')

        # ---- 4b. Arm anti-crossing guard ----
        # After bone enforcement ensure left arm is laterally left of right arm.
        for l_kp, r_kp, lbl in [
            (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER, 'shoulder'),
            (KP.LEFT_ELBOW,    KP.RIGHT_ELBOW,    'elbow'),
            (KP.LEFT_WRIST,    KP.RIGHT_WRIST,    'wrist'),
        ]:
            lx = kp[l_kp][0]
            rx = kp[r_kp][0]
            if lx > rx:
                kp[l_kp][0], kp[r_kp][0] = rx, lx
                logger.warning(
                    f"[STEP 10] ANTI-CROSS {lbl}: L_X={lx:.1f} > R_X={rx:.1f} — swapped")

        # ---- 5. Legs ----
        _enforce(KP.LEFT_HIP,   KP.LEFT_KNEE,   'thigh_l')
        _enforce(KP.LEFT_KNEE,  KP.LEFT_ANKLE,  'shin_l')
        _enforce(KP.RIGHT_HIP,  KP.RIGHT_KNEE,  'thigh_r')
        _enforce(KP.RIGHT_KNEE, KP.RIGHT_ANKLE, 'shin_r')

        # ---- 5b. Leg anti-crossing guard ----
        #
        # After bone enforcement, verify that left leg joints are on the LEFT
        # side (ICCS_X < right side) and right leg joints are on the RIGHT.
        # This catches the "crossed legs" artifact where pixel-ray refinement
        # for a back-facing person places left_knee at positive ICCS_X.
        #
        # Rule:  LEFT_KP.iccs_x  ≤  midpoint_x  ≤  RIGHT_KP.iccs_x
        # If violated, SWAP the X-coordinates of the crossing pair so the
        # lateral ordering is restored without changing depth/height.
        #
        _pelvis_x = kp[KP.PELVIS_CENTER][0]
        for l_kp, r_kp, label in [
            (KP.LEFT_HIP,   KP.RIGHT_HIP,   'hip'),
            (KP.LEFT_KNEE,  KP.RIGHT_KNEE,  'knee'),
            (KP.LEFT_ANKLE, KP.RIGHT_ANKLE, 'ankle'),
        ]:
            lx = kp[l_kp][0]
            rx = kp[r_kp][0]
            if lx > rx:  # crossed: left is to the right of right
                # Swap X only — preserve Y (depth) and Z (height)
                kp[l_kp][0], kp[r_kp][0] = rx, lx
                logger.warning(
                    f"[STEP 10] ANTI-CROSS {label}: L_X={lx:.1f} > R_X={rx:.1f} — "
                    f"swapped to L_X={kp[l_kp][0]:.1f}, R_X={kp[r_kp][0]:.1f}"
                )

        # ---- Log enforcement summary ----
        logger.info("[STEP 10] Bone lengths enforced (direction preserved, lengths locked)")

    # =========================================================================
    # STEP 10a: Re-anchor computed keypoints from their definitions
    # =========================================================================

    def _reanchor_computed_keypoints(self):
        """
        Re-place the 4 computed keypoints (17-20) from their DEFINING
        keypoint pairs after any step that may have moved joints.

        Definitions (anatomical, not ICCS assumptions):
          - PELVIS_CENTER (19) = midpoint(LEFT_HIP, RIGHT_HIP)  ← THE LAW
          - SHOULDER_CENTER (18) = midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
          - SPINE_MID (20) = pelvis + (shoulder_center − pelvis) × spine_ratio
          - HEAD_CENTER (17) = midpoint(LEFT_EAR, RIGHT_EAR) ← the only
            flexible joint in the RIGID HEAD TRAPEZOID.  All other face
            keypoints (ears, eyes, nose) are placed by rotating the rigid
            template around HEAD_CENTER via _enforce_face_geometry.

        PELVIS_CENTER is computed from hips, NOT assumed to be [0,0,0].
        In a clean frame it IS [0,0,0] — but any drift from depth correction,
        bilateral clamping, or velocity clamping is corrected here, ensuring
        pelvis always lies on the hip line (anatomically mandatory).

        This MUST run after _enforce_bone_lengths and _apply_velocity_clamping
        to prevent chain propagation from overriding data-driven positions.
        """
        kp = self.skeleton.keypoints_iccs
        bl = self.skeleton.bone_lengths

        # PELVIS_CENTER = midpoint(LEFT_HIP, RIGHT_HIP) — the anatomical law.
        # This is the definitive collinearity enforcement: no matter what
        # upstream steps did to individual hip coordinates, the pelvis is
        # ALWAYS on the line between them at the exact midpoint.
        l_hip = kp[KP.LEFT_HIP]
        r_hip = kp[KP.RIGHT_HIP]
        if not (np.allclose(l_hip, 0) and np.allclose(r_hip, 0)):
            new_pelvis = (l_hip + r_hip) / 2.0
            drift = np.linalg.norm(new_pelvis - kp[KP.PELVIS_CENTER])
            if drift > 0.1:  # log only when non-trivial drift occurred
                logger.info(f"[STEP 10a] PELVIS re-anchored: drift={drift:.2f}cm "
                            f"({kp[KP.PELVIS_CENTER].tolist()} → {new_pelvis.tolist()})")
            kp[KP.PELVIS_CENTER] = new_pelvis

        # SHOULDER_CENTER = midpoint(shoulders)
        l_sh = kp[KP.LEFT_SHOULDER]
        r_sh = kp[KP.RIGHT_SHOULDER]
        if not np.allclose(l_sh, 0) or not np.allclose(r_sh, 0):
            kp[KP.SHOULDER_CENTER] = (l_sh + r_sh) / 2

        # SPINE_MID = proportional between PELVIS_CENTER and SHOULDER_CENTER.
        # V3 FIX: Preserve any perpendicular offset from shell fitting.
        # Old code forced SPINE_MID exactly onto the pelvis-shoulder line,
        # making lower_spine and upper_spine have identical directions.
        # New code: only adjust the ALONG-AXIS component to maintain the
        # bone length ratio, but preserve perpendicular deviation.
        pc = kp[KP.PELVIS_CENTER]
        sc = kp[KP.SHOULDER_CENTER]
        if not np.allclose(sc, 0):
            ls = bl.get('lower_spine', 1.0)
            us = bl.get('upper_spine', 1.0)
            ratio = ls / (ls + us) if (ls + us) > 0 else 0.5
            # Ideal on-line position
            on_line = pc + (sc - pc) * ratio
            # Current SPINE_MID position
            current_sm = kp[KP.SPINE_MID]
            if not np.allclose(current_sm, 0):
                # Compute perpendicular offset from the pelvis-shoulder line
                spine_axis = sc - pc
                spine_len = np.linalg.norm(spine_axis)
                if spine_len > 0.1:
                    spine_unit = spine_axis / spine_len
                    # Vector from pelvis to current SPINE_MID
                    pc_to_sm = current_sm - pc
                    # Project onto spine axis
                    along = np.dot(pc_to_sm, spine_unit) * spine_unit
                    # Perpendicular component (the deviation we want to preserve)
                    perp = pc_to_sm - along
                    perp_mag = np.linalg.norm(perp)
                    # Preserve perpendicular offset up to 5cm (anatomically reasonable
                    # for spine curvature); clamp larger values as fitting artifacts
                    if perp_mag > 5.0:
                        perp = perp * (5.0 / perp_mag)
                    # New SPINE_MID = on-line position + preserved perpendicular offset
                    kp[KP.SPINE_MID] = on_line + perp
                else:
                    kp[KP.SPINE_MID] = on_line
            else:
                kp[KP.SPINE_MID] = on_line

        # HEAD_CENTER + all face keypoints — enforced via _enforce_face_geometry.
        # The head is a RIGID isosceles trapezoid: _enforce_face_geometry
        # determines orientation from ear XZ positions, then rotates the
        # stored rigid_head_template to place all 5 face keypoints and
        # HEAD_CENTER as a single rigid body.
        facing = getattr(self, '_facing_direction', 'toward_camera')
        self._enforce_face_geometry(facing_direction=facing)

        logger.debug("[STEP 10a] Computed keypoints re-anchored from definitions")

    # =========================================================================
    # STEP 12: Enforce Range of Motion (ROM) limits
    # =========================================================================

    def _enforce_rom_limits(self) -> int:
        """
        Check all segment angles against anatomical ROM limits.
        If a segment's back-calculated angle exceeds its limit, clamp it
        and reposition the child keypoint accordingly.

        This prevents physically impossible poses (e.g. knee bending
        backwards, spine bending 90°) from entering the temporal history,
        which would otherwise poison velocity clamping in subsequent frames.

        Processing order follows the kinematic tree from root to tips so
        that parent corrections cascade correctly to child segments.

        Returns:
            Number of segments that had ROM violations corrected.
        """
        kp = self.skeleton.keypoints_iccs
        bl = self.skeleton.bone_lengths
        violations = 0

        # Process segments in kinematic order (root → tips)
        CHAIN_ORDER = [
            'lower_spine', 'upper_spine', 'neck', 'head',
            'hip_offset_l', 'hip_offset_r',
            'shoulder_offset_l', 'shoulder_offset_r',
            'thigh_l', 'shin_l', 'thigh_r', 'shin_r',
            'upper_arm_l', 'forearm_l', 'upper_arm_r', 'forearm_r',
        ]

        for seg_name in CHAIN_ORDER:
            seg = self.skeleton.segments.get(seg_name)
            if seg is None:
                continue

            changed = False

            # Check and clamp each DoF
            for axis, limits, current in [
                ('rx', seg.rx_limits, seg.rx),
                ('ry', seg.ry_limits, seg.ry),
                ('rz', seg.rz_limits, seg.rz),
            ]:
                if limits is None:
                    continue  # No limit on this axis → no DoF (locked at 0)
                lo, hi = limits
                if current < lo:
                    setattr(seg, axis, lo)
                    changed = True
                    logger.info(
                        f"[STEP 12] ROM {seg_name}.{axis}: {current:.1f}° < {lo}° → clamped to {lo}°"
                    )
                elif current > hi:
                    setattr(seg, axis, hi)
                    changed = True
                    logger.info(
                        f"[STEP 12] ROM {seg_name}.{axis}: {current:.1f}° > {hi}° → clamped to {hi}°"
                    )

            if changed:
                violations += 1
                # Recompute child keypoint from clamped angles.
                # PHASE 2B FIX: apply the CLAMPED Euler rotation to the
                # segment's rest direction, so the child position is
                # consistent with the angles that were just clamped.
                # The old code preserved the pre-clamp direction, which
                # meant positions and angles were always inconsistent.
                parent_pos = kp[seg.parent_kp]
                child_pos  = kp[seg.child_kp]
                bone_len = bl.get(seg_name, np.linalg.norm(child_pos - parent_pos))

                if bone_len < 0.001:
                    continue

                # Get rest direction for this segment (matches back_calculate_all_segment_angles).
                # Import inside method body to avoid circular import at module level
                # (anatomical_skeleton already imports from movement_index inside
                # fit_to_cluster_shell, so this is safe).
                try:
                    from movement_index import _get_segment_rest_direction
                    rest_dir = _get_segment_rest_direction(seg_name, seg.parent_kp, seg.child_kp)
                except Exception:
                    # Fallback: keep existing direction, just re-scale to bone_len
                    old_dir = child_pos - parent_pos
                    old_dist = np.linalg.norm(old_dir)
                    if old_dist > 0.001:
                        kp[seg.child_kp] = parent_pos + (old_dir / old_dist) * bone_len
                    continue

                # Apply CLAMPED Euler rotation to rest direction
                try:
                    R = Rotation.from_euler('xyz',
                        [seg.rx, seg.ry, seg.rz], degrees=True).as_matrix()
                except Exception:
                    continue

                new_dir = R @ rest_dir
                new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-10)
                kp[seg.child_kp] = parent_pos + new_dir * bone_len

        if violations > 0:
            logger.info(f"[STEP 12] Total ROM violations corrected: {violations}")
        return violations
    
    # =========================================================================
    # PLY-SURFACE FITTING  (visualization / poisson_humanoid export)
    # =========================================================================

    @staticmethod
    def _ply_to_world_verts(ply_verts: np.ndarray) -> np.ndarray:
        """
        Convert vertices from native PLY file space to WORLD space.

        create_meshes.py saves PLY with:
            PLY_X = -world_X
            PLY_Y =  world_Z   (Y↔Z swap)
            PLY_Z =  world_Y

        Inverse (PLY → world):
            world_X = -PLY_X
            world_Y =  PLY_Z
            world_Z =  PLY_Y

        Args:
            ply_verts: (N,3) vertices as stored on disk (PLY space).
        Returns:
            (N,3) vertices in world / ICCS-aligned space (cm).
        """
        v = np.asarray(ply_verts, dtype=np.float64)
        world = np.empty_like(v)
        world[:, 0] = -v[:, 0]   # world_X = -PLY_X
        world[:, 1] =  v[:, 2]   # world_Y =  PLY_Z
        world[:, 2] =  v[:, 1]   # world_Z =  PLY_Y
        return world

    def fit_to_ply_surface(
        self,
        ply_verts_world: np.ndarray,
        canonical_kps_world: np.ndarray,
        facing_direction: str = 'toward_camera',
    ) -> np.ndarray:
        """
        Fit canonical Skeleton-21 humanoid joints to the Poisson PLY surface.

        The canonical humanoid (built by build_skeleton21_from_cluster_bbox) is
        already positioned inside the point cloud, but its joints may not touch
        the mesh surface.  This method "inflates" the skeleton outward along the
        kinematic chain until every joint lies on the surface (offset inward by
        its flesh radius), while preserving bone lengths.

        FITTING ORDER — hips and shoulders fitted FIRST so that distal joints
        inherit correctly anchored parents:

          1. Stance ankle    — anchored to floor Z slab of PLY mesh (toehold)
          2. Stance knee     — surface-snapped, shin bone-length enforced from ankle
          3. Stance hip      — surface-snapped, thigh bone-length enforced from knee
          4. Pelvis center   — surface-snapped, then re-anchored = midpoint(hips)
          5. Swing hip       — surface-snapped, hip_width enforced from pelvis
                               (pelvis re-anchored again after both hips placed)
          6. Spine chain     — shoulder_center surface-snapped + torso length
                               from pelvis; spine_mid set proportionally
          7. Both shoulders  — surface-snapped, shoulder_width enforced from SC
                               (shoulder_center re-anchored = midpoint(shoulders))
          8. Head center     — surface-snapped, neck length from shoulder_center
          9. Swing leg       — knee surface-snap + thigh length; ankle surface-snap
                               + shin length
         10. Both elbows     — surface-snap + upper_arm length from each shoulder
         11. Both wrists     — surface-snap + forearm length from each elbow
         12. Face joints     — rigid body translation: canonical offset from old
                               HEAD_CENTER applied unchanged to new HEAD_CENTER

        Bone lengths come from self.skeleton.bone_lengths (locked from height).
        Flesh radii come from self._get_flesh_radius(kp_idx).

        Args:
            ply_verts_world:     (N,3) PLY mesh vertices in WORLD coordinates.
                                 Use _ply_to_world_verts() to convert from file.
            canonical_kps_world: (21,3) starting joint positions in world coords
                                 (from build_skeleton21_from_cluster_bbox or
                                 shell_fitted_21).
            facing_direction:    'toward_camera' or 'away_from_camera'.

        Returns:
            (21,3) fitted world-space keypoints.  Returns canonical_kps_world
            unchanged if scipy is unavailable or the mesh is degenerate.
        """
        try:
            from scipy.spatial import KDTree as _KDTree
        except ImportError:
            logger.warning("[PLY_FIT] scipy unavailable — returning canonical kps unchanged")
            return canonical_kps_world.copy()

        if ply_verts_world is None or len(ply_verts_world) < 4:
            logger.warning("[PLY_FIT] Empty / degenerate PLY mesh — returning canonical kps")
            return canonical_kps_world.copy()

        kps  = canonical_kps_world.copy().astype(np.float64)
        bl   = self.skeleton.bone_lengths
        tree = _KDTree(ply_verts_world)
        ply_centroid = ply_verts_world.mean(axis=0)

        # ── helpers ──────────────────────────────────────────────────────────

        def _snap(joint_pos: np.ndarray, kp_idx: int) -> np.ndarray:
            """
            Nearest-surface point on the PLY mesh, offset inward by flesh radius.
            Returns a position the BONE JOINT should occupy (inside the skin).
            """
            _, idx = tree.query(joint_pos)
            surface_pt = ply_verts_world[int(idx)]
            flesh_r = self._get_flesh_radius(kp_idx)
            if flesh_r > 0.0:
                offset = self._compute_flesh_inward_offset(
                    surface_pt, ply_centroid, flesh_r)
                return surface_pt + offset
            return surface_pt.copy()

        def _bone(parent_kp: int, child_kp: int,
                  bone_key: str, snapped_child: np.ndarray) -> np.ndarray:
            """
            Place child at exactly bone_length from parent in the direction of
            snapped_child.  If direction is degenerate, fall back to the
            canonical direction so we never lose the proportional shape.
            """
            L = bl.get(bone_key, 0.0)
            if L <= 0.0:
                return snapped_child.copy()
            parent_pos = kps[parent_kp]
            d = snapped_child - parent_pos
            dist = float(np.linalg.norm(d))
            if dist < 0.5:
                # degenerate: use canonical bone direction as fallback
                d_can = (canonical_kps_world[child_kp] -
                         canonical_kps_world[parent_kp])
                dist_can = float(np.linalg.norm(d_can))
                if dist_can > 0.5:
                    d = d_can
                    dist = dist_can
                else:
                    return parent_pos + np.array([0.0, 0.0, -L])  # last resort: down
            return parent_pos + (d / dist) * L

        # ── STEP 1: Identify stance ankle (lower world Z = floor contact) ────

        kp15_z = float(kps[KP.LEFT_ANKLE][2])
        kp16_z = float(kps[KP.RIGHT_ANKLE][2])
        stance_kp  = KP.LEFT_ANKLE  if kp15_z <= kp16_z else KP.RIGHT_ANKLE
        swing_kp   = KP.RIGHT_ANKLE if stance_kp == KP.LEFT_ANKLE else KP.LEFT_ANKLE
        stance_s   = 'l' if stance_kp == KP.LEFT_ANKLE else 'r'
        swing_s    = 'r' if stance_s == 'l' else 'l'
        stance_knee_kp = KP.LEFT_KNEE  if stance_s == 'l' else KP.RIGHT_KNEE
        stance_hip_kp  = KP.LEFT_HIP   if stance_s == 'l' else KP.RIGHT_HIP
        swing_knee_kp  = KP.RIGHT_KNEE if stance_s == 'l' else KP.LEFT_KNEE
        swing_hip_kp   = KP.RIGHT_HIP  if stance_s == 'l' else KP.LEFT_HIP

        # Floor Z: lowest PLY vertex within 20 cm XY of stance ankle
        ankle_xy   = kps[stance_kp, :2]
        xy_dist    = np.sqrt(np.sum((ply_verts_world[:, :2] - ankle_xy) ** 2, axis=1))
        mask_near  = xy_dist < 20.0
        if mask_near.sum() < 3:
            mask_near = xy_dist < 50.0
        floor_z = float(ply_verts_world[mask_near, 2].min()) if mask_near.sum() > 0 \
                  else float(ply_verts_world[:, 2].min())

        ankle_flesh = self._get_flesh_radius(stance_kp)
        kps[stance_kp][2] = floor_z + ankle_flesh
        # XY from canonical (already placed inside cluster bbox)

        logger.info(f"[PLY_FIT] stance=KP{stance_kp}({KEYPOINT_NAMES[stance_kp]}) "
                    f"floor_z={floor_z:.1f} ankle_z={kps[stance_kp][2]:.1f}cm")

        # ── STEP 2: Stance leg IK upward (ankle → knee → hip) ────────────────

        kps[stance_knee_kp] = _bone(
            stance_kp, stance_knee_kp, f'shin_{stance_s}',
            _snap(kps[stance_knee_kp], stance_knee_kp))

        kps[stance_hip_kp] = _bone(
            stance_knee_kp, stance_hip_kp, f'thigh_{stance_s}',
            _snap(kps[stance_hip_kp], stance_hip_kp))

        # ── STEP 3: Pelvis — surface-snapped ─────────────────────────────────

        half_hip = bl.get('hip_width', 32.0) / 2.0
        kps[KP.PELVIS_CENTER] = _snap(kps[KP.PELVIS_CENTER], KP.PELVIS_CENTER)

        # ── STEP 4: Swing hip — hip_width from pelvis; re-anchor pelvis ──────

        swing_hip_snapped = _snap(kps[swing_hip_kp], swing_hip_kp)
        hip_dir  = swing_hip_snapped - kps[KP.PELVIS_CENTER]
        hip_dist = float(np.linalg.norm(hip_dir))
        if hip_dist > 0.5:
            kps[swing_hip_kp] = kps[KP.PELVIS_CENTER] + (hip_dir / hip_dist) * half_hip
        # Also enforce stance hip lateral distance from pelvis
        shire_dir  = kps[stance_hip_kp] - kps[KP.PELVIS_CENTER]
        shire_dist = float(np.linalg.norm(shire_dir))
        if shire_dist > 0.5:
            kps[stance_hip_kp] = kps[KP.PELVIS_CENTER] + (shire_dir / shire_dist) * half_hip

        # Anatomical law: pelvis = midpoint(hips)
        kps[KP.PELVIS_CENTER] = (kps[KP.LEFT_HIP] + kps[KP.RIGHT_HIP]) / 2.0

        logger.info(f"[PLY_FIT] Pelvis={kps[KP.PELVIS_CENTER].tolist()}, "
                    f"L_hip={kps[KP.LEFT_HIP].tolist()}, "
                    f"R_hip={kps[KP.RIGHT_HIP].tolist()}")

        # ── STEP 5: Spine — shoulder_center surface-snapped + torso length ───

        torso_L = bl.get('torso',
                         bl.get('lower_spine', 22.0) + bl.get('upper_spine', 22.0))
        kps[KP.SHOULDER_CENTER] = _bone(
            KP.PELVIS_CENTER, KP.SHOULDER_CENTER, 'torso',
            _snap(kps[KP.SHOULDER_CENTER], KP.SHOULDER_CENTER))

        ls    = bl.get('lower_spine', 1.0)
        us    = bl.get('upper_spine',  1.0)
        ratio = ls / (ls + us) if (ls + us) > 0 else 0.5
        kps[KP.SPINE_MID] = (kps[KP.PELVIS_CENTER] +
                              (kps[KP.SHOULDER_CENTER] - kps[KP.PELVIS_CENTER]) * ratio)

        # ── STEP 6: Both shoulders — shoulder_width from shoulder_center ─────

        half_shoulder = bl.get('shoulder_width', 39.95) / 2.0
        for sh_kp in (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER):
            sh_snapped = _snap(kps[sh_kp], sh_kp)
            sh_dir  = sh_snapped - kps[KP.SHOULDER_CENTER]
            sh_dist = float(np.linalg.norm(sh_dir))
            if sh_dist > 0.5:
                kps[sh_kp] = kps[KP.SHOULDER_CENTER] + (sh_dir / sh_dist) * half_shoulder
            else:
                # fallback: keep canonical X offset from SC
                d_can = canonical_kps_world[sh_kp] - canonical_kps_world[KP.SHOULDER_CENTER]
                d_len = float(np.linalg.norm(d_can))
                if d_len > 0.5:
                    kps[sh_kp] = kps[KP.SHOULDER_CENTER] + (d_can / d_len) * half_shoulder

        # Anti-crossing: left shoulder must be at lower world-X than right
        # (assuming camera looks along +Y, so world-X is left/right)
        if kps[KP.LEFT_SHOULDER][0] > kps[KP.RIGHT_SHOULDER][0]:
            kps[KP.LEFT_SHOULDER][0], kps[KP.RIGHT_SHOULDER][0] = (
                kps[KP.RIGHT_SHOULDER][0], kps[KP.LEFT_SHOULDER][0])
            logger.warning("[PLY_FIT] Shoulder anti-cross: swapped X")

        # Re-anchor shoulder_center = midpoint(shoulders)
        kps[KP.SHOULDER_CENTER] = (kps[KP.LEFT_SHOULDER] + kps[KP.RIGHT_SHOULDER]) / 2.0

        logger.info(f"[PLY_FIT] SC={kps[KP.SHOULDER_CENTER].tolist()}, "
                    f"L_sh={kps[KP.LEFT_SHOULDER].tolist()}, "
                    f"R_sh={kps[KP.RIGHT_SHOULDER].tolist()}")

        # ── STEP 7: Head center — neck length from shoulder_center ───────────

        kps[KP.HEAD_CENTER] = _bone(
            KP.SHOULDER_CENTER, KP.HEAD_CENTER, 'neck',
            _snap(kps[KP.HEAD_CENTER], KP.HEAD_CENTER))

        # ── STEP 8: Swing leg (hip already placed; knee → ankle) ─────────────

        kps[swing_knee_kp] = _bone(
            swing_hip_kp, swing_knee_kp, f'thigh_{swing_s}',
            _snap(kps[swing_knee_kp], swing_knee_kp))

        kps[swing_kp] = _bone(
            swing_knee_kp, swing_kp, f'shin_{swing_s}',
            _snap(kps[swing_kp], swing_kp))

        # ── STEP 9: Both elbows → wrists ─────────────────────────────────────

        for sh_kp, el_kp, wr_kp, side in (
            (KP.LEFT_SHOULDER,  KP.LEFT_ELBOW,  KP.LEFT_WRIST,  'l'),
            (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST, 'r'),
        ):
            kps[el_kp] = _bone(sh_kp, el_kp, f'upper_arm_{side}',
                               _snap(kps[el_kp], el_kp))
            kps[wr_kp] = _bone(el_kp, wr_kp, f'forearm_{side}',
                               _snap(kps[wr_kp], wr_kp))

        # ── STEP 10: Face joints — rigid body from HEAD_CENTER ───────────────
        # The 5 face keypoints keep their canonical relative offsets from
        # HEAD_CENTER.  We only translate them to the new head_center position.

        hc_delta = kps[KP.HEAD_CENTER] - canonical_kps_world[KP.HEAD_CENTER]
        for face_kp in (KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE,
                        KP.LEFT_EAR, KP.RIGHT_EAR):
            kps[face_kp] = canonical_kps_world[face_kp] + hc_delta

        logger.info(f"[PLY_FIT] Done. HC={kps[KP.HEAD_CENTER].tolist()}, "
                    f"stance_ankle=KP{stance_kp} z={kps[stance_kp][2]:.1f}cm")
        return kps

    def _voxel_to_world(self, voxel_grid, 
                        voxel_idx: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Convert voxel index to world coordinates (cell center)."""
        if voxel_grid.bounds is None:
            return None
        
        return np.array([
            voxel_grid.bounds[0][0] + (voxel_idx[0] + 0.5) * voxel_grid.resolution,
            voxel_grid.bounds[0][1] + (voxel_idx[1] + 0.5) * voxel_grid.resolution,
            voxel_grid.bounds[0][2] + (voxel_idx[2] + 0.5) * voxel_grid.resolution,
        ])
    
    def _get_cell_centroid(self, voxel_grid,
                           voxel_idx: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Get cell centroid from voxel grid.
        
        Prefers actual centroid from cell_metadata, falls back to cell center.
        """
        # Try to get actual centroid from metadata
        if hasattr(voxel_grid, 'cell_metadata') and voxel_idx in voxel_grid.cell_metadata:
            metadata = voxel_grid.cell_metadata[voxel_idx]
            if 'centroid' in metadata:
                centroid = metadata['centroid']
                if isinstance(centroid, (list, tuple)):
                    return np.array(centroid)
                return centroid
        
        # Try cell_centroid dict
        if hasattr(voxel_grid, 'cell_centroid') and voxel_idx in voxel_grid.cell_centroid:
            return np.array(voxel_grid.cell_centroid[voxel_idx])
        
        # Fall back to cell center
        return self._voxel_to_world(voxel_grid, voxel_idx)


# =============================================================================
# SKELETON MANAGER
# =============================================================================
class SkeletonManager:
    """
    Manages anatomical skeleton across video frames.
    
    Handles:
      - Skeleton initialization and calibration
      - Frame-by-frame pose fitting
      - Temporal smoothing
    """
    
    def __init__(self):
        self.skeleton: Optional[AnatomicalSkeleton] = None
        self.history: List[np.ndarray] = []
        self.max_history = 5
        self.frame_count = 0
    
    def initialize(self, height_cm: float = 170.0) -> AnatomicalSkeleton:
        """Initialize new skeleton"""
        self.skeleton = AnatomicalSkeleton(height_cm)
        self.history.clear()
        self.frame_count = 0
        logger.info(f"SkeletonManager initialized with height={height_cm}cm")
        return self.skeleton
    
    def process_frame(self,
                      cluster_centroid: np.ndarray,
                      rotation_angle: float,
                      detected_keypoints: Optional[np.ndarray] = None,
                      frame_num: int = None) -> np.ndarray:
        """
        Process a frame and return fitted skeleton keypoints.
        
        Args:
            cluster_centroid: Person cluster centroid in world coords
            rotation_angle: Detected body rotation (yaw) in degrees
            detected_keypoints: Optional MMPose 3D keypoints for pose fitting
            frame_num: Frame number (for logging)
            
        Returns:
            17x3 fitted keypoints in world coordinates
        """
        if self.skeleton is None:
            self.initialize()
        
        self.frame_count += 1
        
        # Calibrate from good detections
        if detected_keypoints is not None:
            valid_count = np.sum(np.abs(detected_keypoints).sum(axis=1) > 0.1)
            if valid_count >= 8 and not self.skeleton.is_calibrated:
                self.skeleton.calibrate(detected_keypoints)
        
        # Fit skeleton
        if detected_keypoints is not None:
            keypoints = self.skeleton.fit_to_detection(
                detected_keypoints, cluster_centroid, rotation_angle
            )
        else:
            keypoints = self.skeleton.fit_to_cluster(
                cluster_centroid, rotation_angle
            )
        
        # Temporal smoothing
        self.history.append(keypoints.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        if len(self.history) >= 3:
            keypoints = self._temporal_smooth(keypoints)
        
        return keypoints
    
    def _temporal_smooth(self, current: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """Apply exponential moving average smoothing"""
        if len(self.history) < 2:
            return current
        
        # Weighted blend: more weight on recent frames
        smoothed = current * alpha
        weight_sum = alpha
        
        for i, past in enumerate(reversed(self.history[:-1])):
            w = (1 - alpha) * (0.5 ** i)
            smoothed += past * w
            weight_sum += w
        
        return smoothed / weight_sum
    
    def get_skeleton(self) -> Optional[AnatomicalSkeleton]:
        """Get current skeleton"""
        return self.skeleton
    
    def reset(self):
        """Reset manager state"""
        self.skeleton = None
        self.history.clear()
        self.frame_count = 0


# =============================================================================