"""Perception for the task program: ArUco tag pose from the wrist camera, and
locating the cable housing in the J slot.

Only images, the camera intrinsics, the hand-eye calibration (camera pose in
the TCP frame) and the known fixture geometry are used here, never the
simulator's ground truth.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from geometry import P, Params, Pose, R_FIX_TAG, camera_intrinsics, mat_to_quat, quat_to_mat


class TagDetector:
    def __init__(self, p: Params = P, K=None):
        self.p = p
        self.K = camera_intrinsics(p) if K is None else K
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, p.aruco_dict))
        prm = cv2.aruco.DetectorParameters()
        prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        prm.cornerRefinementWinSize = 5
        prm.cornerRefinementMaxIterations = 50
        prm.cornerRefinementMinAccuracy = 0.01
        self.det = cv2.aruco.ArucoDetector(d, prm)
        s = p.tag_size / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
                            np.float64)

    def detect(self, img, tag_id):
        """Returns the 4 image corners of `tag_id` (TL, TR, BR, BL) or None."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if int(i) == tag_id:
                return c.reshape(4, 2).astype(np.float64)
        return None

    def pose(self, corners):
        """Tag pose in the camera frame (OpenCV axes) plus the RMS
        reprojection error [px]. IPPE gives the two planar solutions; take the
        one that reprojects best."""
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.obj, corners, self.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        i = int(np.argmin(np.asarray(errs).ravel()))
        R, _ = cv2.Rodrigues(rvecs[i])
        return Pose(R, tvecs[i].ravel()), float(np.asarray(errs).ravel()[i])

    def fixture_pose(self, img, tag_id, T_base_cam: Pose):
        """Fixture frame (tag-centred, x out of the face) in the base frame,
        or None if the tag is not seen. Also returns diagnostics."""
        c = self.detect(img, tag_id)
        if c is None:
            return None, None
        T_cam_tag, err = self.pose(c)
        T_base_tag = T_base_cam @ T_cam_tag
        T_base_fix = Pose(T_base_tag.R @ R_FIX_TAG.T, T_base_tag.t)
        centre = c.mean(axis=0)
        return T_base_fix, dict(reproj_px=err, centre_px=centre,
                                size_px=float(np.linalg.norm(c[0] - c[2])),
                                dist=float(np.linalg.norm(T_cam_tag.t)))


def average_poses(poses):
    """Mean position and chordal-mean rotation."""
    t = np.mean([p.t for p in poses], axis=0)
    q0 = mat_to_quat(poses[0].R)
    qs = []
    for p in poses:
        q = mat_to_quat(p.R)
        qs.append(q if q @ q0 >= 0 else -q)
    q = np.mean(qs, axis=0)
    return Pose(quat_to_mat(q / np.linalg.norm(q)), t)


def housing_mask(img):
    """Pixels of the housing's green front flange."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # OpenCV hue is 0..179; green ~ 60. Require strong saturation.
    m = cv2.inRange(hsv, (40, 90, 40), (85, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 150:
        return None
    return lab == k


def locate_housing(img, K, T_base_cam: Pose, T_base_slot: Pose,
                   p: Params = P, z_band=0.008):
    """Neck position along the slot (slot-frame y) from the green front
    flange. Every flange pixel is back-projected onto the plane through the
    middle of the flange (parallel to the plate). The spacer, groove and plug
    sit in front of the flange and hide part of it, but from a view above
    the slot they hide its lower half; the flange's left and right edges at
    the height of its centre stay visible. So the neck is taken as the
    midpoint of the flange's extent in y within a band around the channel
    floor height."""
    m = housing_mask(img)
    if m is None:
        return None, None
    v, u = np.nonzero(m)
    pts = np.stack([u, v, np.ones_like(u)], axis=0).astype(np.float64)
    rays_cam = np.linalg.inv(K) @ pts                       # OpenCV camera
    T_slot_cam = T_base_slot.inv() @ T_base_cam
    o = T_slot_cam.t
    d = T_slot_cam.R @ rays_cam
    x_mid = p.plate_thickness / 2 + 0.002 + p.flange_thickness / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (x_mid - o[0]) / d[0]
    ok = s > 0
    y = o[1] + s[ok] * d[1, ok]
    z = o[2] + s[ok] * d[2, ok]
    band = np.abs(z) < z_band
    if band.sum() < 30:
        return None, None
    y0, y1 = np.percentile(y[band], [1, 99])
    est = (y0 + y1) / 2
    width = y1 - y0
    return float(est), dict(width=float(width), n=int(band.sum()))
