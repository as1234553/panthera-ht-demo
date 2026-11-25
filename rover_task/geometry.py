"""Geometry shared by the scene builder, the task program and the Monte Carlo.

Frames (all right-handed, metres):

  base    Panthera-HT base frame on the rover deck (z up, deck top at z = 0).
  hook    J-hook body frame, origin at the centre of the hook's ArUco tag
          (on its printed surface). x = plate normal pointing out of the front
          face (towards the robot), z = up the stem of the J, y = z x x (to
          the right as seen from the robot). The Monte Carlo tilts the hook
          about this origin.
  slot    Same axes as `hook`, origin at the resting centre of the housing's
          neck (bottom of the J, mid-plane of the plate).
  port    Port body frame, origin at the centre of the port's ArUco tag.
          x = face normal pointing towards the robot, z = up, y = z x x.
          The bore axis (into the hole) is -x. Tilted about this origin.
  tag     OpenCV marker frame: x right, y up, z out of the printed face.
          For both fixtures tag = (y, z, x) of the fixture frame.
  tcp     Tool frame between the finger pads (as in peg_in_hole): z points
          back up the gripper (fingers point along -z), y is the closing
          axis. The housing is held along the tcp x axis with its plug
          towards tcp -x.
  cam     Wrist camera (OpenCV convention: z forward, x right, y down). It
          looks along tcp -x, i.e. the direction the plug points.

The housing ("peg" body) keeps the peg_in_hole convention: its local z axis
runs along the housing from the plug tip (local -z) to the cable end, and its
origin is the middle of the neck, which is also its centre of mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Params:
    # ---------------- cable housing (from the plug tip back) ----------------
    plug_radius: float = 0.010
    plug_length: float = 0.095        # 75 mm into the bore + 20 mm standout
    groove_radius: float = 0.008      # grip groove for the 32 mm pads
    groove_length: float = 0.040
    spacer_length: float = 0.010      # keeps the palm 17 mm off the plate
    flange_radius: float = 0.019
    flange_thickness: float = 0.008
    neck_radius: float = 0.010
    neck_length: float = 0.012        # 8 mm plate + 2 mm each side
    rear_radius: float = 0.013        # strain relief behind the plate
    cable_radius: float = 0.004       # visual stub only
    cable_stub: float = 0.080
    peg_density: float = 2700.0
    # ---------------- gripper / TCP ----------------
    grip_depth: float = 0.017         # TCP 17 mm in from the fingertips
    finger_stroke: float = 0.041
    finger_open: float = 0.029        # inner faces from centre when open
    grip_force: float = 25.0
    # ---------------- wrist camera (in the TCP frame) ----------------
    cam_pos_tcp: tuple = (-0.045, 0.0, 0.125)
    cam_width: int = 1280
    cam_height: int = 960
    cam_fovy: float = 55.0            # deg
    # ---------------- ArUco tags ----------------
    aruco_dict: str = "DICT_4X4_50"
    hook_tag_id: int = 7
    port_tag_id: int = 11
    tag_size: float = 0.044           # black square edge
    tag_border: float = 0.003         # white border printed round it
    tag_thickness: float = 0.0005
    # ---------------- J-hook plate (slot frame, metres) ----------------
    plate_thickness: float = 0.008
    slot_width: float = 0.028         # 4 mm clearance either side of the neck
    slot_slide: float = 0.085         # rest position -> stem centre
    slot_stem: float = 0.060          # stem height above the channel top
    slot_tip: tuple = (0.024, 0.022)  # J tip pocket: width, rise above channel
    plate_margin: float = 0.040
    hook_tag_yz: tuple = (0.0415, 0.048)   # tag centre in the slot frame
    exit_clear: float = 0.010         # flange clearance over the plate on exit
    # ---------------- port ----------------
    hole_radius: float = 0.0105       # 0.5 mm radial clearance
    hole_depth: float = 0.075
    chamfer: float = 0.004            # 45 deg x 4 mm
    floor_thickness: float = 0.008
    boss_radius: float = 0.035
    n_segments: int = 48
    port_hole_below_tag: float = 0.070     # hole axis this far below the tag
    port_plate_size: tuple = (0.24, 0.24)  # width, height of the mount plate
    port_plate_thickness: float = 0.012
    # ---------------- nominal layout in the base frame ----------------
    # Side-on: the hook sits on the robot's right (-y), the port on its left
    # (+y), and both faces point along the rover's fore-aft axis (+x), so the
    # housing lies tangentially to the arm. Turning joint 1 by 180 deg with
    # the wrist fixed carries a housing pulled out of the hook (plug along
    # +x) to the port side with its plug along -x, into the bore.
    hook_face: tuple = (1.0, 0.0, 0.0)
    hook_neck_pos: tuple = (-0.044, -0.30, 0.13)   # resting neck centre
    port_face: tuple = (1.0, 0.0, 0.0)
    port_hole_pos: tuple = (-0.035, 0.30, 0.13)    # centre of the opening
    # Arm start pose (joint angles)
    home_q: tuple = (0.0, 2.2, 2.6, 0.4, 0.0, 0.0)


P = Params()


# ------------------------------------------------------------- rotations
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tilt_matrix(roll, pitch, yaw):
    """Fixture tilt (radians) in the fixture's own nominal frame:
    R = Rz(yaw) Ry(pitch) Rx(roll). roll turns the plate in its own plane
    (about the face normal x), pitch tips the face up/down (about y) and yaw
    turns it left/right (about z)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def mat_to_quat(R):
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_angle(R):
    """Angle of a rotation matrix [rad]."""
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


def rotvec(R):
    """Rotation vector (axis * angle) of a rotation matrix."""
    a = rot_angle(R)
    if a < 1e-9:
        return np.zeros(3)
    if a > math.pi - 1e-6:
        # axis from the symmetric part
        M = (R + np.eye(3)) / 2
        ax = np.sqrt(np.clip(np.diag(M), 0, None))
        k = int(np.argmax(ax))
        ax[k] = math.copysign(ax[k], 1)
        for j in range(3):
            if j != k:
                ax[j] = math.copysign(ax[j], M[k, j])
        return ax / np.linalg.norm(ax) * a
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * math.sin(a)) * a


def exp_rot(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class Pose:
    """Rigid transform: p_parent = R @ p_child + t."""

    __slots__ = ("R", "t")

    def __init__(self, R=None, t=None):
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.t = np.zeros(3) if t is None else np.asarray(t, float)

    def __matmul__(self, other):
        if isinstance(other, Pose):
            return Pose(self.R @ other.R, self.R @ other.t + self.t)
        return self.R @ np.asarray(other, float) + self.t

    def inv(self):
        return Pose(self.R.T, -self.R.T @ self.t)

    def quat(self):
        return mat_to_quat(self.R)

    def copy(self):
        return Pose(self.R.copy(), self.t.copy())

    def __repr__(self):
        return f"Pose(t={np.round(self.t, 4)}, rpy?)"


# Fixture frame -> tag frame (tag x = fixture y, tag y = fixture z, tag z =
# fixture x). Columns are the tag axes in fixture coordinates.
R_FIX_TAG = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
# TCP -> camera (OpenCV): cam z (forward) = -tcp x, cam y (down) = -tcp z,
# cam x (right) = cam y x cam z.
R_TCP_CAM = np.column_stack([
    np.cross([0, 0, -1.0], [-1.0, 0, 0]), [0, 0, -1.0], [-1.0, 0, 0]])


def nominal_fixture_R(normal_towards_base):
    """Fixture frame whose x axis is `normal_towards_base` (horizontal) and z
    is world up."""
    x = np.asarray(normal_towards_base, float)
    x = x / np.linalg.norm(x)
    z = np.array([0, 0, 1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def hook_nominal(p: Params = P) -> Pose:
    """Nominal hook (tag) frame in the base frame."""
    R = nominal_fixture_R(p.hook_face)
    return Pose(R, np.asarray(p.hook_neck_pos) - R @ slot_in_hook(p).t)


def port_nominal(p: Params = P) -> Pose:
    """Nominal port (tag) frame in the base frame."""
    R = nominal_fixture_R(p.port_face)
    return Pose(R, np.asarray(p.port_hole_pos) - R @ hole_in_port(p).t)


def tilted(T_nominal: Pose, roll, pitch, yaw) -> Pose:
    """Fixture tilted about its own origin (the tag centre)."""
    return Pose(T_nominal.R @ tilt_matrix(roll, pitch, yaw), T_nominal.t)


# ------------------------------------------------------------- housing
def housing_sections(p: Params = P):
    """(name, radius, s0, s1, rgba) along the housing from the plug tip (s=0).
    Returns (sections, s_neck_centre, s_groove_centre, s_end)."""
    green = [0.10, 0.70, 0.35, 1]
    grey = [0.45, 0.48, 0.52, 1]
    rear_len = rear_length(p)
    spec = [("plug", p.plug_radius, p.plug_length, [0.05, 0.05, 0.05, 1]),
            ("groove", p.groove_radius, p.groove_length, [0.22, 0.22, 0.24, 1]),
            ("spacer", p.plug_radius, p.spacer_length, grey),
            ("flange_front", p.flange_radius, p.flange_thickness, green),
            ("neck", p.neck_radius, p.neck_length, grey),
            ("flange_rear", p.flange_radius, p.flange_thickness, grey),
            ("rear", p.rear_radius, rear_len, [0.12, 0.12, 0.13, 1])]
    secs, s = [], 0.0
    for name, r, L, rgba in spec:
        secs.append((name, r, s, s + L, rgba))
        s += L
    neck = next(x for x in secs if x[0] == "neck")
    groove = next(x for x in secs if x[0] == "groove")
    return secs, (neck[2] + neck[3]) / 2, (groove[2] + groove[3]) / 2, s


def rear_length(p: Params = P):
    """Length of the rear section that puts the centre of mass in the middle
    of the neck, so the housing rests level in the hook."""
    a = math.pi
    front = [(p.flange_radius, p.flange_thickness),
             (p.plug_radius, p.spacer_length),
             (p.groove_radius, p.groove_length),
             (p.plug_radius, p.plug_length)]
    s = p.neck_length / 2
    moment = 0.0
    for r, L in front:
        moment += a * r * r * L * (s + L / 2)
        s += L
    s0 = p.neck_length / 2
    moment -= a * p.flange_radius ** 2 * p.flange_thickness * (
        s0 + p.flange_thickness / 2)
    s1 = s0 + p.flange_thickness
    A = a * p.rear_radius ** 2
    # A * L * (s1 + L/2) = moment
    return (-s1 + math.sqrt(s1 * s1 + 2 * moment / A))


def housing_offsets(p: Params = P):
    """Distances along the housing measured from the groove centre (where
    the TCP grips it) towards the plug: tip, neck centre (= CoM)."""
    _, s_neck, s_groove, s_end = housing_sections(p)
    return dict(tip=s_groove, neck=s_groove - s_neck, end=s_groove - s_end)


def tcp_to_peg(p: Params = P) -> Pose:
    """Pose of the housing body in the TCP frame for the nominal grasp
    (groove centred between the pads, plug towards tcp -x)."""
    off = housing_offsets(p)
    # peg local z (tip -> rear) = tcp +x ; choose peg local x = tcp z.
    R = np.column_stack([[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]])
    # check right-handed: x=(0,0,1), y=(0,-1,0), z=(1,0,0): x cross y =
    # (0*0-1*-1, ...) = (1,0,0) = z  -> ok
    return Pose(R, np.array([-off["neck"], 0, 0]))


def tip_in_tcp(p: Params = P):
    return np.array([-housing_offsets(p)["tip"], 0, 0])


# ------------------------------------------------------------- J slot
def slot_key(p: Params = P):
    """Key coordinates of the J slot in the slot frame (y, z)."""
    s, L = p.slot_width, p.slot_slide
    tw, th = p.slot_tip
    z_bot = -p.neck_radius
    return dict(z_bot=z_bot, z_chan=z_bot + s, z_top=z_bot + s + p.slot_stem,
                y_tip=tw / 2, z_tip=z_bot + s + th,
                y_stem_l=L - s / 2, y_stem_r=L + s / 2)


def slot_boxes(p: Params = P):
    """(y0, y1, z0, z1) boxes in the slot frame making up the plate minus the
    J slot. The tip pocket is at y = 0 (left), the stem at y = +slide."""
    k = slot_key(p)
    tw = k["y_tip"]
    ymin, ymax = -tw - p.plate_margin, k["y_stem_r"] + p.plate_margin
    zb, zc, zt = k["z_bot"], k["z_chan"], k["z_top"]
    return [
        (ymin, ymax, zb - p.plate_margin, zb),   # below the channel
        (ymin, -tw, zb, zt),                     # left of the tip pocket
        (k["y_stem_r"], ymax, zb, zt),           # right of the stem
        (tw, k["y_stem_l"], zc, zt),             # between the tip and the stem
        (-tw, tw, k["z_tip"], zt),               # above the tip pocket
    ]


def neck_free_region(p: Params = P):
    """Where the neck centre can be inside the slot (slot y, z), as a list of
    axis-aligned rectangles (y0, y1, z0, z1)."""
    k = slot_key(p)
    r = p.neck_radius
    return [
        (-k["y_tip"] + r, k["y_tip"] - r, k["z_bot"] + r, k["z_tip"] - r),
        (-k["y_tip"] + r, k["y_stem_r"] - r, k["z_bot"] + r, k["z_chan"] - r),
        (k["y_stem_l"] + r, k["y_stem_r"] - r, k["z_bot"] + r, 1.0),
    ]


def slot_in_hook(p: Params = P) -> Pose:
    """Slot frame expressed in the hook (tag) frame."""
    ty, tz = p.hook_tag_yz
    return Pose(np.eye(3), np.array(
        [-(p.plate_thickness / 2 + p.tag_thickness), -ty, -tz]))


def lift_out_path(y_neck, p: Params = P, extra_up=0.03):
    """Neck-centre waypoints (slot frame, 3D) that lead the housing out of
    the J: up to the middle of the channel, along to the stem, up the stem
    until both flanges clear the top of the plate, then a little higher. The
    whole housing is then above the plate (the flanges are its widest part),
    so the 180 deg swing can carry it away along its own axis."""
    k = slot_key(p)
    z_mid = (k["z_bot"] + k["z_chan"]) / 2
    z_exit = k["z_top"] + p.flange_radius + p.exit_clear
    return [
        ("lift", np.array([0.0, y_neck, z_mid])),
        ("slide", np.array([0.0, p.slot_slide, z_mid])),
        ("up the stem", np.array([0.0, p.slot_slide, z_exit])),
        ("clear", np.array([0.0, p.slot_slide, z_exit + extra_up])),
    ]


# ------------------------------------------------------------- port
def hole_in_port(p: Params = P) -> Pose:
    """Hole frame in the port (tag) frame: origin at the centre of the
    opening, x pointing INTO the bore (= -port x), z = port z."""
    boss_len = p.floor_thickness + p.hole_depth
    o = np.array([-p.tag_thickness + boss_len, 0, -p.port_hole_below_tag])
    R = np.column_stack([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    return Pose(R, o)


# ------------------------------------------------------------- camera
def camera_intrinsics(p: Params = P):
    f = p.cam_height / 2 / math.tan(math.radians(p.cam_fovy) / 2)
    return np.array([[f, 0, (p.cam_width - 1) / 2],
                     [0, f, (p.cam_height - 1) / 2], [0, 0, 1.0]])


def tcp_to_cam(p: Params = P) -> Pose:
    return Pose(R_TCP_CAM, np.array(p.cam_pos_tcp))
