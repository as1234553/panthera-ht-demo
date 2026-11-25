"""Build the Panthera-HT peg-in-hole MuJoCo scene (J-hook + horizontal hole).

Starts from the official HighTorque MuJoCo model (third_party/Panthera-HT_SDK)
and adds:
  * the parallel gripper as two prismatic fingers (mimic-coupled, as documented
    in the SDK's urdf/README.md) with box friction pads, and a TCP for a side
    grasp: between the pads, x along the gripped cylinder,
  * a cable housing on a free joint (the grip adapter on the end of a long
    cable): a plug, two flanges with a neck between them, and a grip groove,
  * a J-hook: a vertical plate with a J-shaped slot that the housing's neck
    rests in, the flanges either side of the plate,
  * a fixture whose blind bore is horizontal (opening facing the robot) with a
    45 deg chamfer at the opening, on a vertical mounting plate.

MuJoCo only collides convex shapes, so the bore wall is built from N convex
annular sectors. Each sector's r-z profile is a rectangle with one corner cut
off by the chamfer, which keeps it convex.

Run:  python peg_in_hole/build_scene.py      -> writes peg_in_hole/scene.xml
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SDK_MJ_DIR = os.path.join(
    HERE, "..", "third_party", "Panthera-HT_SDK", "panthera_python",
    "Panthera-HT_description_V1.1.0", "mujoco")
OUT_XML = os.path.join(HERE, "scene.xml")

# Collision bitmasks: contact if (typeA & affB) or (typeB & affA).
ENV, ROBOT, PEG, PAD = 1, 2, 4, 8


@dataclass(frozen=True)
class Params:
    # Cable housing (aluminium grip adapter on the end of a long cable). Along
    # its axis from the plug tip: plug, front flange, neck (sits in the J-hook
    # slot), rear flange, shaft, grip groove (where the fingers close), rear
    # shaft, then the cable out of the back. The lengths put the centre of
    # mass in the middle of the neck, so the housing rests level in the hook.
    peg_radius: float = 0.010        # plug, neck and shaft radius
    plug_length: float = 0.082       # 75 mm in the bore + 7 mm to the flange
    flange_radius: float = 0.019
    flange_thickness: float = 0.008
    neck_length: float = 0.012       # hook plate + 2 mm each side
    grip_offset: float = 0.025       # rear flange -> grip groove
    grip_length: float = 0.036       # finger pads are 32 mm wide
    grip_radius: float = 0.008       # groove 2 mm deep
    rear_length: float = 0.028       # grip groove -> rear end
    cable_radius: float = 0.004      # visual stub only
    cable_stub: float = 0.060
    peg_density: float = 2700.0
    # Side grasp: the TCP sits between the finger pads, this far in from the
    # fingertips, with TCP x along the housing (towards the plug) and TCP z
    # pointing back up the gripper. TCP quat identity = gripper pointing down.
    grip_depth: float = 0.017
    # Start pose of the TCP with the housing in hand (--start hand)
    start_tcp: tuple = (0.20, 0.10, 0.14)
    # Hole: centre of the opening; the bore axis runs along world +x. Low and
    # close, because the arm can only point the gripper straight down below
    # about z = 0.17 m and within about 0.43 m of the base.
    hole_open: tuple = (0.41, 0.0, 0.12)
    hole_radius: float = 0.0105      # 0.5 mm radial clearance
    hole_depth: float = 0.075
    chamfer: float = 0.004           # 45 deg x 4 mm
    floor_thickness: float = 0.008
    boss_radius: float = 0.035
    n_segments: int = 48
    plate_thickness: float = 0.012
    plate_half_width: float = 0.10
    plate_height: float = 0.26
    # Gripper
    finger_stroke: float = 0.041
    finger_open: float = 0.029       # inner faces from centre when open
    grip_force: float = 25.0
    # J-hook (cable holder): a vertical plate with a J-shaped slot, off to
    # the robot's left. The housing's neck rests at the bottom of the J,
    # perpendicular to the plate, with a flange either side of it; the stem of
    # the J is on the robot's right and open at the top.
    hook_yaw: float = 60.0           # deg, direction of the hook from the base
    hook_radius: float = 0.34        # resting neck centre: distance from base
    hook_height: float = 0.05        # resting neck centre: height
    hook_slot: float = 0.028         # slot width (4 mm clearance per side)
    hook_slide: float = 0.085        # sideways, resting position -> stem
    hook_stem: float = 0.060         # stem length above the channel
    hook_tip: tuple = (0.024, 0.022)  # J tip: width, rise above the channel
    hook_thickness: float = 0.008
    hook_margin: float = 0.040       # plate border around the slot
    hook_exit_clear: float = 0.010   # flange clearance above the plate on exit
    # Arm start pose for the hook task, relative to the grasp pose
    home_back: float = 0.04
    home_up: float = 0.08


P = Params()
PAD_Z = 0.004          # finger pads' centre along the gripper's z axis


def housing_sections(p: Params):
    """(name, radius, s0, s1) along the housing axis, s measured from the
    plug tip, plus the arc positions of the neck centre and grip centre."""
    secs, s = [], 0.0
    for name, r, length in (
            ("plug", p.peg_radius, p.plug_length),
            ("flange_front", p.flange_radius, p.flange_thickness),
            ("neck", p.peg_radius, p.neck_length),
            ("flange_rear", p.flange_radius, p.flange_thickness),
            ("shaft", p.peg_radius, p.grip_offset),
            ("grip", p.grip_radius, p.grip_length),
            ("rear", p.peg_radius, p.rear_length)):
        secs.append((name, r, s, s + length))
        s += length
    neck = next(x for x in secs if x[0] == "neck")
    grip = next(x for x in secs if x[0] == "grip")
    return secs, (neck[2] + neck[3]) / 2, (grip[2] + grip[3]) / 2


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    """xyaxes for a camera at `pos` looking at `target` (camera looks down -z)."""
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return list(np.concatenate([x, y]))


def hook_frame(p: Params):
    """Hook frame: origin at the resting cable centre (mid-plane of the
    plate), x along the cable away from the robot, y to the robot's left,
    z up. Returns (origin, quat, x axis)."""
    psi = math.radians(p.hook_yaw)
    n = np.array([math.cos(psi), math.sin(psi), 0.0])
    origin = p.hook_radius * n + [0, 0, p.hook_height]
    return origin, [math.cos(psi / 2), 0, 0, math.sin(psi / 2)], n


def hook_slot(p: Params):
    """Key coordinates of the J slot in the hook frame (y, z)."""
    s, L = p.hook_slot, p.hook_slide
    tw, th = p.hook_tip
    z_bot = -p.peg_radius                 # cable rests on the channel floor
    return dict(z_bot=z_bot, z_chan=z_bot + s, z_top=z_bot + s + p.hook_stem,
                y_tip=tw / 2, z_tip=z_bot + s + th,
                y_stem_l=-(L - s / 2), y_stem_r=-(L + s / 2))


def hook_boxes(p: Params):
    """(y0, y1, z0, z1) boxes in the hook frame that make up the plate minus
    the J slot (channel along the bottom, tip rising on the left, stem rising
    on the right and open at the top)."""
    k = hook_slot(p)
    tw = k["y_tip"]
    ymin, ymax = k["y_stem_r"] - p.hook_margin, tw + p.hook_margin
    zb, zc, zt = k["z_bot"], k["z_chan"], k["z_top"]
    return [
        (ymin, ymax, -p.hook_height, zb),      # below the channel
        (tw, ymax, zb, zt),                    # left of the J tip
        (ymin, k["y_stem_r"], zb, zt),         # right of the stem
        (k["y_stem_l"], -tw, zc, zt),          # between the tip and the stem
        (-tw, tw, k["z_tip"], zt),             # above the J tip
    ]


def sector_vertices(r_in, r_out, z0, z1, chamfer, th0, th1, n):
    """Vertices of one convex sector of a chamfered bore wall (bore axis = z,
    opening at z1). Radii are the inscribed (flat-face) radii of the n-gon, so
    the flat faces of the faceted bore sit exactly at r_in."""
    k = 1.0 / math.cos(math.pi / n)
    profile = [  # (r, z) - convex: rectangle with the top-inner corner cut
        (r_in, z0), (r_out, z0), (r_out, z1),
        (r_in + chamfer, z1), (r_in, z1 - chamfer),
    ]
    verts = []
    for th in (th0, th1):
        c, s = math.cos(th), math.sin(th)
        for r, z in profile:
            verts += [k * r * c, k * r * s, z]
    return verts


def build(p: Params = P) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(os.path.join(SDK_MJ_DIR, "scene.xml"))
    spec.modelname = "panthera_ht_peg_in_hole"
    spec.meshdir = os.path.relpath(SDK_MJ_DIR, HERE)
    spec.modelfiledir = HERE

    # ---------------- options ----------------
    spec.option.timestep = 0.001
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Soft-contact friction otherwise lets the overhanging peg creep (rotate)
    # in the grasp under its own weight; noslip removes that drift.
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # ---------------- robot collision setup ----------------
    for g in spec.geoms:
        if g.contype == 2 and g.conaffinity == 1:          # robot link hulls
            g.contype, g.conaffinity = ROBOT, ENV | PEG
    floor = spec.geom("floor")
    floor.contype, floor.conaffinity = ENV, ROBOT | PEG | PAD
    floor.friction = [1.0, 0.005, 0.0001]
    floor.pos = [0, 0, 0]        # SDK scene has it 2 mm low; table top is z=0

    # ---------------- gripper: prismatic fingers + pads ----------------
    gc = spec.body("gripper_center_link")
    # Side-grasp TCP: between the pads, x along the gripper's z axis (the
    # gripped cylinder's axis), z back along -x (towards the palm).
    gc.add_site(name="tcp", pos=[-p.grip_depth, 0, PAD_Z], size=[0.004] * 3,
                quat=[math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0],
                rgba=[1, 0, 0, 0.6], group=4)
    for side, sgn in (("L", +1), ("R", -1)):
        body = spec.body(f"gripper_{side}_link")
        # The convex hull of each finger mesh is a poor contact surface; use pads.
        for g in body.geoms:
            if g.contype:
                g.contype = g.conaffinity = 0
        rng = [-p.finger_stroke, 0] if sgn > 0 else [0, p.finger_stroke]
        body.add_joint(name=f"gripper_{side}_joint",
                       type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0],
                       range=rng, damping=8.0, armature=0.01)
        # Finger inner face sits at body y = 0 (TCP y = +-41 mm when open).
        body.add_geom(name=f"pad_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[-0.0172, sgn * 0.004, PAD_Z],
                      size=[0.0225, 0.004, 0.016],
                      contype=PAD, conaffinity=ENV | PEG, condim=4,
                      friction=[1.5, 0.02, 0.0005],
                      solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                      rgba=[0.15, 0.15, 0.15, 1], group=1, density=0)

    eq = spec.add_equality(name="gripper_mimic")
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.name1, eq.name2 = "gripper_R_joint", "gripper_L_joint"
    eq.data[:5] = [0, -1, 0, 0, 0]                 # q_R = -q_L
    eq.solref = [0.005, 1]

    act = spec.add_actuator(name="gripper", target="gripper_L_joint",
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=3000.0)
    act.ctrlrange = [-p.finger_stroke, 0.0]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-p.grip_force, p.grip_force]

    # ------- cable housing ("peg"): axis = local z, plug tip at local -z,
    # origin at the middle of the neck (= centre of mass) -------
    w = spec.worldbody
    peg = w.add_body(name="peg", pos=[0, 0, p.flange_radius])
    peg.add_freejoint(name="peg_free")
    secs, s_neck, s_grip = housing_sections(p)
    for name, r, s0, s1 in secs:
        peg.add_geom(name=f"peg_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     pos=[0, 0, (s0 + s1) / 2 - s_neck],
                     size=[r, (s1 - s0) / 2, 0],
                     density=p.peg_density, contype=PEG,
                     conaffinity=ENV | ROBOT | PAD, condim=4,
                     friction=[0.3, 0.005, 0.0001], solref=[0.004, 1],
                     solimp=[0.95, 0.99, 0.0005, 0.5, 2],
                     rgba=([0.2, 0.2, 0.22, 1] if name == "grip"
                           else [0.85, 0.45, 0.10, 1]))
    s_end = secs[-1][3]
    peg.add_geom(name="cable", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0, 0, s_end + p.cable_stub / 2 - s_neck],
                 size=[p.cable_radius, p.cable_stub / 2, 0],
                 contype=0, conaffinity=0, density=0, group=1,
                 rgba=[0.05, 0.05, 0.05, 1])
    for name, s, rgba in (("peg_tip", 0.0, [0, 1, 0, 1]),
                          ("peg_rear", s_end, [0, 0, 1, 1]),
                          ("peg_grip", s_grip, [1, 0, 1, 1])):
        peg.add_site(name=name, pos=[0, 0, s - s_neck], size=[0.002] * 3,
                     rgba=rgba, group=4)

    # ---------------- hole fixture ----------------
    # Fixture local frame: bore axis = local +z pointing OUT of the opening.
    # Rotating by -90 deg about y maps local z -> world -x (opening faces the
    # robot), local x -> world +z, local y -> world +y.
    # Geom groups: 3 = collision (hidden), 2 = full visual, 5 = section-view
    # visual (far half only, cut on the vertical plane through the bore axis).
    z_top = p.floor_thickness + p.hole_depth
    hx, hy, hz = p.hole_open
    fx = w.add_body(name="fixture", pos=[hx + z_top, hy, hz],
                    quat=[math.cos(-math.pi / 4), 0, math.sin(-math.pi / 4), 0])
    collide = dict(contype=ENV, conaffinity=ROBOT | PEG | PAD, condim=3,
                   friction=[0.3, 0.005, 0.0001], solref=[0.004, 1],
                   solimp=[0.95, 0.99, 0.0005, 0.5, 2], group=3,
                   rgba=[0.5, 0.5, 0.5, 0.3])
    visual = dict(contype=0, conaffinity=0, density=0,
                  rgba=[0.62, 0.66, 0.72, 1])
    floor_kw = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[0, 0, p.floor_thickness / 2],
                    size=[p.boss_radius, p.floor_thickness / 2, 0])
    fx.add_geom(name="bore_floor", **floor_kw, **collide)
    fx.add_geom(name="bore_floor_vis", **floor_kw, **visual, group=2)
    fx.add_geom(name="bore_floor_sec", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0, p.boss_radius / 2, p.floor_thickness / 2],
                size=[p.boss_radius, p.boss_radius / 2, p.floor_thickness / 2],
                **visual, group=5)
    n = p.n_segments
    for k in range(n):
        th0, th1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        mesh = spec.add_mesh(name=f"bore_seg{k}")
        mesh.uservert = sector_vertices(p.hole_radius, p.boss_radius,
                                        p.floor_thickness, z_top, p.chamfer,
                                        th0, th1, n)
        kw = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bore_seg{k}")
        fx.add_geom(name=f"bore_seg{k}", **kw, **collide)
        fx.add_geom(name=f"bore_seg{k}_vis", **kw, **visual, group=2)
        if th1 <= math.pi + 1e-9:          # local y >= 0  <=>  world y >= 0
            fx.add_geom(name=f"bore_seg{k}_sec", **kw, **visual, group=5)
    fx.add_site(name="hole_top", pos=[0, 0, z_top], size=[0.002] * 3,
                rgba=[1, 1, 0, 1], group=4)
    fx.add_site(name="hole_bottom", pos=[0, 0, p.floor_thickness],
                size=[0.002] * 3, rgba=[1, 0, 1, 1], group=4)

    # Vertical mounting plate behind the boss, standing on the table.
    t, W, H = p.plate_thickness, p.plate_half_width, p.plate_height
    px = hx + z_top + t / 2
    plate_rgba = dict(rgba=[0.35, 0.38, 0.42, 1])
    w.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[px, hy, H / 2], size=[t / 2, W, H / 2],
               **{**collide, **plate_rgba})
    w.add_geom(name="plate_vis", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[px, hy, H / 2], size=[t / 2, W, H / 2],
               **{**visual, **plate_rgba}, group=2)
    w.add_geom(name="plate_sec", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[px, hy + W / 2, H / 2], size=[t / 2, W / 2, H / 2],
               **{**visual, **plate_rgba}, group=5)

    # ---------------- J-hook ----------------
    h_origin, h_quat, n = hook_frame(p)
    hk = w.add_body(name="hook", pos=list(h_origin), quat=h_quat)
    hook_rgba = dict(rgba=[0.74, 0.74, 0.71, 1])
    for i, (y0, y1, z0, z1) in enumerate(hook_boxes(p)):
        box = dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[0, (y0 + y1) / 2, (z0 + z1) / 2],
                   size=[p.hook_thickness / 2, (y1 - y0) / 2, (z1 - z0) / 2])
        hk.add_geom(name=f"hook{i}", **box, **{**collide, **hook_rgba})
        hk.add_geom(name=f"hook{i}_vis", **box, **{**visual, **hook_rgba},
                    group=2)
    # Neck-centre waypoints for lifting the housing out along the J; on exit
    # the flanges (either side of the plate) clear its top edge.
    k = hook_slot(p)
    lift = (k["z_chan"] + k["z_bot"]) / 2           # middle of the channel
    for name, y, z in (
            ("hook_rest", 0.0, 0.0),
            ("hook_lift", 0.0, lift),
            ("hook_corner", -p.hook_slide, lift),
            ("hook_exit", -p.hook_slide,
             k["z_top"] + p.flange_radius + p.hook_exit_clear)):
        hk.add_site(name=name, pos=[0, y, z], size=[0.002] * 3,
                    rgba=[0, 1, 1, 1], group=4)

    for key in ("chamfer", "hole_radius", "hole_depth", "peg_radius",
                "flange_radius", "hook_slot", "hook_slide", "hook_stem",
                "hook_margin"):
        spec.add_numeric(name=key, data=[getattr(p, key)])
    spec.add_numeric(name="hook_tip", data=list(p.hook_tip))

    # ---------------- cameras & lights ----------------
    up = np.array([0, 0, 1.0])
    left = np.cross(up, n)
    j_mid = h_origin - left * p.hook_slide / 2 + up * 0.03   # middle of the J
    hole = np.array(p.hole_open)
    cams = {  # name: (position, look-at target, fovy)
        "overview": ([0.55, -0.85, 0.52], [0.24, 0.08, 0.08], 42),
        "closeup": (list(hole + [-0.20, -0.30, 0.16]),
                    list(hole + [-0.04, 0, 0]), 36),
        "section": ([hx + 0.01, hy - 0.42, hz], [hx + 0.01, hy, hz], 27),
        # robot side, from the right and above (the side the cable leaves)
        "hook_closeup": (list(j_mid - 0.22 * n - 0.24 * left + 0.16 * up),
                         list(j_mid), 45),
        # far side, looking back at the plate along the cable
        "hook_back": (list(j_mid + 0.40 * n + 0.04 * up), list(j_mid), 36),
    }
    for name, (cpos, tgt, fovy) in cams.items():
        w.add_camera(name=name, pos=cpos, xyaxes=lookat_xyaxes(cpos, tgt),
                     fovy=fovy)
    spec.material("groundplane").reflectance = 0.05
    w.add_light(name="key2", pos=[0.6, -0.6, 1.2], dir=[-0.4, 0.4, -1],
                diffuse=[0.5, 0.5, 0.5], castshadow=False)

    return spec


def ready_qpos(model: mujoco.MjModel, p: Params) -> np.ndarray:
    """Arm at the start pose holding the housing horizontally by its grip
    groove, gripper pointing down."""
    from kinematics import ik_pose, quat_mul, axis_angle_quat
    q_tool = np.array([1.0, 0, 0, 0])            # housing along world +x
    q_arm = ik_pose(model, np.array(p.start_tcp), q_tool)
    qpos = model.qpos0.copy()
    qpos[:6] = q_arm
    q_close = -(p.finger_stroke - p.grip_radius)  # pads touching the groove
    qpos[model.jnt_qposadr[model.joint("gripper_L_joint").id]] = q_close
    qpos[model.jnt_qposadr[model.joint("gripper_R_joint").id]] = -q_close
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(model, d)
    sid = model.site("tcp").id
    R = d.site_xmat[sid].reshape(3, 3)
    q_tcp = np.zeros(4)
    mujoco.mju_mat2Quat(q_tcp, d.site_xmat[sid])
    _, s_neck, s_grip = housing_sections(p)
    centre = d.site_xpos[sid] + R[:, 0] * (s_grip - s_neck)   # body origin
    # peg local -z (tip) -> tool +x : rotate -90 deg about tool y
    q_peg = quat_mul(q_tcp, axis_angle_quat([0, 1, 0], -math.pi / 2))
    jadr = model.jnt_qposadr[model.joint("peg_free").id]
    qpos[jadr:jadr + 3] = centre
    qpos[jadr + 3:jadr + 7] = q_peg
    return qpos


def hook_qpos(model: mujoco.MjModel, p: Params) -> np.ndarray:
    """Housing resting in the J-hook; arm back from and above its grip groove
    with the gripper open and pointing down."""
    from kinematics import ik_pose, quat_mul, axis_angle_quat
    origin, q_tool, n = hook_frame(p)
    _, s_neck, s_grip = housing_sections(p)
    grasp = origin - (s_grip - s_neck) * n          # grip groove centre
    home = grasp - p.home_back * n + [0, 0, p.home_up]
    qpos = model.qpos0.copy()
    qpos[:6] = ik_pose(model, home, np.array(q_tool))
    q_open = -(p.finger_stroke - p.finger_open)     # = GRIP_OPEN
    qpos[model.jnt_qposadr[model.joint("gripper_L_joint").id]] = q_open
    qpos[model.jnt_qposadr[model.joint("gripper_R_joint").id]] = -q_open
    jadr = model.jnt_qposadr[model.joint("peg_free").id]
    qpos[jadr:jadr + 3] = origin
    qpos[jadr + 3:jadr + 7] = quat_mul(q_tool,
                                       axis_angle_quat([0, 1, 0], -math.pi / 2))
    return settle_peg(model, qpos)


def settle_peg(model, qpos, T=1.0):
    """Let the cable settle into its contacts with everything else frozen."""
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    jq = model.jnt_qposadr[model.joint("peg_free").id]
    jv = model.jnt_dofadr[model.joint("peg_free").id]
    fq = np.ones(model.nq, bool)
    fq[jq:jq + 7] = False
    fv = np.ones(model.nv, bool)
    fv[jv:jv + 6] = False
    steps = int(round(T / model.opt.timestep))
    for k in range(steps):
        mujoco.mj_step(model, d)
        d.qpos[fq], d.qvel[fv] = qpos[fq], 0
        if k == steps // 2:
            half = d.qpos[jq:jq + 3].copy()
    drop = np.linalg.norm(d.qpos[jq:jq + 3] - qpos[jq:jq + 3])
    creep = np.linalg.norm(d.qpos[jq:jq + 3] - half)
    print(f"housing settled in hook: moved {drop * 1e3:.3f} mm, "
          f"{creep * 1e3:.4f} mm in the last {T / 2:.1f} s")
    return d.qpos.copy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chamfer", type=float, default=P.chamfer,
                    help="chamfer size [m]; 0 builds a plain hole")
    ap.add_argument("--out", default=OUT_XML)
    args = ap.parse_args()
    params = replace(P, chamfer=args.chamfer)
    out_xml = os.path.abspath(args.out)

    spec = build(params)
    model = spec.compile()
    keys = {"hook": hook_qpos(model, params),      # task start: cable in hook
            "ready": ready_qpos(model, params)}    # cable already in hand
    for k in list(spec.keys):
        spec.delete(k)
    for name, qpos in keys.items():
        spec.add_key(name=name, qpos=list(qpos))

    model = spec.compile()
    with open(out_xml, "w") as f:
        f.write(spec.to_xml())
    print(f"wrote {out_xml}: chamfer={params.chamfer * 1e3:.1f} mm "
          f"nq={model.nq} nu={model.nu} ngeom={model.ngeom}")


if __name__ == "__main__":
    main()
