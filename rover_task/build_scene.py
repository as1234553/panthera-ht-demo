"""Build the rover cable-plugging scene for the Panthera-HT (MuJoCo).

Starts from the official HighTorque MuJoCo model (third_party/Panthera-HT_SDK)
and adds:
  * prismatic, mimic-coupled fingers with friction pads and a side-grasp TCP
    (as in peg_in_hole), plus box collision proxies for the fin-ray fingers so
    they cannot pass through the fixtures,
  * a wrist camera on the gripper, looking along the direction the plug
    points,
  * the cable housing on a free joint: plug, grip groove, spacer, front
    flange, neck, rear flange and a rear strain relief that balances it,
  * the J-hook: a plate with a J-shaped slot and an ArUco tag between the tip
    of the J and its stem, on the robot's right, front face towards the base,
  * the port: a chamfered blind bore in a boss on a mounting plate, with an
    ArUco tag above it, on the robot's left (180 deg round from the hook).

The hook and the port are static bodies whose origins are their tags, so the
Monte Carlo can tilt them about the tag by writing model.body_pos/body_quat.

Run:  python rover_task/build_scene.py   -> rover_task/scene.xml (+ tag PNGs)
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geometry import (P, Params, R_FIX_TAG, hook_nominal, housing_sections,  # noqa: E402
                      mat_to_quat, port_nominal, slot_boxes, slot_in_hook,
                      slot_key)

SDK_MJ_DIR = os.path.join(
    HERE, "..", "third_party", "Panthera-HT_SDK", "panthera_python",
    "Panthera-HT_description_V1.1.0", "mujoco")
OUT_XML = os.path.join(HERE, "scene.xml")
ASSETS = os.path.join(HERE, "assets")

# Collision bitmasks: contact if (typeA & affB) or (typeB & affA).
ENV, ROBOT, PEG, PAD, FINGER = 1, 2, 4, 8, 16
PAD_Z = 0.004          # finger pads' centre along the gripper's z axis
R_G_TCP = np.array([[0.0, 0, -1], [0, 1, 0], [1, 0, 0]])   # TCP axes in gripper
P_G_TCP = np.array([-P.grip_depth, 0, PAD_Z])

# Finger collision proxies in the TCP frame for the LEFT finger at the
# reference opening q_L = -0.033 (gripping the 16 mm groove): (x0, x1, y0, y1,
# z0, z1) from slices of the SDK finger mesh. The right finger is mirrored.
FINGER_REF_Q = -0.033
FINGER_BOXES = [
    (-0.018, 0.010, 0.0085, 0.032, -0.022, 0.030),   # fin-ray finger, lower
    (-0.031, 0.018, 0.0095, 0.043, 0.030, 0.075),    # fin-ray finger, upper
    (-0.074, 0.018, 0.010, 0.043, 0.075, 0.105),     # carriage horn
]


def make_tag_png(tag_id, p: Params, path, px_per_cell=80):
    """ArUco marker with a white border, as an RGB PNG. The texture covers the
    full printed square: black marker (6 cells for 4x4) plus border."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, p.aruco_dict))
    cells = d.markerSize + 2
    img = cv2.aruco.generateImageMarker(d, tag_id, cells * px_per_cell)
    b = int(round(p.tag_border / p.tag_size * cells * px_per_cell))
    full = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, cv2.cvtColor(full, cv2.COLOR_GRAY2BGR))
    return path


def sector_vertices(r_in, r_out, z0, z1, chamfer, th0, th1, n):
    """One convex sector of a chamfered bore wall (bore axis = z, opening at
    z1); flat faces of the n-gon sit exactly at r_in."""
    k = 1.0 / math.cos(math.pi / n)
    profile = [(r_in, z0), (r_out, z0), (r_out, z1),
               (r_in + chamfer, z1), (r_in, z1 - chamfer)]
    verts = []
    for th in (th0, th1):
        c, s = math.cos(th), math.sin(th)
        for r, z in profile:
            verts += [k * r * c, k * r * s, z]
    return verts


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return list(np.concatenate([x, y]))


def add_robot(spec: mujoco.MjSpec, p: Params):
    for g in spec.geoms:
        if g.contype == 2 and g.conaffinity == 1:          # robot link hulls
            g.contype, g.conaffinity = ROBOT, ENV | PEG
    floor = spec.geom("floor")
    floor.contype, floor.conaffinity = ENV, ROBOT | PEG | PAD
    floor.friction = [1.0, 0.005, 0.0001]
    floor.pos = [0, 0, 0]

    gc = spec.body("gripper_center_link")
    gc.add_site(name="tcp", pos=list(P_G_TCP), size=[0.004] * 3,
                quat=list(mat_to_quat(R_G_TCP)), rgba=[1, 0, 0, 0.6], group=4)
    for side, sgn in (("L", +1), ("R", -1)):
        body = spec.body(f"gripper_{side}_link")
        for g in body.geoms:
            if g.contype:
                g.contype = g.conaffinity = 0
        rng = [-p.finger_stroke, 0] if sgn > 0 else [0, p.finger_stroke]
        body.add_joint(name=f"gripper_{side}_joint",
                       type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0],
                       range=rng, damping=8.0, armature=0.01)
        body.add_geom(name=f"pad_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[-0.0172, sgn * 0.004, PAD_Z],
                      size=[0.0225, 0.004, 0.016],
                      contype=PAD, conaffinity=ENV | PEG, condim=4,
                      friction=[1.5, 0.02, 0.0005],
                      solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                      rgba=[0.15, 0.15, 0.15, 1], group=1, density=0)
        # Finger proxies: collide with the environment only (never with the
        # housing, which only the pads touch). Body frame = gripper frame
        # shifted by (-0.0003, sgn*(0.041 + q), 0); at the reference q the
        # TCP-frame box maps to body coordinates as below.
        y_body0 = sgn * (0.041 + FINGER_REF_Q)
        for i, (x0, x1, y0, y1, z0, z1) in enumerate(FINGER_BOXES):
            c_tcp = np.array([(x0 + x1) / 2, sgn * (y0 + y1) / 2, (z0 + z1) / 2])
            c_g = P_G_TCP + R_G_TCP @ c_tcp
            c_body = c_g - np.array([-0.0003, y_body0, 0])
            half_tcp = np.array([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2])
            half_g = np.abs(R_G_TCP) @ half_tcp
            body.add_geom(name=f"finger_{side}{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=list(c_body), size=list(half_g),
                          contype=FINGER, conaffinity=ENV, group=3,
                          rgba=[1, 0.3, 0.3, 0.3], density=0)

    eq = spec.add_equality(name="gripper_mimic")
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.name1, eq.name2 = "gripper_R_joint", "gripper_L_joint"
    eq.data[:5] = [0, -1, 0, 0, 0]
    eq.solref = [0.005, 1]
    act = spec.add_actuator(name="gripper", target="gripper_L_joint",
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=3000.0)
    act.ctrlrange = [-p.finger_stroke, 0.0]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-p.grip_force, p.grip_force]

    # Wrist camera: looks along tcp -x (where the plug points), image up =
    # tcp +z. MuJoCo cameras look along their -z with y up.
    c_tcp = np.array(p.cam_pos_tcp)
    cam_pos = P_G_TCP + R_G_TCP @ c_tcp
    x_mj = R_G_TCP @ np.array([0, 1.0, 0])
    y_mj = R_G_TCP @ np.array([0, 0, 1.0])
    gc.add_camera(name="wrist", pos=list(cam_pos),
                  xyaxes=list(np.r_[x_mj, y_mj]), fovy=p.cam_fovy)
    body_c = P_G_TCP + R_G_TCP @ (c_tcp + np.array([0.013, 0, 0]))
    gc.add_geom(name="wrist_cam_body", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(body_c), quat=list(mat_to_quat(R_G_TCP)),
                size=[0.012, 0.018, 0.012], contype=FINGER, conaffinity=ENV,
                rgba=[0.1, 0.1, 0.12, 1], group=1, density=0)
    gc.add_geom(name="wrist_cam_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=list(cam_pos), quat=list(mat_to_quat(
                    R_G_TCP @ np.column_stack([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]]))),
                size=[0.006, 0.002, 0], contype=0, conaffinity=0,
                rgba=[0.2, 0.3, 0.5, 1], group=1, density=0)


def add_housing(spec, p: Params):
    w = spec.worldbody
    peg = w.add_body(name="peg", pos=[0, 0, 0.5])
    peg.add_freejoint(name="peg_free")
    secs, s_neck, s_groove, s_end = housing_sections(p)
    for name, r, s0, s1, rgba in secs:
        peg.add_geom(name=f"peg_{name}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                     pos=[0, 0, (s0 + s1) / 2 - s_neck],
                     size=[r, (s1 - s0) / 2, 0], density=p.peg_density,
                     contype=PEG, conaffinity=ENV | ROBOT | PAD, condim=4,
                     friction=[0.3, 0.005, 0.0001], solref=[0.004, 1],
                     solimp=[0.95, 0.99, 0.0005, 0.5, 2], rgba=rgba)
    peg.add_geom(name="cable", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0, 0, s_end + p.cable_stub / 2 - s_neck],
                 size=[p.cable_radius, p.cable_stub / 2, 0],
                 contype=0, conaffinity=0, density=0, group=1,
                 rgba=[0.95, 0.40, 0.10, 1])
    for name, s in (("peg_tip", 0.0), ("peg_rear", s_end), ("peg_grip", s_groove)):
        peg.add_site(name=name, pos=[0, 0, s - s_neck], size=[0.002] * 3,
                     rgba=[0, 1, 0, 1], group=4)


COLLIDE = dict(contype=ENV, conaffinity=ROBOT | PEG | PAD, condim=3,
               friction=[0.3, 0.005, 0.0001], solref=[0.004, 1],
               solimp=[0.95, 0.99, 0.0005, 0.5, 2], group=3,
               rgba=[0.5, 0.5, 0.5, 0.3])
VISUAL = dict(contype=0, conaffinity=0, density=0)


def add_tag(spec, body, name, tag_id, p: Params):
    png = make_tag_png(tag_id, p, os.path.join(ASSETS, f"{name}.png"))
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=os.path.relpath(png, HERE))
    mat = spec.add_material(name=name)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    half = p.tag_size / 2 + p.tag_border
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[-p.tag_thickness / 2, 0, 0],
                  quat=list(mat_to_quat(R_FIX_TAG)),
                  size=[half, half, p.tag_thickness / 2], material=name,
                  group=2, **VISUAL)


def add_hook(spec, p: Params):
    """J-hook plate, body origin at the tag centre."""
    T0 = hook_nominal(p)
    hk = spec.worldbody.add_body(name="hook", pos=list(T0.t),
                                 quat=list(mat_to_quat(T0.R)))
    S = slot_in_hook(p)
    rgba = dict(rgba=[0.74, 0.74, 0.71, 1])
    for i, (y0, y1, z0, z1) in enumerate(slot_boxes(p)):
        c = S @ np.array([0, (y0 + y1) / 2, (z0 + z1) / 2])
        box = dict(type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(c),
                   size=[p.plate_thickness / 2, (y1 - y0) / 2, (z1 - z0) / 2])
        hk.add_geom(name=f"hook{i}", **box, **{**COLLIDE, **rgba})
        hk.add_geom(name=f"hook{i}_vis", **box, **{**VISUAL, **rgba}, group=2)
    add_tag(spec, hk, "hook_tag", p.hook_tag_id, p)
    # dark backing behind the slot so the camera sees a dark J
    k = slot_key(p)
    yb0, yb1 = -k["y_tip"] - 0.01, k["y_stem_r"] + 0.01
    zb0, zb1 = k["z_bot"] - 0.01, k["z_top"]
    c = S @ np.array([-0.075, (yb0 + yb1) / 2, (zb0 + zb1) / 2])
    hk.add_geom(name="hook_backing", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=list(c), size=[0.002, (yb1 - yb0) / 2, (zb1 - zb0) / 2],
                rgba=[0.25, 0.25, 0.27, 1], group=2, **VISUAL)


def add_port(spec, p: Params):
    """Port: mounting plate with the tag, and a boss with the chamfered bore
    below it. Body origin at the tag centre, x towards the robot."""
    T0 = port_nominal(p)
    pt = spec.worldbody.add_body(name="port", pos=list(T0.t),
                                 quat=list(mat_to_quat(T0.R)))
    W, H = p.port_plate_size
    tz = -p.port_hole_below_tag / 2
    plate = dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[-p.tag_thickness - p.port_plate_thickness / 2, 0, tz],
                 size=[p.port_plate_thickness / 2, W / 2, H / 2])
    prgba = dict(rgba=[0.62, 0.64, 0.68, 1])
    pt.add_geom(name="port_plate", **plate, **{**COLLIDE, **prgba})
    pt.add_geom(name="port_plate_vis", **plate, **{**VISUAL, **prgba}, group=2)
    add_tag(spec, pt, "port_tag", p.port_tag_id, p)

    # Bore: child body, local +z = port +x (out of the opening).
    bore = pt.add_body(name="bore",
                       pos=[-p.tag_thickness, 0, -p.port_hole_below_tag],
                       quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
    z1 = p.floor_thickness + p.hole_depth
    vis = dict(**VISUAL, rgba=[0.78, 0.80, 0.84, 1])
    floor_kw = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[0, 0, p.floor_thickness / 2],
                    size=[p.boss_radius, p.floor_thickness / 2, 0])
    bore.add_geom(name="bore_floor", **floor_kw, **COLLIDE)
    bore.add_geom(name="bore_floor_vis", **floor_kw, **vis, group=2)
    n = p.n_segments
    for k in range(n):
        th0, th1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        mesh = spec.add_mesh(name=f"bore_seg{k}")
        mesh.uservert = sector_vertices(p.hole_radius, p.boss_radius,
                                        p.floor_thickness, z1, p.chamfer,
                                        th0, th1, n)
        kw = dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"bore_seg{k}")
        bore.add_geom(name=f"bore_seg{k}", **kw, **COLLIDE)
        bore.add_geom(name=f"bore_seg{k}_vis", **kw, **vis, group=2)
        # section view: keep the half with local x <= 0 (port z >= 0 side
        # removed), i.e. cut through the bore axis on a horizontal plane
        if math.pi / 2 - 1e-9 <= th0 and th1 <= 3 * math.pi / 2 + 1e-9:
            bore.add_geom(name=f"bore_seg{k}_sec", **kw, **vis, group=5)
    bore.add_site(name="hole_top", pos=[0, 0, z1], size=[0.002] * 3,
                  rgba=[1, 1, 0, 1], group=4)
    bore.add_site(name="hole_bottom", pos=[0, 0, p.floor_thickness],
                  size=[0.002] * 3, rgba=[1, 0, 1, 1], group=4)


def build(p: Params = P) -> mujoco.MjSpec:
    os.makedirs(ASSETS, exist_ok=True)
    spec = mujoco.MjSpec.from_file(os.path.join(SDK_MJ_DIR, "scene.xml"))
    spec.modelname = "panthera_ht_rover_cable"
    spec.meshdir = os.path.relpath(SDK_MJ_DIR, HERE)
    spec.texturedir = "."
    spec.modelfiledir = HERE
    spec.option.timestep = 0.001
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = max(1920, p.cam_width)
    spec.visual.global_.offheight = max(1080, p.cam_height)

    add_robot(spec, p)
    add_housing(spec, p)
    add_hook(spec, p)
    add_port(spec, p)

    w = spec.worldbody
    cams = {
        "overview": ([0.95, 0.05, 0.75], [0.0, 0.05, 0.12], 50),
        "hook_view": ([0.35, -0.05, 0.42], [0.0, -0.33, 0.13], 45),
        "port_view": ([0.38, 0.12, 0.42], [0.0, 0.44, 0.17], 45),
    }
    for name, (cpos, tgt, fovy) in cams.items():
        w.add_camera(name=name, pos=cpos, xyaxes=lookat_xyaxes(cpos, tgt),
                     fovy=fovy)
    spec.material("groundplane").reflectance = 0.05
    w.add_light(name="key2", pos=[0.6, 0.0, 1.2], dir=[-0.4, 0.0, -1],
                diffuse=[0.45, 0.45, 0.45], castshadow=False)
    w.add_light(name="fill2", pos=[-0.4, 0.0, 0.9], dir=[0.3, 0.0, -1],
                diffuse=[0.3, 0.3, 0.3], castshadow=False)
    return spec


def main():
    spec = build(P)
    model = spec.compile()
    q = model.qpos0.copy()
    q[:6] = P.home_q
    for k in list(spec.keys):
        spec.delete(k)
    spec.add_key(name="home", qpos=list(q))
    model = spec.compile()
    with open(OUT_XML, "w") as f:
        f.write(spec.to_xml())
    _, s_neck, s_groove, s_end = housing_sections(P)
    mass = model.body_subtreemass[model.body("peg").id]
    com = model.body_ipos[model.body("peg").id]
    print(f"wrote {OUT_XML}: nq={model.nq} nu={model.nu} ngeom={model.ngeom}")
    print(f"housing: length {s_end * 1e3:.0f} mm, mass {mass * 1e3:.0f} g, "
          f"CoM offset from neck centre {com[2] * 1e3:.2f} mm")


if __name__ == "__main__":
    main()
