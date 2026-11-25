"""Small kinematics helpers for the Panthera-HT MuJoCo model."""

from __future__ import annotations

import math

import mujoco
import numpy as np

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


def quat_err(q_des, q_cur):
    """World-frame rotation vector taking q_cur to q_des."""
    qe = quat_mul(q_des, quat_conj(q_cur))
    if qe[0] < 0:
        qe = -qe
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def site_pose(model, data, site):
    sid = model.site(site).id
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[sid])
    return data.site_xpos[sid].copy(), q


def ik_pose(model, pos_des, quat_des, site="tcp", seeds=None, iters=300,
            tol=1e-6):
    """Damped-least-squares IK for the 6 arm joints. Returns joint angles."""
    data = mujoco.MjData(model)
    jids = [model.joint(j).id for j in ARM_JOINTS]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    dadr = np.array([model.jnt_dofadr[j] for j in jids])
    lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    sid = model.site(site).id
    yaw = math.atan2(pos_des[1], pos_des[0])
    if seeds is None:
        # Elbow-up seeds. Tool pitch is q2 - q3 - q4, so q4 ~= q2 - q3 for a
        # horizontal tool and q2 - q3 - pi/2 for a downward one.
        seeds = [np.array([yaw, a, b, a - b - off, 0.0, 0.0])
                 for off in (0.0, math.pi / 2)
                 for a in (0.6, 1.0, 1.4, 1.8, 2.2)
                 for b in (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)]
        rng = np.random.default_rng(0)
        seeds += [rng.uniform(lo, hi) for _ in range(20)]
    best = None
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for seed in seeds:
        data.qpos[:] = model.qpos0
        data.qpos[qadr] = np.clip(seed, lo, hi)
        for _ in range(iters):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            p, q = site_pose(model, data, site)
            err = np.r_[pos_des - p, quat_err(quat_des, q)]
            if err @ err < tol ** 2:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J = np.vstack([jacp, jacr])[:, dadr]
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            data.qpos[qadr] = np.clip(data.qpos[qadr] + dq, lo, hi)
        e = float(np.linalg.norm(err))
        cand = data.qpos[qadr].copy()
        score = e + 1e-3 * np.linalg.norm(cand[3:])
        if best is None or score < best[0]:
            best = (score, e, cand)
    if best[1] > 1e-4:
        raise RuntimeError(f"IK did not converge (err={best[1]:.2e})")
    return best[2]
