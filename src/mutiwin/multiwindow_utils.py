"""多窗口 NoPo4D 的纯 numpy 工具（不依赖 torch，可单独测试）。

约定
----
- c2w / w2c 均为 (N,4,4) 的 numpy 数组，OpenCV 相机系（x 右、y 下、z 前）。
- 与你的 encoder.normalize_extrinsics 一致：输入 w2c，
  先乘第一台相机的 c2w（以第一台相机为世界系），
  再把平移除以「所有相机到原点距离的中位数」（torch.median，偶数个取较小的那个）。

本文件包含
----------
1. azimuth_order / greedy_chain：沿方位角排序并按最小重叠贪心选链
2. make_windows / owner_window：滑动窗口划分与「谁负责哪台相机」
3. normalize_w2c / local_to_world / world_to_local / camera_world_to_local：窗口局部系与标定世界系互转
4. shared_camera_residual：共享相机在两个窗口里的深度比值 -> 尺度残差
5. blend_weights：按视角方位角在相邻负责相机之间做平滑权重
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 1. 选相机
# --------------------------------------------------------------------------- #
def azimuth_order(centers: np.ndarray, focus: np.ndarray, up: np.ndarray) -> np.ndarray:
    """按绕 up 轴的方位角给相机排序。
    参考方向取所有相机水平方向的平均，这样 atan2 的跨越点落在相机弧的背面，
    弧内排序不会被 ±pi 折返打乱。centers:(N,3)。"""
    up = up / np.linalg.norm(up)
    v = centers - focus[None]
    v = v - (v @ up)[:, None] * up[None]          # 投影到水平面
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    ref = v.mean(axis=0)
    if np.linalg.norm(ref) < 1e-6:                # 相机绕满一圈，退化情形
        ref = v[0]
    ref /= np.linalg.norm(ref)
    side = np.cross(up, ref)
    ang = np.arctan2(v @ side, v @ ref)
    return np.argsort(ang)


def greedy_chain(order: np.ndarray, overlap: np.ndarray, min_overlap: float = 0.5) -> list[int]:
    """沿 order 贪心选链：每一步选满足 overlap>=min_overlap 的最远一台，
    使每台相机贡献的覆盖角度尽量大。overlap:(N,N)，下标对应相机编号。"""
    order = list(map(int, order))
    chain = [order[0]]
    pos = 0
    while pos < len(order) - 1:
        last = chain[-1]
        nxt = None
        for k in range(pos + 1, len(order)):
            if overlap[last, order[k]] >= min_overlap:
                nxt = k          # 一直往后找，保留最远的合格者
            else:
                break            # 按方位角排序时，重叠随距离单调下降，可提前停
        if nxt is None:
            break                # 出现断档：需要补相机，链到此为止
        chain.append(order[nxt])
        pos = nxt
    return chain


# --------------------------------------------------------------------------- #
# 2. 窗口划分
# --------------------------------------------------------------------------- #
def make_windows(n: int, window: int = 4, stride: int = 2) -> list[list[int]]:
    """把链上 n 台相机（编号 0..n-1）切成滑动窗口，最后一个窗口贴齐末尾。"""
    if n <= window:
        return [list(range(n))]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return [list(range(s, s + window)) for s in starts]


def owner_window(windows: list[list[int]], n: int) -> list[int]:
    """每台相机由哪个窗口负责渲染（返回窗口下标）。
    规则：优先选相机处于窗口内部（两侧都有邻居）的窗口；
    若有多个，取靠前的；链两端的相机只能由端窗口负责。"""
    owner = [-1] * n
    for cam in range(n):
        best, best_score = -1, -1
        for wi, w in enumerate(windows):
            if cam not in w:
                continue
            p = w.index(cam)
            score = 2 if 0 < p < len(w) - 1 else 1
            if score > best_score:
                best, best_score = wi, score
        owner[cam] = best
    return owner


# --------------------------------------------------------------------------- #
# 3. 局部系 <-> 标定世界系
# --------------------------------------------------------------------------- #
def _inv_rt(T: np.ndarray) -> np.ndarray:
    R, t = T[..., :3, :3], T[..., :3, 3:]
    out = np.zeros_like(T)
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3:] = -np.swapaxes(R, -1, -2) @ t
    out[..., 3, 3] = 1.0
    return out


def _lower_median(x: np.ndarray) -> float:
    s = np.sort(x.ravel())
    return float(s[(len(s) - 1) // 2])      # 与 torch.median 一致：偶数取较小者


def normalize_w2c(w2c: np.ndarray, min_dist: float = 1e-1):
    """复现 normalize_extrinsics：返回 (归一化后的 w2c, 中位距离 m, 第一台 c2w)。"""
    first_c2w = _inv_rt(w2c[:1])[0]
    ex = w2c @ _inv_rt(w2c[:1])                    # = w2c @ first_c2w
    c2w_norm = _inv_rt(ex)
    m = max(_lower_median(np.linalg.norm(c2w_norm[:, :3, 3], axis=-1)), min_dist)
    ex = ex.copy()
    ex[:, :3, 3] = ex[:, :3, 3] / m
    return ex, m, first_c2w


def local_to_world(points_local: np.ndarray, m: float, first_c2w: np.ndarray) -> np.ndarray:
    """窗口局部系下的点 -> 标定世界系： P_world = R_first (m * P_local) + t_first。"""
    R, t = first_c2w[:3, :3], first_c2w[:3, 3]
    return (m * points_local) @ R.T + t


def world_to_local(points_world: np.ndarray, m: float, first_c2w: np.ndarray) -> np.ndarray:
    R, t = first_c2w[:3, :3], first_c2w[:3, 3]
    return ((points_world - t) @ R) / m


def camera_world_to_local(c2w_world: np.ndarray, m: float, first_c2w: np.ndarray) -> np.ndarray:
    """把标定世界系下的目标相机 c2w 变到某个窗口的局部（归一化）系，
    用于「每个窗口在自己的局部系里渲染」：
        R_l = R_f^T R_w,   t_l = R_f^T (t_w - t_f) / m
    """
    Rf, tf = first_c2w[:3, :3], first_c2w[:3, 3]
    out = np.eye(4)
    out[:3, :3] = Rf.T @ c2w_world[:3, :3]
    out[:3, 3] = Rf.T @ (c2w_world[:3, 3] - tf) / m
    return out


# --------------------------------------------------------------------------- #
# 4. 共享相机的尺度残差
# --------------------------------------------------------------------------- #
def shared_camera_residual(
    depth_a: np.ndarray, m_a: float,
    depth_b: np.ndarray, m_b: float,
    valid: np.ndarray | None = None,
    trim: float = 0.1,
) -> dict:
    """同一台相机在窗口 A、B 里各有一张深度图（各自局部尺度）。
    先乘各自的 m 换到标定世界系的单位，再取比值：
        ratio = (D_a * m_a) / (D_b * m_b)
    返回稳健中位数、log 比值的 MAD（接缝不一致度）以及有效像素占比。"""
    da, db = depth_a * m_a, depth_b * m_b
    ok = np.isfinite(da) & np.isfinite(db) & (da > 1e-6) & (db > 1e-6)
    if valid is not None:
        ok &= valid
    if ok.sum() < 100:
        return {"scale_b_to_a": 1.0, "log_mad": float("nan"), "valid_frac": float(ok.mean())}
    lr = np.log(da[ok] / db[ok])
    lo, hi = np.quantile(lr, [trim, 1 - trim])
    core = lr[(lr >= lo) & (lr <= hi)]
    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med)))
    return {"scale_b_to_a": float(np.exp(med)), "log_mad": mad, "valid_frac": float(ok.mean())}


# --------------------------------------------------------------------------- #
# 5. 相邻负责相机之间的平滑融合权重
# --------------------------------------------------------------------------- #
def blend_weights(view_angle: float, owner_angles: np.ndarray, width: float) -> np.ndarray:
    """view_angle：目标视角的方位角；owner_angles：各负责相机的方位角（升序）；
    width：从 100% 渐变到 0% 的角度宽度。只在相邻两台之间过渡，返回和为 1 的权重。"""
    a = np.asarray(owner_angles, dtype=np.float64)
    w = np.zeros(len(a))
    if view_angle <= a[0]:
        w[0] = 1.0
        return w
    if view_angle >= a[-1]:
        w[-1] = 1.0
        return w
    i = int(np.searchsorted(a, view_angle) - 1)
    mid = 0.5 * (a[i] + a[i + 1])
    x = np.clip((view_angle - (mid - width / 2)) / max(width, 1e-9), 0.0, 1.0)
    s = x * x * (3 - 2 * x)                      # smoothstep
    w[i], w[i + 1] = 1.0 - s, s
    return w
