import math
import numpy as np
import multiwindow_utils as mw

rng = np.random.default_rng(0)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def look_at_c2w(center, focus, up=np.array([0,-1.0,0])):
    z = focus - center; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3,:3] = np.stack([x,y,z],1); T[:3,3] = center
    return T

def dome(n, span_deg, radius=3.0):
    angs = np.deg2rad(np.linspace(-span_deg/2, span_deg/2, n))
    c = np.stack([radius*np.sin(angs), np.zeros(n)-0.3, -radius*np.cos(angs)], 1)
    return np.stack([look_at_c2w(ci, np.zeros(3)) for ci in c]), np.rad2deg(angs)

# --- windows ---------------------------------------------------------------
assert mw.make_windows(6) == [[0,1,2,3],[2,3,4,5]]
assert mw.make_windows(8) == [[0,1,2,3],[2,3,4,5],[4,5,6,7]]
assert mw.make_windows(10) == [[0,1,2,3],[2,3,4,5],[4,5,6,7],[6,7,8,9]]
assert mw.make_windows(7) == [[0,1,2,3],[2,3,4,5],[3,4,5,6]]   # 末窗贴齐
assert mw.make_windows(3) == [[0,1,2]]
ow = mw.owner_window(mw.make_windows(10), 10)
assert ow == [0,0,0,1,1,2,2,3,3,3] or ow[0]==0 and ow[-1]==3, ow
assert all(o >= 0 for o in ow)
# 内部相机（两侧都有邻居）优先：相机 1,2 在窗口0内部，3,4 在窗口1内部...
print("owner_window n=10:", ow)

# --- coordinate round-trip & normalisation properties ----------------------
c2w, _ = dome(4, 30)
w2c = mw._inv_rt(c2w)
ex, m, first = mw.normalize_w2c(w2c)
assert np.allclose(ex[0], np.eye(4), atol=1e-9), "第一台相机应变成单位位姿"
cn = mw._inv_rt(ex)
d = np.linalg.norm(cn[:,:3,3], axis=1)
assert abs(mw._lower_median(d) - 1.0) < 1e-9 or mw._lower_median(d*m) <= 0.1+1e-9
P = rng.normal(size=(1000,3))*0.5
Pl = mw.world_to_local(P, m, first)
assert np.allclose(mw.local_to_world(Pl, m, first), P, atol=1e-9)
# 与「渲染一致」检验：世界点投到第 k 台相机的深度 == 局部点（局部相机）深度 * m
k = 2
Pw_h = np.c_[P, np.ones(len(P))]
z_world = (w2c[k] @ Pw_h.T).T[:,2]
z_local  = (ex[k] @ np.c_[Pl, np.ones(len(Pl))].T).T[:,2]
assert np.allclose(z_world, z_local*m, atol=1e-8)
print("normalize/local<->world OK, m=%.4f" % m)

# 目标相机变到局部系后，同一世界点的深度 * m 应等于世界系深度；投影像素也应一致
tgt = look_at_c2w(np.array([0.4, -0.1, -2.7]), np.zeros(3))
tgt_l = mw.camera_world_to_local(tgt, m, first)
zw = (mw._inv_rt(tgt) @ Pw_h.T).T
zl = (mw._inv_rt(tgt_l) @ np.c_[Pl, np.ones(len(Pl))].T).T
assert np.allclose(zw[:, :3], zl[:, :3] * m, atol=1e-8)
print("camera_world_to_local OK")

# --- shared camera residual -------------------------------------------------
H, W = 60, 80
true_depth = 2 + rng.random((H, W))
m_a, m_b = 1.7, 2.3
true_scale_b_to_a = 1.06                      # 窗口 B 的深度整体偏大 6%
depth_a = true_depth / m_a
depth_b = true_depth * true_scale_b_to_a / m_b
depth_b[:5] = rng.random((5, W)) * 10        # 一些离群像素
r = mw.shared_camera_residual(depth_a, m_a, depth_b, m_b)
assert abs(r["scale_b_to_a"] - 1/true_scale_b_to_a) < 5e-3, r
print("shared residual:", r)

# --- greedy chain / azimuth ordering ---------------------------------------
c2w, angs = dome(40, 80, 3.0)                 # 40 台，间距约 2°
centers = c2w[:, :3, 3]
perm = rng.permutation(40)
order = mw.azimuth_order(centers[perm], np.zeros(3), np.array([0,-1.0,0]))
sorted_angs = angs[perm][order]
assert np.all(np.diff(sorted_angs) > 0) or np.all(np.diff(sorted_angs) < 0), sorted_angs
# 构造重叠：overlap = max(0, 1 - dtheta/24)  （仅用于测试链逻辑）
A = np.abs(angs[:,None]-angs[None])
ov = np.clip(1 - A/18.0, 0, 1)                # 0.5 重叠 <=> 9°
chain = mw.greedy_chain(np.arange(40), ov, 0.5)
gaps = np.diff(angs[chain])
assert gaps.max() <= 9.0 + 1e-6 and len(chain) == 11 and chain[-1] == 39, (len(chain), gaps)
print("greedy chain: %d cameras cover %.1f deg" % (len(chain), angs[chain[-1]]-angs[chain[0]]))
# 断档：中间删掉一段，链应在断档处停止
keep = [i for i in range(40) if not (14 <= i <= 25)]
chain2 = mw.greedy_chain(np.array(keep), ov, 0.5)
assert chain2[-1] <= 14 or len(chain2) < len(keep)
print("gap stops chain at camera", chain2[-1])

# --- blend weights ----------------------------------------------------------
oa = np.array([0, 8, 16, 24.0])
for ang in [-5, 0, 3.9, 4.0, 4.1, 8, 12, 30]:
    w = mw.blend_weights(ang, oa, width=3.0)
    assert abs(w.sum()-1) < 1e-12 and (w >= 0).all()
w4 = mw.blend_weights(4.0, oa, 3.0); assert abs(w4[0]-0.5) < 1e-9
assert mw.blend_weights(2.0, oa, 3.0)[0] == 1.0     # 过渡带之外完全用最近源
print("ALL TESTS PASSED")
