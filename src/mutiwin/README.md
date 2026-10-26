# NoPo4D 多窗口实时 4D

最终演示命令：

```bash
cd /home/yang/lk/NoPo4D
PYTHONUTF8=1 /home/yang/miniconda3/envs/nopo4d/bin/python \
  src/mutiwin/realtime_4d_demo.py --frames 240 --fps 60
```

默认使用两个独立的 4 相机窗口和八个物理锚点，沿
`40 → 64 → 65 → 41 → 42 → 66 → 67 → 43` 构成真正的滑动链。
每个锚点仅在线渲染中心 4° 可靠扇区。扇区间使用方向性羽化擦除：
始终由有效锚点覆盖画面，不再闪黑；过渡约持续 5 帧，只在一条约 4%
画面宽度的移动接缝中局部混合，避免全屏交叉淡化产生双人重影。
相比旧版只保留 cam40/41/42/43，该路径不会丢弃窗口内部的四台辅助相机，
因此除 cam41→cam42 的真实低重叠缺口外，其余周期性大跨度跳变被消除。

输出为 `output_nas_realtime_multiwindow_4d/realtime_4d.mp4`。加 `--interactive` 可同时在本机 OpenCV 窗口预览，按 `Q` 或 `Esc` 退出；已有的鼠标拖动、时间滑条交互页面仍位于 `output_nas_wide_4d/index.html`。

`--path_mode calibrated` 会沿 cam40→cam43 的物理路径强行连续插值，仅用于观察低重叠区为什么产生空洞与重影，不建议用于最终展示。

## 真正鼠标操作的 4D Gaussian viewer

下面的程序不是 WebP 网格，也不是轮播图片。它把两套 4D Gaussian 常驻 GPU，鼠标每次移动都修改目标相机 c2w，并立即调用 gsplat：

```bash
cd /home/yang/lk/NoPo4D
PYTHONUTF8=1 MPLCONFIGDIR=/tmp/matplotlib_nopo4d \
  /home/yang/miniconda3/envs/nopo4d/bin/python \
  src/mutiwin/interactive_gaussian_viewer.py
```

如果服务器没有 `DISPLAY`/X11，程序会自动启动真正的远程实时渲染服务，而不会调用 OpenCV/Qt 窗口。默认监听 `127.0.0.1:8001`。在本地电脑另开终端建立安全隧道：

```bash
ssh -N -L 8001:127.0.0.1:8001 yang@lyg0168
```

然后在本地浏览器打开 `http://127.0.0.1:8001`。网页只发送鼠标控制参数并显示服务器即时编码的帧，每次请求仍在服务器调用 4D Gaussian + gsplat，不是预渲染图片网格。

也可以显式启动：

```bash
PYTHONUTF8=1 MPLCONFIGDIR=/tmp/matplotlib_nopo4d \
  /home/yang/miniconda3/envs/nopo4d/bin/python \
  src/mutiwin/interactive_gaussian_viewer.py --server --port 8001
```

控制方式：

- 鼠标左键拖动：连续改变 yaw/pitch；
- 鼠标滚轮：改变观察距离；
- 鼠标右键横向拖动或 `4D time` 滑条：改变 4D 时间；
- 空格：播放/暂停时间；
- 数字 `1`–`8`：显式切换八个物理源相机；默认 `1` 是效果最稳定的 cam64；
- `R`：复位；`S`：截图；`Q`/`Esc`：退出。

为了避免伪造连续性，鼠标拖动不会自动跨窗口切源。当前 8 台相机在 cam41–cam42 之间只有约 0.24 重叠，数字键跨区切换是显式的。要让鼠标无缝覆盖完整 52° 前桌面范围，拍摄侧必须补足这段重叠或使用更宽镜头。

无图形界面机器可以使用同一渲染路径录制验证：

```bash
PYTHONUTF8=1 MPLCONFIGDIR=/tmp/matplotlib_nopo4d \
  /home/yang/miniconda3/envs/nopo4d/bin/python \
  src/mutiwin/interactive_gaussian_viewer.py \
  --record output_nas_true_interactive_4d/mouse_orbit_demo.mp4 \
  --frames 180 --fps 60
```

## 六台同焦段稀疏相机：Batch 隔离版

`sparse_calibrated_stream.py` 面向 `cam38,40,42,44,46,48`：每台相机放在独立
Batch，避免 DA3 在零重叠视角之间做 global attention；六套深度通过标定焦点 Z
统一尺度，并在流式更新时锁定首帧尺度。实时显示只 rasterize 相邻两套 Gaussian。

```bash
PYTHONUTF8=1 MPLCONFIGDIR=/tmp/matplotlib_nopo4d CUDA_VISIBLE_DEVICES=0 \
  /home/yang/miniconda3/envs/nopo4d/bin/python \
  src/mutiwin/sparse_calibrated_stream.py \
  --image_dir assets/custom_input1013 \
  --calibration_json src/calibration_results-19-11-12_ByteDT.json \
  --camera_ids 38,40,42,44,46,48 \
  --input_frames 1 --max_long_edge 392 \
  --server --host 127.0.0.1 --port 8002 \
  --output_dir output_nas_sparse6_live
```

本机实测输入回渲 28.44 dB，392×280 稳态前馈 115.27 ms（8.68 FPS）。
这能让六个物理锚点分别高清、快速更新，但不能把锚点回渲指标解释为无缝120°自由
视点。会聚相机的重叠不能简单由光轴差/FOV判断，修正分析和接缝优化见报告第17节。

## 新80cam五机约180°输入

已下载新标定并通过HTTP范围读取提取同步源帧118、121：
`assets/nas80_20261015_cam4_50_52_54_56`。
物理顺序为`56,54,52,50,4`，实测方位跨度182.02°，水平FOV约64.6°–64.8°。

本轮桌面结果为`output_nas80_5cam_180_desk_final/sparse_calibrated_two_pass.mp4`，
完整参数、准确指标口径和残余接缝说明见优化报告第18节。该结果仍有局部亮色轮廓与
几何错位，尚未达到完全无缝验收。视频8s只对应0.1s的两帧动作窗口。
