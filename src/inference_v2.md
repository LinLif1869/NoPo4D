# inference_v2：输入、质量诊断与视频输出

保留原 inference.py，不依赖 mutiwin 中缺失的扩展模型接口。
需要原 CUDA/gsplat 推理环境、SciPy，以及支持 libx264 的 ffmpeg。
本地仅完成静态检查，模型推理和 GPU 渲染需在服务器验证。

## 示例

```bash
python src/inference_v2.py \
  --image_dir images --num_cameras 4 --output_dir output_v2 \
  --video_frames 240 --fps 30 \
  --trajectory camera_path --path_geometry arc --path_cycle pingpong \
  --intrinsics_mode fixed --reference_camera 0 --time_mode dynamic
```

输出目录需为空。已删除 PLY 快照功能及 --ply_times/--ply_min_opacity 参数。
保留 gaussians.pt 完整动态状态，供重新渲染。

## 什么输入生成 240 帧？

例如输入 4 台同步、固定机位相机，每台 4 个时刻，共 16 张 RGB：

```text
cam0_t0.png cam0_t1.png cam0_t2.png cam0_t3.png
cam1_t0.png cam1_t1.png cam1_t2.png cam1_t3.png
cam2_t0.png cam2_t1.png cam2_t2.png cam2_t3.png
cam3_t0.png cam3_t1.png cam3_t2.png cam3_t3.png
```

自然数字排序后按 camera-major 分组，得到 [1,16,3,H,W]。
每台相机默认时间戳为 [0,1/3,2/3,1]。模型一次前馈生成动态高斯，
然后为输出帧 j=0..239 构造目标相机、像素内参和场景时间，再逐帧渲染。
默认 dynamic 的场景时间为 t_min+(t_max-t_min)*j/239；
相机路径独立地按 pingpong 前进再返回。ffmpeg 以 30 FPS 编码 240 张图，
文件播放时长为 240/30=8 秒。这不是输入必须有 240 帧，也不表示推理速度为 30 FPS。

原采样时间跨度不会从图片自动恢复。例如采集 30 FPS 视频的第 0/1/2/3 帧，
实际仅覆盖 0.1 秒动作；仍然输出 8 秒时，就是把这个短动作窗口延长展示。
更多输出帧不会增加真实观测信息，也不保证快速动作的插值准确。
单时刻输入不会获得真实动态运动；仍可输出相机移动的静态场景视频。

图片须同尺寸且 H/W 均为 14 的倍数。--resize HEIGHT WIDTH 可显式缩放，
但改变宽高比会拉伸图像。程序不直接读取 MP4，需先提取同步 RGB 帧。

非均匀采样用 --manifest inputs.json 指定文件顺序和归一化时间：

```json
[
 {"file":"cam0_t0.png","timestamp":0.0},
 {"file":"cam0_t1.png","timestamp":1.0},
 {"file":"cam1_t0.png","timestamp":0.0},
 {"file":"cam1_t1.png","timestamp":1.0}
]
```

所有相机须具有相同数量、相同时间戳的同步帧；各相机内部时间严格递增。
metadata 的 source_duration_seconds 为 null，因为归一化时间不能恢复秒数。

## 轨迹控制

- --camera_order 0 2 3 1：空间遍历顺序，索引指输入分组，不是文件中的物理 ID。
- --path_geometry arc：估计光轴交汇中心，沿中心方向球面插值、半径线性插值。
  等半径机位保持半径，避免弦段向主体靠近。旋转使用最短弧 SLERP。
- --path_geometry linear：恢复直线位置插值。
- --path_cycle pingpong：默认，首相机到末相机再返回；不跨越未采集的首尾缺口。
- --path_cycle open：首相机到末相机，仅单程。
- --path_cycle closed：闭环遍历，省略重复终点；适合确实环绕布置的机位。
- --intrinsics_mode fixed：默认，固定参考相机的完整 K，避免内参变化带来缩放。
- --reference_camera 0：固定 K 的输入相机索引，不受 camera_order 重排影响。
- --intrinsics_mode interpolate：相邻机位 K 线性插值。
- --orbit_center X Y Z：也可给 arc 指定中心。自动中心不可靠时明确报错；
  平行相机或异常标定可选择 linear，不会静默生成不可信圆弧。
- --trajectory orbit：独立生成圆周 look-at 路径；支持 --orbit_up、
  --orbit_radius_scale、--orbit_degrees。此模式使用固定参考 K，不使用 path_cycle。
  一台相机也可使用，但未观察的区域可能有空洞。

全部路径使用模型预测坐标，而非外部标定；未移植多窗口、来源切换或擦除转场。
场景时间独立控制：dynamic 从输入最早到最晚，freeze 固定 --freeze_time，
pingpong 前进再返回。冻结时间仍为归一化值。循环视频的动作连续性取决于时间模式。

## 质量判断和保留结果

- quality/reconstruction.json：在真实输入相机和输入时间上重新渲染，逐视图、
  逐相机及总体报告 MSE/PSNR、alpha<0.5 空洞比例、非有限值检查。
  总体 PSNR 由总体 MSE 转换，并非平均各帧 PSNR。
- quality/camera_*_gt_render_error_alpha.png：每台相机中间输入帧的
  GT、渲染、4 倍绝对误差、alpha 四列对比。
- metadata.json：编码器耗时、编码器峰值 CUDA 已分配显存、轨迹配置、
  文件顺序、视频时长与逐输出帧 alpha 空洞比例；status=complete 表示运行结束。

指标没有设置武断的合格阈值。输入回渲 PSNR 高不保证新视角正确；
alpha 低只表示覆盖不足，不直接等于几何错误。视频新视角没有 GT，因此不报告其 PSNR。
编码器计时使用 CUDA 同步，不包含加载、文件写入、质量检查或视频生成耗时。

其他输出：camera_path.mp4、images/ 原机位回放图、optical_flow/ 光流可视化、
rendered_depth_alpha/ 浮点深度与 alpha、predictions.pt 原始深度/光流/运动概率、
cameras.npz 相机参数、trajectory.npz 实际视频路径/时间、gaussians.pt 完整动态高斯。
--render_timestamps 仅控制原机位 images/ 每台相机输出数量，不控制 MP4 长度。
--save_video_frames / --save_video_depth 可额外保存视频逐帧 RGB 或深度/alpha。

c2w 使用 OpenCV 约定，K 为像素单位，尺度为模型重建尺度；深度沿用 gsplat RGB+D
累积深度语义，并非已除 alpha 的深度。奇数宽高的视频会由 ffmpeg 在右/底部补齐。
--render_batch_size 默认 1，控制渲染显存，不降低编码所有输入所需的显存。
