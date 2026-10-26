# 稀疏图片的匹配覆盖率评估

入口 `src/evaluate_camera_overlap.py`。仅需 numpy 与提供 SIFT 的 OpenCV；
服务器无桌面环境可使用 opencv-python-headless（不要同时安装多个 OpenCV 发行包）。
不加载 NoPo4D、不需要 CUDA，不筛选或移动相机图片。

## 输入与执行

每个相机选同一真实时刻的一张图片，放入独立目录：

```bash
python src/evaluate_camera_overlap.py \
  --image_dir assets/overlap_same_time \
  --output_dir output/overlap_same_time
```

或显式指定图片，避免把多时刻目录的所有帧混在一起：

```bash
python src/evaluate_camera_overlap.py \
  --images assets/cam0_t0.png assets/cam1_t0.png assets/cam2_t0.png assets/cam3_t0.png \
  --output_dir output/overlap_t0
```

目录模式会评估其中每一张图片，不根据文件名猜测相机或时间。N 张图评估
N(N-1)/2 对；四相机共六对。目录不递归读取。输出目录必须为空。
若希望评估实际 inference 输入的覆盖，应使用相同裁剪后的图，而不是原始全幅。
支持不同尺寸图片。处理最长边默认最多 1600，可用 --max_edge 修改；只缩小不放大。

## 方法与指标定义

1. 每张图提取 SIFT，默认最多 8000 个特征。
2. 双向 Lowe ratio 筛选并保留互相匹配的点，默认 ratio=.75。
3. 基础矩阵 RANSAC 去除不满足双视图几何的点；默认阈值为处理分辨率下 2 像素。
4. 至少 20 个内点且内点比例至少 .3，才输出面积估计。
5. 分别计算两图中内点凸包面积 / 该图面积，记作 coverage_a 和 coverage_b。
   coverage_mean=(coverage_a+coverage_b)/2；coverage_min 是两者较小值。
6. 同时报告 12×12 网格中包含内点的格子比例，辅助检查匹配的空间分布。

这些数值在 [0,1]；0.4 表示 40%。不是 IoU，也不是严格的共同可见表面面积。
凸包包含内部无匹配区域，可能跨过遮挡或不连续区域，造成高估；
无纹理、反光、大基线、动态物体或重复纹理可能导致漏匹配或误匹配。
低纹理的真实重叠也可能无法估计。status!=estimated 时面积字段为 null，
CSV 留空，不得当成零重叠。对角线 1 是同图比较的约定，不是匹配测量结果。
不要把单一分数当成模型输入质量的验收标准；必须查看匹配可视化。

默认 --geometry fundamental 用于普通具有视差的三维场景。
仅在近似平面或纯旋转条件下使用 --geometry homography；它不适合一般多深度场景。
两种模式不会自动选择更高分，避免通过模型选择抬高估计。

## 输出

- report.json：完整参数、图片尺寸、关键点数、全部图片对结果与局限说明。
- pairs.csv：每对图的内点数、内点比例、双向面积和网格覆盖率。
- coverage_matrix.csv：对称的 coverage_mean 矩阵，行列标明索引与文件名。
- pairs/000_001.jpg 等：几何内点匹配线和绿色覆盖凸包；最多显示 100 条线，
  覆盖率使用全部内点。失败时不绘制覆盖面积。

## 与主管 select_cameras.py 的区别

主管脚本的 pair_overlap 用相对光轴 yaw/pitch 和 FOV 计算角域矩形交叠，
然后对两个方向求平均；它不读取图像、不使用相机平移或场景深度，
还忽略相对滚转。因此即使标定完全正确，它也不是有限深度场景实际重叠面积。
本脚本提供的是当前实际图片中，经几何验证的匹配支持区域，两者不要混称同一指标。
对主管汇报建议名称为“图像匹配覆盖率（SIFT + RANSAC 凸包估计）”，
同时附内点数、网格覆盖率、可视化以及无法估计的图片对。
若需要精确的几何共同可见率，还需要正确裁剪/去畸变后的内参、外参、
深度或场景表面，并进行遮挡检查。仅靠稀疏 RGB 图片不能保证精确测量。

本地完成静态语法检查，未运行实际图片匹配或验证数据效果。
