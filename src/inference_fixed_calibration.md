# 固定标定相机实验

本实验通过 `--calibration` 开启。不提供时，inference_v2 继续使用原来的预测相机模式。
这里的“固定”是跳过 camera_head，直接采用标定 R、T 和换算后的 K；不涉及训练或冻结网络权重。
同一相机所有输入时间帧共用相同矩阵。深度头、特征提取、运动头、Gaussian 头、输出项目和时间采样逻辑保持原有实现。
运动与 Gaussian 的结果会随相机参数改变，这正是本实验希望观察的影响。

## 服务器运行

需要同步这些文件到服务器项目的相同相对路径：

- `src/model/encoder/encoder.py`
- `src/inference_v2.py`
- `src/inference_utils/calibration.py`
- `config/inference/crops_20261021_172146.json`
- `scripts/run_inference_v2_calibrated.sh`

默认使用最近一次的 6 个相机、crop_short448 和路径顺序 5 0 1 2 3 4：

```bash
bash scripts/run_inference_v2_calibrated.sh
```

标定默认路径为 `/home/yang/lifenglin/NoPo4D/assets/20261021_172146/extr.json`。
默认输出 `output/20261021_172146/crop_short448_view6_fixed_calibration`，必须为空或不存在。
脚本最后转发额外 CLI 参数，可覆盖默认设置。更换路径和四相机示例：

```bash
IMAGE_DIR=assets/20261021_172146/crop_short448_view4 \
OUTPUT_DIR=output/20261021_172146/crop_short448_view4_fixed_calibration \
bash scripts/run_inference_v2_calibrated.sh \
  --num_cameras 4 --camera_order 0 1 2 3
```

四相机示例中的路径顺序仅为输入顺序，需按实际选用相机调整。相机 ID 从文件名
`CAMERAID_FRAME.png` 的首段读取，每个连续时间块必须属于同一个相机；不会用 camera_order 来匹配标定。
默认六相机脚本适用于输入顺序
22010714、22010716、22053917、22070935、22139907、22139911。

## 内外参约定

按 OpenCV 世界到相机约定解释 extr.json：`X_camera = R X_world + T`。
保留标定世界原点与方向，默认不缩放 T，也不做第一相机归一化。
输出 `cameras.npz` 同时包含 w2c、c2w、像素 K，且这些就是实际参与重建的矩阵。
`calibration.json` 和 metadata 中保存每个相机的 ID、裁剪框、像素变换、最终矩阵和畸变参数，便于复核。

预处理配置逐项录入了用户提供的 172146 场景裁剪记录，包括两种裁剪方式的 8 个相机框。
它只适用于这些实际裁剪操作；换场景或重做裁剪后必须更新配置。

从标定尺寸 3000×4096 到源帧 1500×2048，再裁剪、缩放到实际网络输入尺寸：

```text
K_final = S_output × C_crop × S_half × K_calibration
S_half = diag(0.5, 0.5, 1)
C_crop = [[1,0,-x0], [0,1,-y0], [0,0,1]]
S_output = diag(W_final/(x1-x0), H_final/(y1-y0), 1)
```

使用常见像素坐标缩放约定；裁剪框按右下开区间处理。
crop_short448 的横向/纵向比例分别计算，不能把 448×602 当成严格等比例缩放。
提供 `--resize` 时采用最终输入宽高换算 K。旋转、平移不会因裁剪缩放而改变。
原始文件尺寸必须与配置中该 profile 的输出尺寸吻合，防止误用配置。
正方形输入需要 `--calibration_profile crop_448x448`。
未裁剪原图可选 `full`，并使用 `--resize HEIGHT WIDTH` 使网络输入宽高为 14 的倍数。

## 必须保留的实验解释边界

1. **标定平移尺度与预测深度尚未对齐。** 模型原来的深度、位姿是共同预测的；
   换为真实 R/T 后，不应假定模型深度与标定 T 具有相同长度单位。
   默认 `--calibration_translation_scale 1.0` 是原值替换实验，不能据此声称获得了真实尺度重建。
   该选项仅将 T 乘一个正数，不改变 R、K 或深度。若已知标定单位到模型单位的换算，
   可明确设置；不得凭一次结果把某个倍率当成真实标定修正。
   若需要严格的几何一致性实验，应进一步通过对应点三角化等方法估计深度尺度，并验证跨视角一致性。
2. **当前图片没有去畸变。** 此实验保持输入不变，distCoeff 会记录，但不用于图像重采样；
   反投影和渲染采用针孔模型。裁剪后的 K 计算正确不等于消除了畸变误差。
   后续去畸变实验应使用去畸变输出对应的新 K，并重新明确裁剪操作，不能直接复用当前配置。
3. **不额外给 backbone 输入相机 token。** 本实验只替换原先 camera_head 的输出，
   避免同时改变特征提取条件。原有 input_extrinsics/input_intrinsics 参数仅用于条件输入，
   不等价于固定相机；本实验新增的是 fixed_extrinsics/fixed_intrinsics。
4. **质量指标是输入视角的重建指标。** 重点对比各相机的 quality 报告、原视角回放和视频，
   保持相同输入、时间与路径设置。结果变差可能来自深度尺度或畸变模型不匹配，不能单独证明标定错误。

240 帧、30 FPS 的输出依然为 8 秒。4 个输入时间戳仍归一化到 0–1；
0411–0414 在 30 FPS 源视频中的首末时间跨度为 0.1 秒，不会因相机固定而改变。

本地仅做 Python 语法、配置及 shell 静态检查；CUDA 推理与渲染效果需要服务器验证。
