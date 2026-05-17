# 预测结果文件说明

由于 OCT/OCTA 三维预测结果文件体积较大，本仓库不直接上传 `.nii.gz` 文件。完整预测结果文件已上传至网盘，供 Demo 查看和 3D Slicer 可视化使用。

## 1. 网盘链接

网盘链接：https://pan.quark.cn/s/c6a0d2e1461d

## 2. 文件列表

### `pred_185_probability_map.nii.gz`

该文件为模型推理后输出的血管概率图，用于表示每个体素属于血管区域的可能性。该结果可用于后续阈值化处理、分割结果分析和三维可视化。

### `pred_185_binary_mask_thr0.3.nii.gz`

该文件为对 probability map 采用 0.3 阈值处理后得到的二值血管掩模。其中血管区域被标记为前景，非血管区域被标记为背景，可直接导入 3D Slicer 进行三维重构展示。

## 3. 使用方式

1. 从网盘下载上述 `.nii.gz` 文件；
2. 使用 3D Slicer 打开 `pred_185_binary_mask_thr0.3.nii.gz`；
3. 通过 Volume Rendering 或 Segment Editor 等功能进行三维重构展示；

## 4. 当前说明

当前结果主要展示模型在仿体 OCT/OCTA 数据上的三维血管分割与重构效果。鼠背皮窗真实数据由于与仿体数据存在分布差异，仍在进一步优化模型泛化能力和分割稳定性。
