# OCT-3D-UNet-Vessel-Segmentation
基于 PyTorch 与 MONAI 的 OCT 血管三维分割与重构项目，包含数据预处理、3D U-Net 模型训练、滑窗推理及 3D Slicer 可视化结果展示
# 基于 3D U-Net 的 OCT 血管三维分割与重构研究

## 1. 项目简介

本项目来源于课题组科研项目，面向 OCT/OCTA 三维血管图像分割任务，基于 PyTorch 与 MONAI 搭建 3D U-Net 医学图像分割模型，实现从三维 OCT 数据读取、数据预处理、模型训练、滑窗推理到血管三维重构可视化的完整流程。

目前项目已完成仿体数据的训练、推理与三维重构展示，能够输出血管概率图和二值分割掩模，并通过 3D Slicer 进行三维可视化展示。鼠背皮窗真实数据的分割效果仍在进一步优化中，当前项目进度约为 70%。

## 2. 项目目标

- 构建适用于 OCT/OCTA 三维数据的血管分割模型；
- 使用 3D U-Net 完成三维体数据的血管结构提取；
- 输出 probability map 和 binary mask 结果；
- 使用 3D Slicer 对预测结果进行三维重构与可视化展示；
- 针对仿体数据与鼠背皮窗真实数据之间的数据分布差异进行优化。

## 3. 我的主要工作

- 使用 nibabel、SimpleITK 等工具读取和处理 NIfTI 格式数据；
- 基于 PyTorch 与 MONAI 搭建 3D U-Net 模型训练流程；
- 完成训练脚本、推理脚本及数据处理流程的调试；
- 使用 DiceCE Loss 等损失函数进行模型训练；
- 采用 3D patch 与 sliding window inference 完成大尺寸三维数据推理；
- 输出 probability map 和 binary mask，并在 3D Slicer 中进行三维重构展示；
- 使用 ChatGPT、Codex 辅助完成代码理解、报错排查、脚本修改和流程梳理。

## 4. 技术栈

- 编程语言：Python
- 深度学习框架：PyTorch、MONAI
- 医学图像处理：nibabel、SimpleITK、NIfTI(.nii/.nii.gz)
- 模型结构：3D U-Net
- 推理方式：3D patch、sliding window inference
- 可视化工具：3D Slicer
- 辅助工具：ChatGPT、Codex
- 开发环境：Linux / Windows、CUDA、VS Code

## 5. 项目流程

整体流程如下：
<img width="1122" height="1402" alt="pineline" src="https://github.com/user-attachments/assets/91d833b7-f2e6-4d22-b9f2-dc1dacefeeaa" />

## 6. 三维重构与分割掩模结果展示

<img width="890" height="615" alt="slicer_result" src="https://github.com/user-attachments/assets/e55fafb0-cbe7-4dae-a13d-75cf1d881f2d" />
<img width="645" height="453" alt="slicer_result_mask" src="https://github.com/user-attachments/assets/cae14214-3ccb-4b11-8763-64935b06c4b9" />

## 7. 完整预测结果文件

由于 `.nii.gz` 三维预测结果文件体积较大，本仓库未直接上传完整结果文件。  
完整 probability map 与 binary mask 文件已上传至网盘，说明见：

[预测结果文件说明](results/result_files.md)
