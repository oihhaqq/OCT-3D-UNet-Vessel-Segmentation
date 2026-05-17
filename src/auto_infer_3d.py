# auto_infer_3d.py
import os
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensityRange
import argparse


def estimate_max_patch_size(img_shape, model, device, safety_factor=0.6):
    """
    根据输入图像大小和显存，估算最大可行 roi_size。
    safety_factor: 控制 patch 大小占显存比例，防止 OOM
    """
    total_mem = torch.cuda.get_device_properties(device).total_memory
    reserved_mem = torch.cuda.memory_reserved(device)
    allocated_mem = torch.cuda.memory_allocated(device)
    free_mem = total_mem - reserved_mem - allocated_mem
    free_mem *= safety_factor  # 留一点安全空间

    # 粗略估算 patch 体积 (float32, batch=1, channels=1)
    bytes_per_voxel = 4 * 2  # 输入+输出 float32
    V = free_mem / bytes_per_voxel
    # 假设 patch 是立方体
    patch_len = int(V ** (1/3))
    # 限制不要超过原图尺寸
    patch_len = min(patch_len, min(img_shape))
    # 至少 64
    min_patch_size = 64
    factor = 8
    patch_len = (patch_len // factor) * factor
    patch_len = max(min_patch_size, patch_len)
    return (patch_len, patch_len, patch_len)


def main():
    parser = argparse.ArgumentParser(description="Auto ROI 3D U-Net inference")
    parser.add_argument('--image', type=str, required=True, help='Input NIfTI image')
    parser.add_argument('--output', type=str, required=True, help='Output NIfTI prediction')
    parser.add_argument('--model', type=str, default='best_model_final.pth', help='Trained model path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--sw_batch_size', type=int, default=1)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # 设备
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取图像
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ])
    img = transforms(args.image)
    print(f"Original img.shape: {img.shape}")
    if img.ndim == 3:
        if img.shape[0] > 10:
            img = np.transpose(img, (1, 2, 0))  # (H, W, D)
        img = np.expand_dims(img, axis=0)  # (1, H, W, D)
    elif img.ndim == 4:
        if img.shape[1] == 1:
            img = np.squeeze(img, axis=1)  # (D, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, D)
            img = np.expand_dims(img, axis=0)  # (1, H, W, D)
        elif img.shape[0] == 1:
            pass  # (C, H, W, D)
        else:
            raise ValueError(f"Cannot infer channel dimension from shape {tuple(img.shape)}")
    if img.ndim != 4 or img.shape[0] != 1:
        raise ValueError(f"Expected channel dimension to be 1, got shape {tuple(img.shape)}")
    img = torch.as_tensor(img).unsqueeze(0).to(device)  # (1, C, H, W, D)

    # 构建模型
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)

    # 加载权重
    checkpoint = torch.load(args.model, map_location=device)
    if 'module.' in next(iter(checkpoint.keys())):
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in checkpoint.items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {args.model}")

    # 自动估算 roi_size
    C, H, W, D = img.shape[1:]
    roi_size = estimate_max_patch_size((H, W, D), model, device)
    print(f"Auto-selected roi_size: {roi_size}")
    print(f"Using roi_size: {roi_size}, input shape: {img.shape}")

    # 滑动窗口推理
    with torch.no_grad():
        pred = sliding_window_inference(
            img,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
            mode="gaussian"
        )
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # 二值化
    if args.threshold > 0:
        mask = (pred >= args.threshold).astype(np.uint8)
    else:
        mask = pred.astype(np.float32)

    # 保存 NIfTI
    orig_img = nib.load(args.image)
    out_img = nib.Nifti1Image(mask, affine=orig_img.affine, header=orig_img.header)
    nib.save(out_img, args.output)
    print(f"Prediction saved to {args.output}")


if __name__ == "__main__":
    main()
