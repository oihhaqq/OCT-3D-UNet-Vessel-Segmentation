import os
import sys
import json
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import monai
from monai.data import Dataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandFlipd, RandRotated, RandZoomd, RandGaussianNoised,
    RandAdjustContrastd, ToTensord, Lambdad, MapTransform,
    RandCropByPosNegLabeld
)
from monai.config import KeysCollection
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ---------------- 自定义前景中心裁剪变换 ----------------
class ForegroundCenterCropd(MapTransform):
    """
    从标签中随机选取一个前景体素作为中心，裁剪出 roi_size 大小的图像和标签块。
    若标签中无前景，则退化为纯随机中心裁剪（避免中断）。
    此变换可确保每个训练块都包含血管，适用于稀疏标注场景。
    """
    def __init__(self, keys: KeysCollection, roi_size, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.roi_size = np.array(roi_size)

    def __call__(self, data):
        d = dict(data)
        # 获取标签 (C, H, W, D) 或 (H, W, D)
        label = d["label"]
        # 找出所有前景体素 (数值 > 0)
        coords = np.argwhere(label > 0)
        if len(coords) == 0:
            # 若无前景，随机生成一个中心点
            spatial_shape = label.shape[1:] if label.ndim == 4 else label.shape
            center = np.array([np.random.randint(0, s) for s in spatial_shape])
        else:
            # 随机选择一个前景体素作为中心
            idx = np.random.randint(len(coords))
            center_point = coords[idx]
            if label.ndim == 4 and label.shape[0] == 1:
                center = center_point[1:]  # 去除通道索引
            else:
                center = center_point[:3]   # 仅取空间坐标

        # 计算裁剪起始坐标，并防止越界
        start = np.clip(center - self.roi_size // 2, 0, None).astype(int)
        end = start + self.roi_size
        spatial_shape = label.shape[1:] if label.ndim == 4 else label.shape
        for i in range(3):
            if end[i] > spatial_shape[i]:
                start[i] = spatial_shape[i] - self.roi_size[i]
                end[i] = spatial_shape[i]
        start = start.astype(int)
        end = end.astype(int)

        # 执行裁剪
        for key in self.keys:
            img = d[key]
            if img.ndim == 4:  # (C, H, W, D)
                d[key] = img[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            else:              # (H, W, D)
                d[key] = img[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return d

# ---------------- 超参数 ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data-pool/chentianhang/OCT_Project/3D_Unet/3d_processed')
parser.add_argument('--checkpoint_dir', default='/data-pool/chentianhang/OCT_Project/3D_Unet/checkpoints')
parser.add_argument('--log_dir', default='/data-pool/chentianhang/OCT_Project/3D_Unet/logs')
parser.add_argument('--json_file', default='datasets_cv.json')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--patch_size', type=int, nargs=3, default=[192, 192, 64])
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--dropout_rate', type=float, default=0.2)
parser.add_argument('--overfit_one', action='store_true', help='Use the first sample as both train and val for debugging')
parser.add_argument('--sanity_check_only', action='store_true', help='Only print image/label sanity information and exit')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# ---------------- 准备数据集划分 JSON ----------------
json_path = os.path.join(args.data_dir, args.json_file)
all_train_val_samples = [
    "diameter_074um_flow_1mmps",
    "diameter_074um_flow_3mmps",
    "diameter_128um_flow_1mmps",
    "diameter_235um_flow_1mmps",
    "diameter_235um_flow_3mmps",
    "diameter_235um_flow_5mmps",
    "diameter_285um_flow_1mmps",
    "diameter_285um_flow_3mmps",
    "diameter_285um_flow_5mmps",
]
test_sample = "diameter_185um_flow_1mmps"  # 隐藏测试集，有图像无标签

data_dicts = []
for name in all_train_val_samples:
    img = os.path.join(args.data_dir, f"{name}_image.nii.gz")
    lbl = os.path.join(args.data_dir, f"{name}_label.nii.gz")
    if os.path.exists(img) and os.path.exists(lbl):
        label = nib.load(lbl).get_fdata()
        foreground_count = int(np.count_nonzero(label > 0))
        if foreground_count == 0:
            print(f"WARNING: {name} label has 0 foreground voxels, skipped: {lbl}")
            continue
        data_dicts.append({"image": img, "label": lbl, "name": name})
        print(f"Valid sample: {name}, label foreground voxels={foreground_count}")
    else:
        print(f"WARNING: missing {name}, skipped")

if len(data_dicts) < 2:
    raise RuntimeError(f"Need at least 2 valid annotated samples after filtering empty labels, got {len(data_dicts)}")

with open(json_path, 'w') as f:
    json.dump(data_dicts, f, indent=2)
print(f"Total {len(data_dicts)} valid annotated samples for CV: {[d['name'] for d in data_dicts]}")


def run_data_sanity_check(samples):
    print("\n===== Data Sanity Check =====")
    for item in samples:
        img_nii = nib.load(item["image"])
        lbl_nii = nib.load(item["label"])
        image = img_nii.get_fdata(dtype=np.float32)
        label = lbl_nii.get_fdata()
        unique_values = np.unique(label)
        foreground_count = int(np.count_nonzero(label > 0))
        foreground_ratio = foreground_count / label.size if label.size > 0 else 0.0
        if unique_values.size > 20:
            unique_repr = f"{unique_values[:20].tolist()} ... total_unique={unique_values.size}"
        else:
            unique_repr = unique_values.tolist()

        print(f"\nname: {item['name']}")
        print(
            f"  image shape={image.shape}, min={image.min():.6g}, "
            f"max={image.max():.6g}, mean={image.mean():.6g}"
        )
        print(f"  label shape={label.shape}, unique values={unique_repr}")
        print(f"  label foreground voxel count={foreground_count}")
        print(f"  label foreground ratio={foreground_ratio:.8f}")
        if foreground_count == 0:
            print(f"  WARNING: label has zero foreground voxels: {item['label']}")


run_data_sanity_check(data_dicts)
if args.sanity_check_only:
    sys.exit(0)

# ---------------- 3D U-Net 模型（轻量+Dropout） ----------------
def build_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=args.dropout_rate
    ).to(device)
    return model


def binarize_label(label):
    if torch.is_tensor(label):
        return (label > 0).float()
    return (np.asarray(label) > 0).astype(np.float32)

# ---------------- 数据变换（含前景中心裁剪） ----------------
def get_transforms(is_train=True):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # 将标签二值化为 0/1
        Lambdad(keys=["label"], func=binarize_label),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ]

    if is_train:
        transforms.append(
            # 核心修改：用前景中心裁剪替代之前的中心+随机裁剪
            # 使用正负样本平衡 patch 裁剪
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=args.patch_size,
                pos=1,
                neg=0 if args.overfit_one else 1,
                num_samples=4 if args.overfit_one else 8,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            )
        )

        if not args.overfit_one:
            transforms.extend([
                RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=[0]),
                RandFlipd(keys=["image", "label"], prob=0.6, spatial_axis=[1]),
                RandRotated(
                    keys=["image", "label"],
                    range_x=np.deg2rad(10),
                    range_y=np.deg2rad(10),
                    range_z=np.deg2rad(5),
                    prob=0.2,
                    mode=("bilinear", "nearest"),
                ),
                RandZoomd(keys=["image", "label"], min_zoom=0.85, max_zoom=1.15, prob=0.7, mode=("trilinear", "nearest")),
                RandGaussianNoised(keys=["image"], prob=0.3, std=0.02),
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
            ])

    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)

# ---------------- 训练一个循环（用于交叉验证） ----------------
def train_one_fold(train_ds, val_ds, fold_idx):
    log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"fold_{fold_idx}"))
    model = build_model()
    loss_func = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best_dice = 0.0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        log_writer.add_scalar("Loss/train", epoch_loss, epoch)

        if (epoch + 1) % args.val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    val_images = batch_data["image"].to(device)
                    val_labels = batch_data["label"].to(device)
                    sample_name = batch_data.get("name", "unknown")
                    if isinstance(sample_name, (list, tuple)):
                        sample_name = sample_name[0]
                    val_outputs = sliding_window_inference(
                        val_images, roi_size=args.patch_size, sw_batch_size=1,
                        predictor=model, overlap=0.25, mode="gaussian"
                    )
                    val_probs = torch.sigmoid(val_outputs)
                    val_preds = (val_probs > 0.5).float()
                    pred_fg = int(val_preds.sum().item())
                    label_fg = int((val_labels > 0).sum().item())
                    print(
                        f"  Val sample {sample_name}: pred_fg={pred_fg}, "
                        f"label_fg={label_fg}, prob_min={val_probs.min().item():.6f}, "
                        f"prob_max={val_probs.max().item():.6f}, prob_mean={val_probs.mean().item():.6f}"
                    )
                    dice_metric(y_pred=val_preds, y=val_labels)
                val_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                log_writer.add_scalar("Dice/val", val_dice, epoch)
                fold_label = fold_idx + 1 if isinstance(fold_idx, int) else fold_idx
                print(f"Fold {fold_label} | Epoch {epoch+1:3d} | Val Dice: {val_dice:.4f}")
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"best_model_fold{fold_idx}.pth"))
        scheduler.step()

    log_writer.close()
    return best_dice

# ---------------- 留一交叉验证主循环 ----------------
if args.overfit_one:
    print("\n===== Overfit-One Debug Mode =====")
    print(f"Using sample as both train and val: {data_dicts[0]['name']}")
    train_ds = Dataset(data=[data_dicts[0]], transform=get_transforms(True))
    val_ds = Dataset(data=[data_dicts[0]], transform=get_transforms(False))
    dice = train_one_fold(train_ds, val_ds, "overfit_one")
    print(f"Overfit-one best Dice: {dice:.4f}")
    sys.exit(0)

print("\n===== Starting Leave-One-Out Cross-Validation =====")
all_dices = []
for i in range(len(data_dicts)):
    val_sample = data_dicts[i]
    train_samples = [data_dicts[j] for j in range(len(data_dicts)) if j != i]
    print(f"\n--- Fold {i+1}/{len(data_dicts)} | Val: {val_sample['name']} ---")

    train_ds = Dataset(data=train_samples, transform=get_transforms(True))
    val_ds = Dataset(data=[val_sample], transform=get_transforms(False))

    dice = train_one_fold(train_ds, val_ds, i)
    all_dices.append(dice)
    print(f"Fold {i+1} best Dice: {dice:.4f}")

print("\n===== Cross-Validation Results =====")
for idx, d in enumerate(all_dices):
    print(f"Fold {idx+1}: {d:.4f}")
print(f"Mean Dice: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")

# ---------------- 训练最终模型（全9个样本） ----------------
print("\n===== Training final model on all 9 samples =====")
full_ds = Dataset(data=data_dicts, transform=get_transforms(True))
final_loader = DataLoader(
    full_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=list_data_collate,
)

model_final = build_model()
optimizer_final = torch.optim.AdamW(model_final.parameters(), lr=args.lr)
scheduler_final = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=args.epochs)
loss_func = DiceCELoss(sigmoid=True)
best_loss = float('inf')

for epoch in range(args.epochs):
    model_final.train()
    epoch_loss = 0.0
    for batch_data in final_loader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        optimizer_final.zero_grad()
        outputs = model_final(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer_final.step()
        epoch_loss += loss.item()
    epoch_loss /= len(final_loader)
    print(f"Final Model - Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f}")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model_final.state_dict(), os.path.join(args.checkpoint_dir, "best_model_final.pth"))
        print("Saved new best final model.")
    scheduler_final.step()

print("Training completed! Final model saved as best_model_final.pth")
