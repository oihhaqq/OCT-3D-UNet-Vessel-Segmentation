import os
import re
import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ---------- 工具函数 ----------
def natural_key(s):
    """自然排序用，按文件名中的数字大小排序"""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def fill_ellipse_contour(img_bgr):
    """
    处理彩色标注图像（绿色轮廓线），返回填充的二值掩膜 (0/255)。
    来源于 prepare_2d_data_v2.py 的成熟逻辑。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 根据你的标注绿色范围进行阈值化（可调节）
    lower_green = np.array([42, 14, 0])
    upper_green = np.array([73, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 形态学闭运算，连接断线
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓并填充最大轮廓（血管管腔）
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_green, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_filled, [largest], -1, 255, thickness=cv2.FILLED)
    return mask_filled

def volume_from_slices(slices_dir, out_path, is_mask=False, fill_contour=False):
    """
    将切片堆叠为 3D NIfTI 体积。
    - fill_contour: 读取彩色标注图并填充轮廓（生成 label）
    - is_mask: 读取二值掩膜 png（如从 mask_png 文件夹）
    输出：标签为 uint8 0/255，图像为 float32 0-255
    """
    files = sorted([f for f in os.listdir(slices_dir) if f.endswith('.png')], key=natural_key)
    if not files:
        print(f"No PNG found in {slices_dir}")
        return None

    slices = []
    for f in files:
        file_path = os.path.join(slices_dir, f)

        if fill_contour:
            # 以彩色模式读取标注图，提取绿色轮廓并填充
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            mask = fill_ellipse_contour(img)   # 返回 0/255 uint8
            slices.append(mask)

        elif is_mask:
            # 直接读取二值掩膜 png，转换为 0/255
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            slices.append(mask)

        else:
            # 原始图像 (tailed) 读取为灰度，归一化到 0-255
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            slices.append(img_normalized.astype(np.float32))

    # 堆叠为 3D 体积 (H, W, D)
    if fill_contour or is_mask:
        vol = np.stack(slices, axis=-1).astype(np.uint8)
    else:
        vol = np.stack(slices, axis=-1).astype(np.float32)

    # 保存为 NIfTI（affine 暂用单位矩阵，可后续用 dcm 头信息替换）
    nii = nib.Nifti1Image(vol, np.eye(4))
    nib.save(nii, out_path)
    return vol

# ---------- 主处理逻辑 ----------
def main():
    # 请根据你的本地 Windows 数据路径调整 raw_root
    raw_root = "D:/OCT_Project/raw_data/phantom"
    out_root = "D:/OCT_Project/3d_processed"

    diameters = ['diameter_074um', 'diameter_128um', 'diameter_185um',
                 'diameter_235um', 'diameter_285um']
    flows = ['flow_1mmps', 'flow_3mmps', 'flow_5mmps']

    os.makedirs(out_root, exist_ok=True)

    for dia in tqdm(diameters, desc="Processing diameters"):
        for flow in flows:
            case_dir = os.path.join(raw_root, dia, flow)
            if not os.path.isdir(case_dir):
                continue

            tailed_dir = os.path.join(case_dir, 'tailed')
            if not os.path.isdir(tailed_dir):
                print(f"Warning: No tailed folder for {dia}/{flow}, skip")
                continue

            # 生成输出文件名（文件名中包含直径和流速信息）
            prefix = f"{dia}_{flow}"
            img_out = os.path.join(out_root, f"{prefix}_image.nii.gz")
            print(f"Creating {img_out}")
            volume_from_slices(tailed_dir, img_out, is_mask=False)

            # 检查标签来源：优先 annotation，其次 mask_png
            annot_dir = os.path.join(case_dir, 'annotation')
            mask_dir = os.path.join(case_dir, 'mask_png')
            label_out = os.path.join(out_root, f"{prefix}_label.nii.gz")

            if os.path.isdir(annot_dir):
                print(f"Creating label from annotation: {label_out}")
                volume_from_slices(annot_dir, label_out, fill_contour=True)
            elif os.path.isdir(mask_dir):
                print(f"Creating label from mask_png: {label_out}")
                volume_from_slices(mask_dir, label_out, is_mask=True)
            else:
                print(f"Warning: No label found for {dia}/{flow}, skip label.")

    print("All done! Check 3D NIfTI files.")


if __name__ == "__main__":
    main()