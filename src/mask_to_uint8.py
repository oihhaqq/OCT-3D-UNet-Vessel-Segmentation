import nibabel as nib
import numpy as np

# ---------------- 用户配置 ----------------
# 输入预测文件（概率图或 float mask）
input_file = 'D:/OCT_Project/results/mouse_pred_auto.nii.gz'
# 输出 Slicer 可见的二值 mask
output_file = 'D:/OCT_Project/results/mouse_pred_uint8.nii.gz'
# 二值化阈值（针对概率图）
threshold = 0.3
# ----------------------------------------

# 读取预测 NIfTI
img = nib.load(input_file)
data = img.get_fdata()  # float32 array

# 二值化并转换为 uint8
mask_uint8 = (data >= threshold).astype(np.uint8) * 255  # 0 或 255

# 使用原图 affine/header 保存
nii_out = nib.Nifti1Image(mask_uint8, affine=img.affine, header=img.header)
nib.save(nii_out, output_file)

print(f"Saved Slicer-visible binary mask to: {output_file}")
print(f"Mask shape: {mask_uint8.shape}, dtype: {mask_uint8.dtype}, unique values: {np.unique(mask_uint8)}")