import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_nifti_data(file_path):
    img = nib.load(str(file_path))
    return img.get_fdata(dtype=np.float32)


def print_volume_stats(file_path, data):
    nonzero_count = int(np.count_nonzero(data))

    print("=" * 80)
    print(f"File: {file_path.name}")
    print(f"Path: {file_path}")
    print(f"Shape: {data.shape}")
    print(f"Min: {float(np.min(data)):.6f}")
    print(f"Max: {float(np.max(data)):.6f}")
    print(f"Mean: {float(np.mean(data)):.6f}")
    print(f"Non-zero voxels: {nonzero_count}")

    if nonzero_count == 0:
        print("Warning: this prediction appears to be empty.")


def get_middle_z_slice(data):
    if data.ndim == 3:
        z_mid = data.shape[2] // 2
        return data[:, :, z_mid], z_mid

    if data.ndim == 4:
        squeezed = np.squeeze(data)
        if squeezed.ndim == 3:
            z_mid = squeezed.shape[2] // 2
            return squeezed[:, :, z_mid], z_mid

    raise ValueError(f"Expected 3D data, got shape {data.shape}")


def show_middle_slice(file_path, data, cmap="hot"):
    middle_slice, z_mid = get_middle_z_slice(data)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(middle_slice.T, cmap=cmap, origin="lower")
    plt.colorbar(im, label="Probability / value")
    plt.title(f"{file_path.name} | z middle slice = {z_mid}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Check selected 3D U-Net NIfTI predictions and visualize middle slices."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=r"D:\OCT_Project\results",
        help="Directory containing the selected prediction files.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="hot",
        help="Matplotlib colormap for probability heatmaps.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    target_filenames = [
        "mouse_pred_auto.nii.gz",
        "mouse_pred_binary_0.3.nii.gz",
        "mouse_pred_uint8.nii.gz",
    ]
    target_files = [results_dir / name for name in target_filenames]

    print(f"Checking {len(target_files)} selected NIfTI file(s) in: {results_dir}")

    for file_path in target_files:
        if not file_path.exists():
            print("=" * 80)
            print(f"File: {file_path.name}")
            print(f"Path: {file_path}")
            print("Warning: file does not exist, skipped.")
            continue

        data = load_nifti_data(file_path)
        print_volume_stats(file_path, data)
        show_middle_slice(file_path, data, cmap=args.cmap)


if __name__ == "__main__":
    main()
