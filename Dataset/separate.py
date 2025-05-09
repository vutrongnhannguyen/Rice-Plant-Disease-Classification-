# !pip install split-folders tqdm opencv-python-headless matplotlib

import splitfolders
import os
import cv2
from enum import Enum
from glob import iglob
from tqdm import tqdm
import matplotlib.pyplot as plt

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def disable_color(img, color: Color):
    img = img.copy()
    if color == Color.RED:
        img[:, :, 2] = 0
    elif color == Color.GREEN:
        img[:, :, 1] = 0
    elif color == Color.BLUE:
        img[:, :, 0] = 0
    return img

def apply_color_preprocessing(input_dir, output_dir):
    """
    For each .jpg in input_dir, generate 4 preprocessed images (red, green, blue disabled, and nipy_spectral)
    and save them to output_dir, preserving folder structure.
    """
    for file in tqdm(iglob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)):
        img = cv2.imread(file)
        if img is None:
            continue
        save_file = file.replace(input_dir, output_dir)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        cv2.imwrite(save_file.replace(".jpg", "_red.jpg"), disable_color(img, Color.RED))
        cv2.imwrite(save_file.replace(".jpg", "_green.jpg"), disable_color(img, Color.GREEN))
        cv2.imwrite(save_file.replace(".jpg", "_blue.jpg"), disable_color(img, Color.BLUE))
        plt.imsave(save_file.replace(".jpg", "_nipy_spectral.jpg"), img[:, :, 0], cmap="nipy_spectral")

def apply_test_color_preprocessing(test_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm(iglob(os.path.join(test_dir, "*.jpg"))):
        img = cv2.imread(file)
        if img is None:
            continue
        base = os.path.basename(file)
        name, ext = os.path.splitext(base)
        cv2.imwrite(os.path.join(output_dir, f"{name}_red{ext}"), disable_color(img, Color.RED))
        cv2.imwrite(os.path.join(output_dir, f"{name}_green{ext}"), disable_color(img, Color.GREEN))
        cv2.imwrite(os.path.join(output_dir, f"{name}_blue{ext}"), disable_color(img, Color.BLUE))
        plt.imsave(os.path.join(output_dir, f"{name}_nipy_spectral{ext}"), img[:, :, 0], cmap="nipy_spectral")

# Split train_images into train and val (preserving subfolders)
splitfolders.ratio(
    "./train_images",
    output="./split_images",
    seed=42,
    ratio=(.8, .2)
)

# Apply color preprocessing to each split and to test set
apply_color_preprocessing("./split_images/train", "./train_preprocessed_images")
apply_color_preprocessing("./split_images/val", "./val_preprocessed_images")
apply_test_color_preprocessing("test_images", "test_preprocessed_images")

print("Splitting and preprocessing completed.")
