import os
from enum import Enum
from glob import iglob

import cv2
from cv2.typing import MatLike
from tqdm import tqdm


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def split_color(img: MatLike, color: Color) -> MatLike:
    match color:
        case Color.RED:
            img[:, :, 0] = 0
            img[:, :, 1] = 0
        case Color.GREEN:
            img[:, :, 0] = 0
            img[:, :, 2] = 0
        case Color.BLUE:
            img[:, :, 1] = 0
            img[:, :, 2] = 0

    return img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "train_images")
    output_dir_base = os.path.join(script_dir, "preprocessed_images")
    
    input_glob_pattern = os.path.join(input_dir, "**", "*.jpg")

    for file in tqdm(iglob(input_glob_pattern, recursive=True)):
        try:
            img = cv2.imread(file)
            if img is None:
                print(f"Warning: Could not read image {file}. Skipping.")
                continue
                
            relative_path = os.path.relpath(file, input_dir)
            save_file_base = os.path.join(output_dir_base, relative_path)
            
            output_subdir = os.path.dirname(save_file_base)
            os.makedirs(output_subdir, exist_ok=True)
            
            save_file_red = save_file_base.replace(".jpg", "_red.jpg")
            save_file_green = save_file_base.replace(".jpg", "_green.jpg")
            save_file_blue = save_file_base.replace(".jpg", "_blue.jpg")

            cv2.imwrite(save_file_red, split_color(img.copy(), Color.RED))
            cv2.imwrite(save_file_green, split_color(img.copy(), Color.GREEN))
            cv2.imwrite(save_file_blue, split_color(img.copy(), Color.BLUE))
        except Exception as e:
            print(f"Error processing file {file}: {e}")


if __name__ == "__main__":
    main()
