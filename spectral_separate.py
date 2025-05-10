from enum import Enum
from glob import iglob
import os 

import cv2
from cv2.typing import MatLike
from tqdm import tqdm
import matplotlib.pyplot as plt


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def disable_color(img: MatLike, color: Color) -> MatLike:
    img = img.copy()
    match color:
        case Color.RED:
            img[:, :, 2] = 0
        case Color.GREEN:
            img[:, :, 1] = 0
        case Color.BLUE:
            img[:, :, 0] = 0

    return img


def main():
    for file in tqdm(iglob(".Dataset/test_images/*.jpg", recursive=True)):
        img = cv2.imread(file)
        save_file = file.replace("Dataset/test_images", "Dataset/preprocessed_test_spectral_images")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        # cv2.imwrite(
        #     save_file.replace(".jpg", "_red.jpg"), disable_color(img, Color.RED)
        # )
        # cv2.imwrite(
        #     save_file.replace(".jpg", "_green.jpg"),
        #     disable_color(img, Color.GREEN),
        # )
        # cv2.imwrite(
        #     save_file.replace(".jpg", "_blue.jpg"), disable_color(img, Color.BLUE)
        # )
        plt.imsave(
            save_file.replace(".jpg", "_nipy_spectral.jpg"),
            img.copy()[:, :, 0],
            cmap="nipy_spectral",
        )


if __name__ == "__main__":
    main()
