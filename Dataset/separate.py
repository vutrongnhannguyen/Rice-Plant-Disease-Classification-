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
    for file in tqdm(iglob("./train_images/**/*.jpg", recursive=True)):
        img = cv2.imread(file)
        save_file = file.replace("train_images", "preprocessed_images")
        cv2.imwrite(
            save_file.replace(".jpg", "_red.jpg"), split_color(img.copy(), Color.RED)
        )
        cv2.imwrite(
            save_file.replace(".jpg", "_green.jpg"),
            split_color(img.copy(), Color.GREEN),
        )
        cv2.imwrite(
            save_file.replace(".jpg", "_blue.jpg"), split_color(img.copy(), Color.BLUE)
        )


if __name__ == "__main__":
    main()
