import os
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

META_CSV = "Dataset/meta_train.csv"
IMG_SIZE = (128, 128)
TASK1_IMG_SIZE = (480, 640)


task1 = load_model("task1.keras")
task2 = load_model("task2_4colors.keras")
task3 = load_model("task3_3colors.keras")


meta_df = pd.read_csv(META_CSV)
variety_labels = meta_df["variety"].unique()
label_labels = np.load("task1_le.npy", allow_pickle=True)


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


def image_transformation(image_id: str) -> str:
    filename = os.path.join("./Dataset/test_images", f"{image_id}.jpg")
    img = cv2.imread(filename)

    cv2.imwrite(filename.replace(".jpg", "_red.jpg"), disable_color(img, Color.RED))
    cv2.imwrite(
        filename.replace(".jpg", "_green.jpg"),
        disable_color(img, Color.GREEN),
    )
    cv2.imwrite(filename.replace(".jpg", "_blue.jpg"), disable_color(img, Color.BLUE))
    plt.imsave(
        filename.replace(".jpg", "_nipy_spectral.jpg"),
        img.copy()[:, :, 0],
        cmap="nipy_spectral",
    )

    return filename


def task1_predict(image_id: str) -> str:
    img = cv2.imread(os.path.join("./Dataset/test_images", f"{image_id}.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TASK1_IMG_SIZE)
    img = img / 255.0

    prediction = task1.predict(np.expand_dims(img, axis=0))
    predicted_index = np.argmax(prediction)
    return label_labels[predicted_index]


def task2_predict(image_id: str) -> str:
    red = (
        img_to_array(
            load_img(
                os.path.join("./Dataset/test_images", f"{image_id}_red.jpg"),
                target_size=IMG_SIZE,
            )
        )
        / 255.0
    )
    green = (
        img_to_array(
            load_img(
                os.path.join("./Dataset/test_images", f"{image_id}_green.jpg"),
                target_size=IMG_SIZE,
            )
        )
        / 255.0
    )
    blue = (
        img_to_array(
            load_img(
                os.path.join("./Dataset/test_images", f"{image_id}_blue.jpg"),
                target_size=IMG_SIZE,
            )
        )
        / 255.0
    )
    spectral = (
        img_to_array(
            load_img(
                os.path.join("./Dataset/test_images", f"{image_id}_nipy_spectral.jpg"),
                target_size=IMG_SIZE,
                color_mode="grayscale",
            )
        )
        / 255.0
    )
    spectral = np.repeat(spectral, 3, axis=-1)  # Convert grayscale to 3 channels

    # Stack into (128, 128, 12)
    img_array = np.concatenate([red, green, blue, spectral], axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = task2.predict(img_array)
    predicted_index = np.argmax(prediction)
    return variety_labels[predicted_index]


def task3_predict(image_id: str) -> float:
    r = cv2.imread(
        os.path.join("./Dataset/test_images", f"{image_id}_red.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    g = cv2.imread(
        os.path.join("./Dataset/test_images", f"{image_id}_green.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    b = cv2.imread(
        os.path.join("./Dataset/test_images", f"{image_id}_blue.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    r = cv2.resize(r, IMG_SIZE)
    g = cv2.resize(g, IMG_SIZE)
    b = cv2.resize(b, IMG_SIZE)

    stacked = np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0

    img_input = np.expand_dims(stacked, axis=0)

    predict = task3.predict(img_input, verbose=0)[0][0]

    return float(round(predict, 2))


def main():
    predict_df = pd.read_csv("prediction_template.csv")

    for i, row in predict_df.iterrows():
        image_id = row["image_id"].replace(".jpg", "")
        image_transformation(image_id)

        label = task1_predict(image_id)
        variety = task2_predict(image_id)
        age = task3_predict(image_id)

        predict_df.at[i, "label"] = label
        predict_df.at[i, "variety"] = variety
        predict_df.at[i, "age"] = age

    predict_df.to_csv("prediction_all.csv", index=False)


if __name__ == "__main__":
    main()
