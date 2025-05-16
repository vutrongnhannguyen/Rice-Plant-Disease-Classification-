import os
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from flask import Flask, abort, flash, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

task1 = load_model("task1.keras")
task2 = load_model("task2_4colors.keras")
task3 = load_model("task3_3colors.keras")

META_CSV = "Dataset/meta_train.csv"
IMG_SIZE = (128, 128)
TASK1_IMG_SIZE = (480, 640)
UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"jpg"}

meta_df = pd.read_csv(META_CSV)
variety_labels = meta_df["variety"].unique()
label_labels = np.load("task1_le.npy", allow_pickle=True)

app = Flask(__name__, static_folder="app/dist", static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "aoidhwioadawhodhuiwdh"


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def allowed_file(filename) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
    filename = os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}.jpg")
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
    img = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}.jpg"))
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
                os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}_red.jpg"),
                target_size=IMG_SIZE,
            )
        )
        / 255.0
    )
    green = (
        img_to_array(
            load_img(
                os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}_green.jpg"),
                target_size=IMG_SIZE,
            )
        )
        / 255.0
    )
    blue = (
        img_to_array(
            load_img(
                os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}_blue.jpg"),
                target_size=IMG_SIZE,
            )
        )
        / 255.0
    )
    spectral = (
        img_to_array(
            load_img(
                os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{image_id}_nipy_spectral.jpg"
                ),
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
        os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}_red.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    g = cv2.imread(
        os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}_green.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    b = cv2.imread(
        os.path.join(app.config["UPLOAD_FOLDER"], f"{image_id}_blue.jpg"),
        cv2.IMREAD_GRAYSCALE,
    )
    r = cv2.resize(r, IMG_SIZE)
    g = cv2.resize(g, IMG_SIZE)
    b = cv2.resize(b, IMG_SIZE)

    stacked = np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0

    img_input = np.expand_dims(stacked, axis=0)

    predict = task3.predict(img_input, verbose=0)[0][0]

    return float(round(predict, 2))


@app.route("/api/predict", methods=["POST"])
def upload_file():
    # check if the post request has the file part
    if "file" not in request.files:
        flash("No file part")
        return abort(400)

    file = request.files["file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if not file.filename or file.filename == "":
        flash("No selected file")
        return abort(400)

    if not allowed_file(file.filename):
        return abort(400)

    filename = secure_filename(file.filename)
    image_id = filename.replace(".jpg", "")
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

    image_transformation(image_id)
    print("Transformed uploaded image into common formats")

    task1_result = task1_predict(image_id)
    print("Predicted values for task 1")

    task2_result = task2_predict(image_id)
    print("Predicted values for task 2")

    task3_result = task3_predict(image_id)
    print("Predicted values for task 3")

    return {"label": task1_result, "variety": task2_result, "age": task3_result}


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    return app.send_static_file("index.html")


def main():
    app.run()


if __name__ == "__main__":
    main()
