#!/usr/bin/env python
# coding: utf-8

import io
import gc
import json
import torch
import tifffile
import cv2

import numpy as np
from flask import Flask, request, render_template, Response

from keras.models import load_model

# Custom visualization functions
import utils

# Load the model
model = load_model("model.keras")
config = json.load(open("config.json", "r"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Get the image from the user
    img = request.files["image"]
    image = tifffile.imread(img)

    # Convert to grayscale if RGB. Selecting green channel
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = image[:, :, 1]

    # If the uploaded tiff file is not uint16, convert it to uint16
    factor = 1
    if image.dtype == np.uint8:
        factor = 256

    # get these from config.json
    padding = config["padding"]
    threshold = config["threshold"]
    detection_threshold = config["detection_threshold"]
    block_size = config["block_size"]

    x_positions = []
    y_positions = []

    class_predictions_list = []
    class_values_list = []

    image = image[padding : (image.shape[0] - padding), padding : (image.shape[1] - padding), np.newaxis]
    cropped_image = image.copy()

    image = np.divide(image, np.max(image))

    cropped_image_list = []

    x_positions, y_positions = zip(
        *[
            (x, y)
            for x in range(block_size + 1, image.shape[0] - block_size - 1)
            for y in range(block_size + 1, image.shape[1] - block_size - 1)
            if image[x, y] > threshold
        ]
    )

    cropped_image_list = [cropped_image[x - block_size : x + block_size, y - block_size : y + block_size] * factor for x, y in zip(x_positions, y_positions)]

    if cropped_image_list:
        cropped_image_array = np.dstack(cropped_image_list)  # (2*block_size, 2*block_size, X)
        cropped_image_array = np.rollaxis(cropped_image_array, -1)  # (X, 2*block_size, 2*block_size)
        cropped_image_array = np.expand_dims(cropped_image_array, axis=3)  # (X, 2*block_size, 2*block_size, 1)

        gc.collect()

        predictions = model.predict(cropped_image_array, verbose=0)

        torch.cuda.empty_cache()

        class_predictions = np.argmax(predictions, axis=1)
        class_values = np.max(predictions, axis=1)

        class_values_list.extend(class_values)
        class_predictions_list.extend(class_predictions)

    class_predictions_list_filtered = []
    x_positions_filtered = []
    y_positions_filtered = []

    _, class_predictions_list_filtered, x_positions_filtered, y_positions_filtered = zip(
        *[(a, b, c, d) for a, b, c, d in zip(class_values_list, class_predictions_list, x_positions, y_positions) if a >= detection_threshold]
    )

    class_max_image_filtered = np.zeros((image.shape[0], image.shape[1]))
    for n in range(1, len(x_positions_filtered)):
        class_max_image_filtered[x_positions_filtered[n], y_positions_filtered[n]] = class_predictions_list_filtered[n] + 1

    prediction_colormap = utils.decode_segmentation_masks(class_max_image_filtered, 6)

    overlay = utils.get_overlay(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB), prediction_colormap)

    img_rgb = utils.plot_samples_matplotlib([cropped_image, prediction_colormap, overlay], (18, 14), class_max_image_filtered)

    # Convert the image to a byte array
    buffer = io.BytesIO()
    img_rgb.save(buffer, format="PNG")
    buffer.seek(0)
    img_bytes = buffer.getvalue()

    # Return the uploaded image to the browser
    return Response(img_bytes, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
