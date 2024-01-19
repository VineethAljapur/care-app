
import io
import cv2
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras

colormap = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.99, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.99, 0.0, 0.0],
    ]
)

colormap = colormap * 100
colormap = colormap.astype(np.uint8)

def get_percentages(mask):
    class_counts = np.bincount(mask.astype(np.int64).flatten())
    total_pixels = np.sum(class_counts[1:])

    # Exclude background
    class_percentages = class_counts[1:] / total_pixels * 100  

    classmap = {1: "cortex", 2: "fibers", 3: "filo", 4: "lamellar", 5: "lamfilo"}

    # Add legend text to the mask
    legend_text = [
        f"{classmap[i]}: {percentage:.2f}%"
        for i, percentage in enumerate(class_percentages, start=1)
    ]

    return legend_text

def decode_segmentation_masks(mask, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    pseudo_mask = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
        pseudo_mask[idx] = l

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.45, colored_mask, 0.55, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3), mask=None):
    titles = ["Image", "Prediction Mask", "Overlay"]
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    legend_text = get_percentages(mask)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(keras.utils.array_to_img(display_list[i]))
            # Adding legned to the prediction mask
            if i == 1:
                for j, text in enumerate(legend_text, start=1):
                    axes[i].text(
                        0.05,
                        1.0 - 0.05 * j,
                        text,
                        color=list(colormap[j]/100),
                        fontsize=10,
                        transform=axes[i].transAxes,
                        va="top",
                        ha="left",
                    )
        else:
            axes[i].imshow(display_list[i])
        axes[i].title.set_text(titles[i])
    
    buf = io.BytesIO()
    plt.savefig(buf)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    return img