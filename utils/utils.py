import os
import io
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from tensorflow import keras
from typing import List, Tuple, Union

colormap: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.99, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.99, 0.0, 0.0],
    ]
)
colormap = colormap * 100* 100
colormap = colormap.astype(np.uint8)


def get_percentages(mask: np.ndarray) -> List[str]:
    """
    Calculate the percentage of each class in the mask.

    Args:
        mask (ndarray): Segmentation mask.

    Returns:
        list: List of strings representing the class name and percentage for each class.
    """
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


def decode_segmentation_masks(mask: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Decode the segmentation mask into RGB color channels.

    Args:
        mask (ndarray): Segmentation mask.
        n_classes (int): Number of classes.

    Returns:
        ndarray: RGB image representing the colored mask.
    """
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


def get_overlay(image: Image.Image, colored_mask: np.ndarray) -> np.ndarray:
    """
    Create an overlay of the image and colored mask.

    Args:
        image (PIL.Image.Image): Input image.
        colored_mask (ndarray): RGB image representing the colored mask.

    Returns:
        ndarray: Overlay image.
    """
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.45, colored_mask, 0.55, 0)
    return overlay


def plot_samples_matplotlib(display_list: List[Union[np.ndarray, Image.Image]], figsize: Tuple[int, int] = (5, 3), mask: np.ndarray = None) -> Image.Image:
    """
    Plot a list of images using matplotlib.

    Args:
        display_list (list): List of images to be displayed.
        figsize (tuple, optional): Figure size. Defaults to (5, 3).
        mask (ndarray, optional): Segmentation mask. Defaults to None.

    Returns:
        PIL.Image.Image: Image of the plotted samples.
    """
    titles = ["Image", "Prediction Mask", "Overlay"]
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    legend_text = get_percentages(mask)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(keras.utils.array_to_img(display_list[i]))
            # Adding legend to the prediction mask
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


def create_pdf(image_folder: str, output_pdf: str) -> None:
    """
    Create a PDF document based on saved images.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_pdf (str): Path to the output PDF file.
    """
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
    c = canvas.Canvas(output_pdf, pagesize=letter)

    for image_path in image_files:

        _ = Image.open(image_path)

        # Get the size of the image (assuming it's in portrait orientation)
        width, height = letter[::-1]
        width /= 1.25
        height /= 1.25

        c.drawInlineImage(image_path, 0, 0, width, height)
        
        c.setFont("Helvetica", 18)
        title = f"{os.path.dirname(image_path).split('/')[-1]}: {os.path.basename(image_path)}"
        c.drawCentredString(width / 2, height - 20, title)

        c.showPage()

    c.save()