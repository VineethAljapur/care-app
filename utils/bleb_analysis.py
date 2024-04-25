#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict, Any

def find_blebs(image: np.ndarray, outer: bool = True) -> List[np.ndarray]:
    """
    Find blebs in the given image.

    Args:
        image (np.ndarray): The input image.
        outer (bool, optional): Flag to indicate whether to find outer blebs only. Defaults to True.

    Returns:
        List[np.ndarray]: List of contours representing the blebs.
    """
    _, binary_image = cv2.threshold(image, 3, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def analyze_bleb(contour: np.ndarray) -> dict:
    """
    Analyze the properties of a bleb contour.

    Args:
        contour (np.ndarray): The contour representing a bleb.

    Returns:
        dict: Dictionary containing the analyzed properties of the bleb.
    """
    area: float = cv2.contourArea(contour)
    perimeter: float = cv2.arcLength(contour, True)

    circularity: float = 0
    if perimeter:
        circularity = 4 * np.pi * area / (perimeter * perimeter)

    _, _, width, height = cv2.boundingRect(contour)

    major_diameter: float = max(width, height)
    minor_diameter: float = min(width, height)

    return {
      "area": area,
      "circularity": circularity,
      "major_diameter": major_diameter,
      "minor_diameter": minor_diameter
    }

def display_blebs(bleb_array: np.ndarray) -> None:
    """
    Display the blebs on an image.

    Args:
        bleb_array (np.ndarray): The binary image containing the blebs.
    """
    contours, _ = cv2.findContours(bleb_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = bleb_array.shape
    display_image = np.zeros((height, width, 3), np.uint8)
    
    bleb_number = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 100:
            color = (0, 255, 0)
            cv2.drawContours(display_image, [cnt], 0, color, 2)


            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7 
            text_thickness = 2 
            cv2.putText(display_image, str(bleb_number), (x, y - 5), font, font_scale, (255, 255, 255), text_thickness)
            bleb_number += 1
            
    plt.imshow(display_image)
    plt.show()


def write_to_excel(filename: str, data_dicts: List[Dict[str, List[Any]]], sheet_names: List[str]) -> None:
    """
    Write data to an Excel file.

    Args:
        filename (str): The name of the Excel file.
        data_dicts (List[Dict[str, List[Any]]]): List of dictionaries containing the data to be written.
        sheet_names (List[str]): List of sheet names corresponding to each data dictionary.
    """
    with pd.ExcelWriter(filename) as writer:
        for i, df in enumerate(data_dicts):
            max_length = max(len(df[a]) for a in df)

            for col in df:
                df[col] += [np.nan] * (max_length - len(df[col]))

            df_temp = pd.DataFrame(df)
            df_temp.to_excel(writer, sheet_name=sheet_names[i], index=False)

