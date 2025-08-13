#!/usr/bin/python3
import jetson_inference
import jetson_utils

import sys
import os
import cv2
import argparse

def truncate_float(number, decimals):
    """
    Truncates a float to a specified number of decimal places.

    Args:
        number (float): The float to truncate.
        decimals (int): The number of decimal places to keep.

    Returns:
        float: The truncated float.
    """
    factor = 10 ** decimals
    return int(number * factor) / factor

def classify_image(basefile, output=False):
    # Load the image
    img = jetson_utils.loadImage(basefile)

    # Load the model
    pathName = os.path.dirname(__file__) + "/googlenet-ASL.onnx"
    net = jetson_inference.imageNet("googlenet",
                                     model=pathName,
                                     input_blob="input_0",
                                     output_blob="output_0",
                                     labels="labels.txt")

    # Classify the image
    class_idx, confidence = net.Classify(img)
    class_desc = net.GetClassDesc(class_idx)

    if output:
        text = str(truncate_float((confidence*100),2))+"% "+str(class_desc)

        Tsize= img.width * 2 / len(text) -2

        if(Tsize > 30):
            Tsize = 30

        font = jetson_utils.cudaFont(size=Tsize)

        x = 5
        y = 5

        # Overlay
        font.OverlayText(img, img.width, img.height, text, x, y, color=(0, 240, 255, 180),background=(0,0,0,120))

        # Save
        jetson_utils.saveImage("outputUI.jpg", img)

    return class_desc, class_idx, confidence