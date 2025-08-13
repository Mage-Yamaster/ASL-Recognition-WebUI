#!/usr/bin/python3
import jetson_inference
import jetson_utils

import StyleUtil as ST
import sys
import os
import cv2
import argparse

pathName = os.path.dirname(__file__) + "/googlenet-ASL.onnx"

ST.printr(ST.getBig("Starting"),255,0,0)
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str,default=None, help="filename of the input image to classify (default: capture from camera)")
parser.add_argument("--output", type=str, default="output.jpg", help="filename of the output image to save the classification result (default: output.jpg)")
parser.add_argument("--network", type=str, default="googlenet", help="model network to use (googlenet, resnet18, resnet50, etc.)")
parser.add_argument("--path", type=str, default=pathName, help="path to the model file.")

opt = parser.parse_args()
basefile = opt.filename

if opt.filename is None:
    ST.printr("CAPTURING! WAIT!", 255, 0, 0)

    # Open camera only just before capture
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        ST.printr("Error: Could not open camera.", 255, 0, 0)
        sys.exit(114514)
    ret, frame = cap.read()
    if not ret:
        ST.printr(ST.getPipeBox("Failed to grab frame."), 255, 0, 0)
        cap.release()
        sys.exit(810)

    # save image
    filename = "captured.jpg"
    cv2.imwrite(filename, frame)
    cap.release()

    basefile = "captured.jpg"
else:
    if not os.path.isfile(opt.filename):
        ST.printr("File not found: " + opt.filename, 255, 0, 0)
        sys.exit(114514)
    basefile = opt.filename

# start processing
ST.printr("Loaded file : " + basefile, 0, 250, 256)

img = jetson_utils.loadImage(basefile)

net = jetson_inference.imageNet(opt.network,
                                    model=opt.path ,    
                                    input_blob="input_0",
                                    output_blob="output_0",
                                    labels="labels.txt")


class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

ST.printr(ST.getBig("DONE!!!"),60,255,0)
ST.printr(ST.getPipeBox("RESULT",2,2,1),255,0,0)
ST.printr("This image recognized as "+str(class_desc),0,60,255)
ST.printr("Class #"+str(class_idx),0,60,255)
ST.printr("Confidence "+str(confidence*100)+"%",0,240,255)
ST.printr("Loaded File "+basefile,235,52,232)


text = str(ST.truncate_float((confidence*100),2))+"% "+str(class_desc)

Tsize= img.width * 2 / len(text) -2

if(Tsize > 30):
    Tsize = 30

font = jetson_utils.cudaFont(size=Tsize)

x = 5
y = 5

# Overlay
font.OverlayText(img, img.width, img.height, text, x, y, color=(0, 240, 255, 180),background=(0,0,0,120))

# Save
jetson_utils.saveImage(opt.output, img)