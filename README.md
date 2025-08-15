# 🧠 ASL (American Sign Language) Classification on NVIDIA Jetson Orin Nano

## 📌 Overview and Logic
This project is a Python-based application designed to run on the **NVIDIA Jetson Orin Nano Developer Kit**. It performs real-time classification of American Sign Language (ASL) gestures using a camera feed or static image input. The system leverages **deep learning inference** powered by NVIDIA's optimized libraries: `jetson-inference` and `jetson-utils`.

This project has two main part.
1. **ASL-WebUI.py** is a Backend of website and it release image-file from website and give it to AI.
2. **ASLR_func.py** is a function of ASL Recognition AI. It used for classification of ASL in **ASL-WebUI.py**

### Using Model
This project is using Fine-tuned **googlenet** model.
---

## 🚀 Getting Started

### 🧩 Requirements

Ensure the following components are available and properly configured:

- **Hardware**: NVIDIA Jetson or NVIDIA GPUs
- **Software Stack**:
  - [JetPack SDK](https://developer.nvidia.com/embedded/jetpack) (includes CUDA, cuDNN, TensorRT)
  - `jetson-inference` and `jetson-utils` libraries
  - Python 3
  - OpenCV (`cv2`)
- **Model Files**:
  - Pre-trained ONNX model (e.g., `resnet18-ASL.onnx`)
  - Label file (`labels.txt`) — one class per line, matching the model's output order

---

### ⚙️ Installation Steps

Follow these steps to set up the environment and build the required libraries:

1.  Clone the `jetson-inference` repository go to cloned directory
```bash
git clone --recursive [https://github.com/dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
cd jetson-inference
```

2. Make and go to "build" directory then Configure the Project with CMake.
```bash
mkdir build
cd build

cmake ../
```

3. Run make to build and install the project.
```bash
make -j$(nproc) # use -j$(nproc) to build fast
sudo make install
```
And Jetson-Inferences setup is done.<br>
### OpenCV
4. Make sure the OpenCV installed on your device before run.
```bash
pip install opencv-python
```
5. Start it!
```bash
cd path/to/ASL-Recognition-WebUI
python3 ASL-WebUI.py
```
And link will appear in console.
```bash
 * Running on http://127.0.0.1:4040
```
Or you can run it in no WebUI
```bash
python3 ASL-recognition.py
```
Arguments
```bash
--filename : filepath of the input image to classify (default: capture from camera)
--output : filename of the output image to save the classification result (default: output.jpg)
```
If you want to use camera in this, connect it to running device.(ex. orin nano)
