
# ComputerVision_FaceRecognition

This repository aims for computer vision projects especially for face recognition using AI models such as GFPGAN, Real-ESRGAN, and ResNet50.

**AI models used:-**

* [GFPGAN](https://github.com/TencentARC/GFPGAN)
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

* [YOLOv8](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

* [ResNet50](https://huggingface.co/microsoft/resnet-50)

for Documentation refer to Colab Demo below
## Features and Functions

* Image Processing: Upscale and enhance images using GFPGAN and Real-ESRGAN.

* Facial Recognition: Embed and save facial details like a unique fingerprint using ResNet50.
* YOLOv8 used for object and face detection
* Light weight project, can even run without gpu


## Goolge Colab Demo

Copy This [Google Colab](https://colab.research.google.com/drive/1VdzxOWoytvJlzVif1FfQcPRPUwtpJjIT#scrollTo=A3GNMozHhSFH)  notebook for Demo

## Install and Run Locally

* Prerequisites
requires
Python 3.x
Jupyter Notebook or Google Colab
, gpu recommended.
```bash
  git clone https://github.com/DivyeshVora79909/ComputerVision_FaceRecognition
```

Go to the project directory

```bash
  cd ComputerVision_FaceRecognition
```

Install Required libraries (install via pip):
```bash
  pip install tensorflow realesrgan basicsr huggingface-hub ultralytics gfpgan pillow matplotlib mtcnn numpy
```

For CUDA PyTorch, install CUDA-compatible version according to the PyTorch website. [Pytorch compatible CUDA](https://pytorch.org/get-started/locally/) For example:
```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
For CPU Pytorch (not recommended)
```bash
  pip install torch torchvision torchaudio
```
Open ComputerVision_FaceRecognition.ipynb
```bash
jupyter notebook your_file.ipynb
```
