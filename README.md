# 2025 Summer Programming Project - Facial Recognition

This is a facial recognition software designed to draw a bounding box around visible faces it sees, coded in Python 3.12 (due to Tensorflow requiring Python 3.9-3.12).

## Dependencies Used

 - labelme
 - tensorflow-cpu
 - opencv-python
 - matplotlib
 - albumentations

## What I Learned

## Errors

 - Tensowflow and opencv-python couldn't be installed with pip
   - Fix: uninstall Python from Microsoft Store and reinstall (same version) from official website, adding python.exe to PATH
 - Couldn't open labelme GUI
   - https://chat.stackexchange.com/transcript/message/68115407#68115407
 - Couldn't find the right mix of drivers, cuDNN, CUDA, and TensorFlow-GPU packages
   - Used Tensorflow (v2.19.0) CPU for Windows 11 instead