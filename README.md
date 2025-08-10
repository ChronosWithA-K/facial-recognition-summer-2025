# 2025 Summer Programming Project - Facial Recognition

This is a facial recognition software designed to draw a bounding box around visible faces it sees, coded in Python 3.12.10 (due to TensorFlow requiring Python 3.9-3.12).

## Dependencies Used

 - labelme
 - tensorflow-cpu
 - opencv-python
 - matplotlib
 - albumentations

## What I Learned

 - Advantages of using a virtual environment (venv/virtualenv)
   - Allows me to install packages within the environment, without modifying the global installation
   - Allows for multiple different codebases with incompatible libraries without them interfering with each other 

## Errors

 - Tensowflow and opencv-python couldn't be installed with pip
   - Fix: uninstall Python from Microsoft Store and reinstall (same version) from official website, adding python.exe to PATH
 - Couldn't open labelme GUI
   - https://stackoverflow.com/questions/79705046/running-labelme-in-vs-code-terminal-gets-importerror-dll-load-failed-while-impo/79723660#79723660
 - Couldn't find the right mix of drivers, cuDNN, CUDA, and TensorFlow-GPU packages
   - Used Tensorflow (v2.19.0) CPU for Windows 11 instead