# 2025 Summer Programming Project - Facial Recognition

This is a facial recognition software designed to draw a bounding box around visible faces it sees, coded in Python 3.12 (due to Tensorflow requiring Python 3.9-3.12).

## Dependencies Used

 - labelme
 - tensorflow
 - opencv-python
 - matplotlib
 - albumentations

## What I Learned

## Errors

 - Tensorflow doesn't support Python 3.13
   - Fix: downgrade to Python 3.12.10
 - Tensowflow and opencv-python couldn't be installed with pip, labelme not recognised as command
   - Fix: uninstall Python from Microsoft and reinstall (same version) from official website, adding python.exe to PATH
