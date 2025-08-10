import albumentations as alb
import os
import json
import cv2

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                         bbox_params=alb.BboxParams(format='albumentations',
                                                    label_fields=['class_labels']))

img = cv2.imread(os.path.join('data', 'train', 'images', '1bcde66b-625d-11f0-a105-782b46b807cb.jpg'))

with open(os.path.join('data', 'train', 'labels', '1bcde66b-625d-11f0-a105-782b46b807cb.json'), 'r') as f:
    label = json.load(f)
