import albumentations as alb
import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
    print(label)

coords = [0, 0, 0, 0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [640, 480, 680, 480]))

augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)),
              (255, 0, 0), 2)

plt.imshow(augmented['image'])
plt.show()

for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')): # Loop through every image
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0, 0, 0.00001, 0.00001] # Create default annotation
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path): # Check an annotation exists (not all images have them)
            with open(label_path, 'r') as f:
                label = json.load(f)
            
            # Assuming a set of coordinates exist, transform image
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]

            coords = list(np.divide(coords, [640, 480, 680, 480]))
        
        try:
            for x in range(60): # Create 60 augmentations per 90 images
                # Run data through augmentation pipeline
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                bbox = augmented['bboxes'][0]
                x_min, y_min, x_max, y_max = bbox[:4]

                if x_max > x_min and y_max > y_min:
                    cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        annotation['bbox'] = [x_min, y_min, x_max, y_max]
                        annotation['class'] = 0

                        with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                            json.dump(annotation, f)
                else:
                    print(f"Invalid bbox: {bbox}")
        except Exception as e:
            print(e)