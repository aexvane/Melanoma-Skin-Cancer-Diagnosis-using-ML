import numpy as np
import shutil
import os
from tqdm import tqdm
import cv2

directory = "./test/"
dimension = (256, 256)

images = os.listdir(directory)

for image in tqdm(range(len(images))):
    img = cv2.imread(f"./test/{images[image]}")
    resized = cv2.resize(img, dimension)
    cv2.imwrite(f"resizedtest/{images[image]}", resized)