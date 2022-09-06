import imgaug
import cv2
import os
from tqdm import tqdm
import numpy as np


seq = [imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1),
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1),
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.GaussianBlur(sigma=(5.0, 6.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.GaussianBlur(sigma=(6.0, 8.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.GaussianBlur(sigma=(8.0, 10.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MedianBlur(k=(5, 9))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MedianBlur(k=(9, 13))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MedianBlur(k=(13, 17))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.LinearContrast((0.4, 0.7))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.LinearContrast((0.7, 1.2))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.LinearContrast((1.2, 1.6))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.GammaContrast((0.5, 0.7))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.GammaContrast((0.7, 1.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.GammaContrast((1.0, 1.25))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.LogContrast(gain=(0.6, 0.8))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.LogContrast(gain=(0.8, 1.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.LogContrast(gain=(1.0, 1.25))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.CLAHE()
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyAndAddToBrightness(mul=(0.5, 0.7), add=(0, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyAndAddToBrightness(mul=(0.7, 1.0), add=(0, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyAndAddToBrightness(mul=(1.0, 1.25), add=(0, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyAndAddToBrightness(mul=(1.0, 1.0), add=(-5, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyAndAddToBrightness(mul=(1.0, 1.0), add=(0, 5))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyAndAddToBrightness(mul=(1.0, 1.0), add=(5, 10))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyHue((0.5, 0.7))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyHue((0.7, 1.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplyHue((1.0, 1.25))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplySaturation((0.5, 0.7))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplySaturation((0.7, 1.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.MultiplySaturation((1.0, 1.25))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ChangeColorTemperature((4000, 8000))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ChangeColorTemperature((8000, 10000))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ChangeColorTemperature((1000, 12000))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.GaussianBlur(sigma=(5.0, 6.0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.GaussianBlur(sigma=(5.0, 6.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.GaussianBlur(sigma=(6.0, 8.0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.GaussianBlur(sigma=(6.0, 8.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.GaussianBlur(sigma=(8.0, 10.0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.GaussianBlur(sigma=(8.0, 10.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.LinearContrast((0.4, 0.7))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.LinearContrast((0.4, 0.7))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.LinearContrast((0.7, 1.2))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.LinearContrast((0.7, 1.2))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.LinearContrast((1.2, 1.6))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.LinearContrast((1.2, 1.6))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.MultiplyAndAddToBrightness(mul=(0.5, 0.7), add=(0, 0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.MultiplyAndAddToBrightness(mul=(0.5, 0.7), add=(0, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.MultiplyAndAddToBrightness(mul=(0.7, 1.0), add=(0, 0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.MultiplyAndAddToBrightness(mul=(0.7, 1.0), add=(0, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.MultiplyAndAddToBrightness(mul=(1.0, 1.25), add=(0, 0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.MultiplyAndAddToBrightness(mul=(1.0, 1.25), add=(0, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.MultiplyHue((0.5, 0.7))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.MultiplyHue((0.5, 0.7))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.MultiplyHue((0.7, 1.0))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.MultiplyHue((0.7, 1.0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Fliplr(1), imgaug.augmenters.MultiplyHue((1.0, 1.25))
    ]), 
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.Flipud(1), imgaug.augmenters.MultiplyHue((1.0, 1.25))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ShearX((-10, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ShearX((0, 10))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ShearY((-10, 0))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.ShearY((0, 10))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(rotate=(-90, -45), fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine( rotate=(-45, 0), fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(rotate=(0, 45), fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(rotate=(45, 90), fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(translate_px={"x": 0, "y": [-10, 0]}, fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(translate_px={"x": 0, "y": [0, 10]}, fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(translate_px={"x": [-10, 0], "y": 0}, fillcolor=(0, 256))
    ]),
    imgaug.augmenters.meta.Sequential([
    imgaug.augmenters.pillike.Affine(translate_px={"x": [0, 10], "y": 0}, fillcolor=(0, 256))
    ]),
]


images = os.listdir("malignant")
os.makedirs("augmented", exist_ok=True)
for seq_iteration in range(len(seq)):
    print(f"Augmenting {seq_iteration+1}/{len(seq)}")
    for image_iteration in tqdm(range(len(images))):
        image = cv2.imread(f"malignant/{images[image_iteration]}")
        image_template = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
        image_template[0, :, :, :] = image
        images_aug = seq[seq_iteration](images=image_template)
        images_aug = images_aug[0, :, :, :]
        img_name_augmented = images[image_iteration].replace(".jpg", f"_{seq_iteration}.jpg")
        cv2.imwrite(f"augmented/{img_name_augmented}", images_aug)


# scale_percent = 10 # percent of original size
# width = int(images_aug.shape[1] * scale_percent / 100)
# height = int(images_aug.shape[0] * scale_percent / 100)
# dim = (width, height)

# # resize image
# resized = cv2.resize(images_aug, dim, interpolation = cv2.INTER_AREA)
# resized_original = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow("original", resized_original)
# cv2.imshow("augmented", resized)
# cv2.waitKey(0)