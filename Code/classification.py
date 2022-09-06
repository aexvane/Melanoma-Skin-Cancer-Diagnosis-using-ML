import pandas as pd
import numpy as np
import shutil
import os
from tqdm import tqdm


train = pd.read_csv("train.csv")

tgt = np.array(train["target"].to_numpy())
name = np.array(train["image_name"].to_numpy())

os.makedirs("benign", exist_ok=True)
os.makedirs("malignant", exist_ok=True)
for image in tqdm(range(len(tgt))):
    if tgt[image] == 0:
        shutil.copy(f"train/{name[image]}.jpg", f"benign/{name[image]}.jpg")
    if tgt[image] == 1:
        shutil.copy(f"train/{name[image]}.jpg", f"malignant/{name[image]}.jpg")