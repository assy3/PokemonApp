from PIL import Image
import os, glob
import numpy as np
import random, math

root_dir = "./pictures"
types =["normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "dragon"]

X = []
Y = []

allfiles = []
for idx, cat in enumerate(types):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

for cat, fname in allfiles:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((240, 240))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

x = np.array(X)
y = np.array(Y)

np.save("test_data_X.npy", x)
np.save("test_data_Y.npy", y)