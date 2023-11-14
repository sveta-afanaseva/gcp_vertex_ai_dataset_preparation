import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import glob

df = pd.read_csv("train.csv")
array = np.array(df)

for i in range(10):
    try:
        os.mkdir(str(i))
    except Exception:
        pass

for row in array:
    target = row[0]
    row = row[1:]
    Image.fromarray(np.reshape(row, (28, 28)).astype(np.uint8)).save(f'{target}/{time.monotonic()}.png')

result = []
for i in range(10):
    for file in glob.glob(f"{i}/*"):
        file = file.replace('\\', '/')
        result.append([f"gs://mnist_dataset/images/{file}.png", i])


pd.DataFrame(result).to_csv("my_train.csv", index=False, header=False)