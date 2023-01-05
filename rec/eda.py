import numpy as np
import pandas as pd

path = "../input/ptb.train.txt"

with open(path, 'r') as f:
    for i in range(20):
        print(f.readline())
