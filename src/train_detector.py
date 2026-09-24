# Import need libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATASET_PATH = "/data_pr/TB_Data/"

df = pd.read_csv(os.path.join(DATASET_PATH, "metadata.csv"))
df = df.dropna(subset=['label'])

print(df.head())


