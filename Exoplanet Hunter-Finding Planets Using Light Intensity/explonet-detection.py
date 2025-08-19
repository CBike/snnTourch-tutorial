import snntorch as snn
from pyexpat import features
from snntorch import surrogate


#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# SMOTE
from imblearn.over_sampling import SMOTE
from collections import Counter


# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import classification_report
from sklearn.metrics import  roc_auc_score



class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        with open(csv_file, 'r') as f:
            self.data = pd.read_csv(f)

        self.labels = self.data.iloc[:,0].values
        self.features = self.data.iloc[:,1:].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]

        if self.transform:
            feature = self.transform(feature)


        sample = {'feature': feature, 'label': label}

        return sample




train_dataset = CustomDataset('')
test_dataset = CustomDataset('')




