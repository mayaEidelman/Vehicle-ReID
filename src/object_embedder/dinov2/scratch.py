import torch
from dataset.data_handling import VeRiDataset
from torch.utils.data import DataLoader
from utils import Triplet_Dataset, dataset_example

train_data = Triplet_Dataset('../../../data/VeRi/VeRi-UAV/image_train/')
dataset_list = train_data.dataset_example()
dataset_example(dataset_list)
