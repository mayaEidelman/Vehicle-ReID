import torch
import torch.optim as optim
from train.train import train_loop
from dataset.data_handling import VeRiDataset
from torch.utils.data import DataLoader
from utils import Triplet_Dataset, DataMiner
from datetime import datetime
import open_clip
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
print("----------- load model -------------------------")
model= torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
model = model.to(device)
print("xxxxxxxxxxxxxxxxxxxxxx model loaded xxxxxxxxxxxxxxxxxxxxxxxxxx")

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
batch_triplet_loss = torch.nn.TripletMarginWithDistanceLoss(reduction='none')
data_miner = DataMiner(batch_triplet_loss) 

train_data = Triplet_Dataset('../../../data/VeRi/VeRi-UAV/image_train/')
training_data = VeRiDataset(train_data.triplet_dataset)
train_dataloader = DataLoader(training_data, batch_size=300, shuffle=True, num_workers=3)

test_data = Triplet_Dataset('../../../data/VeRi/VeRi-UAV/image_test/')
testing_data = VeRiDataset(train_data.triplet_dataset)
test_dataloader = DataLoader(training_data, batch_size=200, shuffle=True, num_workers=3)

current_time = datetime.now()
print("Started at:", current_time)

batch_size=64
epochs = 10
train_loop(model, 'resnet',  optimizer,device,train_dataloader, test_dataloader,data_miner, batch_size, epochs)