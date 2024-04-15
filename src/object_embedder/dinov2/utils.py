import os
import random
import glob
import re
import torch
import matplotlib.pyplot as plt
from PIL import Image


__all__ = ['Metric', 'Triplet_Dataset', 'create_txt_dataset', 'accuracy', 'DataMiner', 'dataset_example']

class Metric:
  def __init__(self):
    self.lst = 0.
    self.sum = 0.
    self.cnt = 0
    self.avg = 0.

  def update(self, val, cnt=1):
    self.lst = val
    self.sum += val * cnt
    self.cnt += cnt
    self.avg = self.sum / self.cnt

##################################################################################################################

class Triplet_Dataset:
    def __init__(self, dir_path):
        self.img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        self.pattern = re.compile(r'([\d]+)_(\d\d)')
        self.dataset = self.create_dataset()
        self.triplet_dataset = self.create_triplet()
        
    def create_dataset(self):
        data = {}
        for img_path in self.img_paths:  
            pid, camid = map(int, self.pattern.search(img_path).groups()) # pid is the vehicle id (as in the dataset description) camid is the vehicle direction         
            if pid == -1: continue  
            assert 0 <= camid <= 7
            if pid in data.keys():
                data[pid].append(img_path) 
            else:
                data[pid] = [img_path]
        return data
    
    def create_triplet(self):
        triplet_dataset = []
        for pid, positive_paths in self.dataset.items():
            for positive_path in positive_paths:
                anchor_path = self.find_anchor_image(positive_path, pid)
                if anchor_path is None:
                    anchor_path = positive_path
                negative_path = self.find_negative_image(positive_path, pid)
                triplet_dataset.append((positive_path, anchor_path, negative_path))
        return triplet_dataset
        
    def find_negative_image(self, image, pid_img):
        while(True):
            pid, rand_images = random.choice(list(self.dataset.items()))
            rand_image = random.choice(rand_images)
            if pid != pid_img and rand_image != image:
                return rand_image
        
    def find_anchor_image(self, image, pid_img):
        paths_with_same_label = self.dataset[pid_img]
        rand_image = random.choice(paths_with_same_label)
        for i in range(len(paths_with_same_label)):
            if rand_image != image and rand_image is not None:
                return rand_image
            
    def dataset_example(self):
        result = []
        while len(result) < 20:
            random_idx = random.randint(0, len(self.dataset)-1)
            pid = list(self.dataset.keys())[random_idx]
            paths = self.dataset.get(pid)
            if len(paths) >= 4:
                for i in range(4):
                    sample = []
                    for i in range(4):
                        x = paths[random.randint(0, 3)]
                        if x not in sample:
                            sample.append(x)
            else:
                continue
            result.extend(sample)
        return result
        
                
            
            
    def create_all_possible_triplets(self):
        dataset = []
        
        return dataset
    
    # def create_example_test_data(self):
    #     test = []
    #     while len(test) < 15:
    #         random_idx = random.randint(0, len(self.dataset))
    #         pid = list(self.dataset.keys())[random_idx]
    #         image_path = self.dataset.get(pid)[0]
    #         test.append((image_path, pid))
    #         test_paths.append(image_path)
    #         while camid > 0 and len(test) < 15:
    #             for data in dataset:
    #                 if (data[1] == dataset_name + "_" + str(pid) and data[0] not in test_paths):
    #                     test_paths.append(data[0])
    #                     test.append((data[0], pid))
    #                     break
    #             camid -= 1
    #     return test
        
##################################################################################################################
# Takes a folder of images and creates a file with their paths
def create_txt_dataset(folder_path, output_file):
    with open(output_file,'w') as f:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image file (you can customize this condition)
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                # Write the absolute path of the image file to the output file
                f.write(os.path.join(folder_path, filename) + '\n')
                
def createFig(ax, image):
    ax.imshow(image.resize((200, 200)))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            
def dataset_example(datasetList):
    fig, axes = plt.subplots(1, len(datasetList) + 1, figsize=(15, 5)) 
    for idx, image_path in enumerate(datasetList):
        testImage = Image.open(image_path)
        createFig(axes[idx+1], testImage)
    plt.tight_layout()
    plt.savefig(f'example_dataset.png')

                
#################################################################################################################
#### ACCURACY
#################################################################################################################
def accuracy(batch_loss):
    return (batch_loss > 0).sum()*1.0/batch_loss.size()[0]

#################################################################################################################

class DataMiner:
    def __init__(self, batch_triplet_loss):
        self.batch_triplet_loss = batch_triplet_loss
    
    def get_batch_loss(self, model, positive, anchor, negative):

        with torch.no_grad():
                p = model(positive)
                a = model(anchor)
                n = model(negative)

        batch_loss = self.batch_triplet_loss(a, p, n)
        return batch_loss
    
    def get_semi_hard_triplets(self, positive, anchor, negative, batch_loss, device):
        indexes = torch.nonzero(batch_loss>0).squeeze(dim=1)
        pos = torch.Tensor([]).to(device=device)
        neg = torch.Tensor([]).to(device=device)
        anc = torch.Tensor([]).to(device=device)
        if indexes.numel() > 0:
            for idx in indexes:
                anc = torch.cat((anc, anchor[idx].unsqueeze(0)), dim=0)
                pos = torch.cat((pos, positive[idx].unsqueeze(0)), dim=0)
                neg = torch.cat((neg, negative[idx].unsqueeze(0)), dim=0)
        return pos, anc, neg
    
    def compose_hard_triplets(self, model, positive, anchor, negative, device):
        batch_loss = self.get_batch_loss(model, positive, anchor, negative)
        pos, anc, neg = self.get_semi_hard_triplets(positive, anchor, negative, batch_loss, device)
        return pos, anc, neg
        
    def compose_semi_hard_batch(self, positive, anchor, negative, loader,  batch_size):
        if len(positive) == 0:
            return ([], [], [])
        if len(positive) < batch_size:
            loader.dataset.set_extra_triplet([positive, anchor, negative])
            return ([], [], [])
        else:
            batch_positive = positive[:batch_size]
            batch_anchor = anchor[:batch_size]
            batch_negative = negative[:batch_size]
            
            extra_positive = positive[batch_size:]
            extra_anchor = anchor[batch_size:]
            extra_negative = negative[batch_size:]
            
            loader.dataset.set_extra_triplet([extra_positive, extra_anchor, extra_negative])
            return([batch_positive, batch_anchor, batch_negative])
###########################################################################################################

            
    
    
                
        