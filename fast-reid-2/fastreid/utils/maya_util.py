from PIL import Image
import matplotlib.pyplot as plt
import torch
# import open_clip
import os
import random
import glob
import re
import torchvision.transforms as T

from fastreid.data.transforms import *


def veri_process_dir_old(dataset_name, dir_path, is_train=True):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        
        pattern = re.compile(r'([\d]+)_(\d\d)')
        
        data = []
        for img_path in img_paths:  
            pid, camid = map(int, pattern.search(img_path).groups()) # pid is the vehicle id (as in the dataset description) camid is the vehicle direction         
            if pid == -1: continue  
            assert 0 <= camid <= 7
            camid -= 1
            if is_train:
                pid = dataset_name + "_" + str(pid)
                camid = dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid)) # the camid appended is camid -1
        return data


def veri_prepare_example_test_data(dataset_name, dir_path):
    dataset = veri_process_dir_old(dataset_name, dir_path, is_train=True)
    test = []
    test_paths = []
    pattern = re.compile(r'([\d]+)_(\d\d)')
    while len(test) < 15:
        random_idx = random.randint(0, len(dataset))
        image_path = dataset[random_idx][0]
        pid, camid = map(int, pattern.search(image_path).groups())
        test.append((image_path, pid))
        test_paths.append(image_path)
        while camid > 0 and len(test) < 15:
            for data in dataset:
                if (data[1] == dataset_name + "_" + str(pid) and data[0] not in test_paths):
                    test_paths.append(data[0])
                    test.append((data[0], pid))
                    break
            camid -= 1
    return test
###########################################################################################
cos = torch.nn.CosineSimilarity(dim=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# transform = T.Compose([
# T.ToTensor(),
# # T.RandomHorizontalFlip(p=0.5),
# # # T.Resize(224, interpolation=T.functional.InterpolationMode.BILINEAR),
# # T.CenterCrop(224),
# # T.Normalize(mean=[0.5], std=[0.5]),
# ])
def test_transform():
    res = []
    res.append(T.Resize(224, interpolation=3))
    res.append(T.CenterCrop(224))
    res.append(ToTensor())
    return T.Compose(res)

def imageSimilarity(model, image1, image2, fastreid):
    transform = test_transform()
    input1 = transform(image1)[:3].unsqueeze(0)
    input2 = transform(image2)[:3].unsqueeze(0)
    input1 = input1.to(device)
    input2 = input2.to(device)
    
    with torch.no_grad():
       outputs1 = model(input1)
       outputs2 = model(input2)
    if fastreid:
        outputs1 = torch.mean(outputs1, dim=(2, 3))
        outputs2 = torch.mean(outputs2, dim=(2, 3))
       
    last_hidden_states1 = outputs1[0]
    last_hidden_states2 = outputs2[0]
    result = cos(last_hidden_states1, last_hidden_states2)

    print("similarity: ", result)
    return result
###########################################################################################
def similarityLoop_veri(model, name, queryImageTuple, datasetList, threshold, fastreid):
    queryImagePath, label = queryImageTuple
    queryImage = Image.open(queryImagePath)
    fig, axes = plt.subplots(1, len(datasetList) + 1, figsize=(15, 5)) 
    createFig(axes[0], queryImage, "Query image")
    for idx, path in enumerate(datasetList):
        image_path = path[0]
        print("Processing image:", image_path)
        testImage = Image.open(image_path)
        score = imageSimilarity(model, queryImage, testImage, fastreid)
        color = choose_color(threshold, score)
        createFig(axes[idx+1], testImage, f'{score:.2f}', color)
    plt.tight_layout()
    plt.savefig(f'vehicle_{name}_comparison.png')
    
def choose_color(threshold, score):
    if score <= threshold:
        return 'red'
    return 'green'

    
def createFig(ax, image, title, color='black'):
    ax.imshow(image.resize((200, 200)))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    for spine in ax.spines.values():
            spine.set_color(color)
            
################################################################################################################

def experiment_loop(fastreid_model):
    data = veri_prepare_example_test_data('VeRi-UAV','../../data/VeRi/VeRi-UAV/image_train/')
    # resnet_model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2") 
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc').to(device)
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    trained_resnet = torch.load("../../src/object_embedder/dinov2/trained_resnet50.pth" )
    # similarityLoop_veri(resnet_model, "resnet", data[1], data, 0.80, false)
    similarityLoop_veri(trained_resnet, "trained-resnet", data[1], data, 0.79, False)
    similarityLoop_veri(dino_model, "dinov2", data[1], data, 0.85, False)
    similarityLoop_veri(fastreid_model, "fastreid", data[1], data, 0.80, True)