# from similarity import imageSimilarity 
from similarity.resnet import imageSimilarity, imageSimilarity_clip
from PIL import Image
import matplotlib.pyplot as plt
from dataset.data_preparation import veri_prepare_example_test_data
import torch
import open_clip

def similarityLoop(queryImagePath, datasetPath, threshold):
    queryImage = Image.open(queryImagePath)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 16, 1)
    plt.imshow(queryImage.resize((200, 200)))
    plt.axis('off')
    plt.title('Query Image')
    with open(datasetPath, 'r') as f:
        
        for idx, path in enumerate(f):
            image_path = path.strip()
            print("Processing image:", image_path)
            testImage = Image.open(image_path)
            score = imageSimilarity(queryImage, testImage)
            color = choose_color(threshold, score)
            plt.subplot(1, 16, idx + 2)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            ax = plt.gca() 
            for spine in ax.spines.values():
               spine.set_color(color)
            testImage = testImage.resize((200, 200))
            plt.imshow(testImage)
            plt.title(f'{score:.2f}')
        plt.tight_layout()
        plt.savefig('image_comparison.png')

def similarityLoop_veri(model, name, queryImageTuple, datasetList, threshold):
    queryImagePath, label = queryImageTuple
    queryImage = Image.open(queryImagePath)
    fig, axes = plt.subplots(1, len(datasetList) + 1, figsize=(15, 5)) 
    createFig(axes[0], queryImage, "Query image")
    for idx, path in enumerate(datasetList):
        image_path = path[0]
        print("Processing image:", image_path)
        testImage = Image.open(image_path)
        score = imageSimilarity(model, queryImage, testImage)
        color = choose_color(threshold, score)
        createFig(axes[idx+1], testImage, f'{score:.2f}', color)
    plt.tight_layout()
    plt.savefig(f'vehicle_{name}_comparison.png')
    
def similarityLoop_veri_clip(model, name, queryImageTuple, datasetList, threshold):
    queryImagePath, label = queryImageTuple
    queryImage = Image.open(queryImagePath)
    fig, axes = plt.subplots(1, len(datasetList) + 1, figsize=(15, 5)) 
    createFig(axes[0], queryImage, "Query image")
    for idx, path in enumerate(datasetList):
        image_path = path[0]
        print("Processing image:", image_path)
        testImage = Image.open(image_path)
        score = imageSimilarity_clip(model, queryImage, testImage)
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
    

def experiment_loop():
    data = veri_prepare_example_test_data('VeRi-UAV','../../../data/VeRi/VeRi-UAV/image_train/')
    resnet_model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2") 
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    trained_resnet = torch.load("trained_resnet50.pth" , map_location=torch.device('cpu'))
    similarityLoop_veri(resnet_model, "resnet", data[1], data, 0.80)
    similarityLoop_veri(trained_resnet, "trained-resnet", data[1], data, 0.79)
    similarityLoop_veri(dino_model, "dinov2", data[1], data, 0.85)
    similarityLoop_veri_clip(clip_model, "clip", data[1], data, 0.80)

experiment_loop()