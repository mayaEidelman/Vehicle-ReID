import os
import random
import glob
import re

__all__ = ['veri_prepare_example_test_data', 'veri_process_dir', 'create_triplet']

# takes a folder of images and creates a file with their paths
def create_txt_deteset(folder_path, output_file):
    with open(output_file,'w') as f:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image file (you can customize this condition)
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                # Write the absolute path of the image file to the output file
                f.write(os.path.join(folder_path, filename) + '\n')
                
create_txt_deteset('../../../data/VeRi/VeRi-UAV/image_train/', 'image_train.txt')


def vru_process_dir(list_file, dataset_name, image_dir, is_train=True):
        img_list_lines = open(list_file, 'r').readlines() 

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split(' ')[1])
            imgid = line.split(' ')[0]
            img_path = os.path.join(image_dir, f"{imgid}.jpg")
            imgid = int(imgid)
            if is_train:
                vid = f"{dataset_name}_{vid}"
                imgid = f"{dataset_name}_{imgid}"
            dataset.append((img_path, vid, imgid))

        if is_train: return dataset
        else:
            random.shuffle(dataset)
            vid_container = set()
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    gallery.append(sample)
                else:
                    query.append(sample)

            return query, gallery
        
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
    
def veri_process_dir(dir_path):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_(\d\d)')
        data = []
        for img_path in img_paths:  
            pid, camid = map(int, pattern.search(img_path).groups()) # pid is the vehicle id (as in the dataset description) camid is the vehicle direction         
            if pid == -1: continue  
            assert 0 <= camid <= 7
            data.append((img_path, pid, camid)) # the camid appended is camid -1
        return data
    
def create_triplet(dir_path):
    dataset = veri_process_dir(dir_path)
    triplet_dataset = []
    for (positive_path, pid, camid) in dataset:
        anchor_path = find_anchor_image(dataset, positive_path, pid)
        negative_path = find_negative_image(dataset, positive_path, pid)
        triplet_dataset.append((positive_path, anchor_path, negative_path))
    return triplet_dataset
        
def find_negative_image(dataset, image, pid_img):
    while(True):
        rand_image, pid, camid = dataset[random.randint(0, len(dataset)-1)]
        if pid != pid_img and rand_image != image:
            return rand_image
        
def find_anchor_image(dataset, image_path, pid_po):
    for (tmp_pth, pid, camid) in dataset:
        if pid == pid_po and tmp_pth != image_path:
            return tmp_pth
    
# Creates random test dataset sample with images from of the same vehicle to create nice baseline ;)
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
    
        
    
# data = veri_process_dir('VeRi-UAV','../../../data/VeRi/VeRi-UAV/image_train/')
# data = veri_prepare_example_test_data('VeRi-UAV','../../../data/VeRi/VeRi-UAV/image_train/', is_train=True)
# print(data)
