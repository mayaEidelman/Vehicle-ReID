import os
import shutil
from PIL import Image

def downsample_jpg(input_file, output_file):
    """
    Downsamples a JPEG file by a factor of 8, and if resulting image is larger than 150x150,
    downsamples it again by a factor of 2.
    """
    img = Image.open(input_file)
    width, height = img.size

    # Downsample by a factor of 8
    img = img.resize((width // 8, height // 8), Image.LANCZOS)

    # If resulting image is larger than 200x200, downsample again by a factor of 2
    if img.size[0] > 100 or img.size[1] > 100:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)

    img.save(output_file, quality=95)

def mirror_directory(source_dir, target_dir):
    """
    Mirrors a directory by copying files. JPEG files are downsampled by a factor of 8.
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate through files in source directory
    for root, dirs, files in os.walk(source_dir):
        print(f"process {root}")
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, os.path.relpath(source_file, source_dir))

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            # If file ends with .jpg, downsample; otherwise, copy
            if file.lower().endswith('.jpg'):
                downsample_jpg(source_file, target_file)
            else:
                shutil.copyfile(source_file, target_file)

# Example usage
source_directory = "/home/labs/training/class46/Aerial-IR-REID/external_gits/fast-reid/datasets/VeRi-UAV"
target_directory = "/home/labs/training/class46/Aerial-IR-REID/external_gits/fast-reid/datasets/VeRi-UAV-DownSampled-100"

mirror_directory(source_directory, target_directory)
