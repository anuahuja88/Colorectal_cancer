import os
import random
import shutil

def split_data(input_dir, output_dir, class_dir, train_ratio=0.8):
    # Create train, validation, and test directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    train_dir = os.path.join(train_dir, class_dir)
    test_dir = os.path.join(test_dir, class_dir)
    for directory in [train_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Get list of image files in input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate the number of images for each split
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    test_count = total_images - train_count
    
    # Copy images to train directory
    for image_file in image_files[:train_count]:
        src = os.path.join(input_dir, image_file)
        dst = os.path.join(train_dir, image_file)
        shutil.copy(src, dst)
    
    # Copy images to test directory
    for image_file in image_files[train_count:]:
        src = os.path.join(input_dir, image_file)
        dst = os.path.join(test_dir, image_file)
        shutil.copy(src, dst)

# Example usage
input_true_directory = './Data/TRUE_Group_for_Microsatellite_Instability_dMMR_MSI'
input_false_directory = './Data/FALSE_Group_for_Microsatellite_instability_pMMR_MSS'
output_directory = './Data'
print("Splitting TRUE cases")
split_data(input_true_directory, output_directory, "class1")
print("Splitting FALSE cases")
split_data(input_false_directory, output_directory, "class2")
