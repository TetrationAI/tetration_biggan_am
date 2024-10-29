import os
import shutil

def copy_directory_contents(source_dir, target_dir):
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)
        
        if os.path.isdir(source_item):
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)  # Copy directory
        else:
            shutil.copy2(source_item, target_item)  # Copy files

def combine_directories(source_dir1, source_dir2, target_dir):
    os.makedirs(target_dir, exist_ok=True)  # Create target directory if it does not exist
    copy_directory_contents(source_dir1, target_dir)
    copy_directory_contents(source_dir2, target_dir)

# Example usage
source_directory1 = '/path/to/source1'
source_directory2 = '/path/to/source2'
target_directory = '/path/to/target'

combine_directories(source_directory1, source_directory2, target_directory)