import os
import random

def delete_20_percent_files(source_folder):
    """
    Randomly delete 20% of the files in the given folder.

    Parameters:
    source_folder (str): The path to the source folder where files are located.
    """
    # List all files in the directory
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    total_files = len(files)
    num_to_delete = int(total_files * 0.085)  # Calculate 20% of total files
    
    # Randomly pick files to delete
    files_to_delete = random.sample(files, num_to_delete)

    # Delete the selected files
    for file in files_to_delete:
        file_path = os.path.join(source_folder, file)
        os.remove(file_path)  # Delete the file

    print(f"Deleted {len(files_to_delete)} out of {total_files} files (20%).")

# Example usage
source_folder = '/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_75/Annotations'  # Source folder path
delete_20_percent_files(source_folder)
