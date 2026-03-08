
import kagglehub
import shutil
import os

# download dataset
dataset_path = kagglehub.dataset_download(
    "caesarmario/oecd-data-crude-oil-production"
)

print("Downloaded to:", dataset_path)

# destination folder
dest = "../SFHA-Project/data/raw"

os.makedirs(dest, exist_ok=True)

# copy files
for file in os.listdir(dataset_path):
    shutil.copy(os.path.join(dataset_path, file), dest)

print("Dataset copied to:", dest)
