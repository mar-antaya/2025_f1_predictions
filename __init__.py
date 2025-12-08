import os

# Specify the directory path you want to create
cache_dir_name = "f1_cache"

# Create the directory if it doesn't already exist
if not os.path.exists(cache_dir_name):
    os.makedirs(cache_dir_name)
    print(f"Directory '{cache_dir_name}' created!")