import os
import json

def download_and_extract_dataset():
    # .kaggle 
    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        
    # Kaggle Credentials
    kaggle_json_content = {
        "username": "balavenkatesh96",
        "key": "bd62f12a0e9bfd678166f640d46df10d"
    }
    
    json_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(json_path, 'w') as f:
        json.dump(kaggle_json_content, f)

    try:
        os.chmod(json_path, 0o600)
    except Exception:
        pass

    os.environ['KAGGLE_USERNAME'] = "balavenkatesh96"
    os.environ['KAGGLE_KEY'] = "bd62f12a0e9bfd678166f640d46df10d"

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    #  Dataset Slug
    dataset_name = "akeshkumarhp/electronics-products-amazon-10k-items"
    data_dir = "data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(" Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
    print("Dataset Downloaded & Extracted Successfully!")

if __name__ == "__main__":
    download_and_extract_dataset()