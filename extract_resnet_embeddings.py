import pandas as pd
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision.transforms as transforms
import torchvision.models as models

def extract_vision_features_fast():
    train_x_path = "data/splits/X_train.csv"
    test_x_path = "data/splits/X_test.csv"

    if not os.path.exists(train_x_path):
        print("Train split files not found!")
        return

    print("Loading Data Splits for ResNet Vision Embeddings...")
    X_train = pd.read_csv(train_x_path)
    X_test = pd.read_csv(test_x_path)

    print("Loading Pretrained ResNet50 Model...")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    headers = {'User-Agent': 'Mozilla/5.0'}

    def process_single_url(url):
        try:
            if pd.isna(url) or not str(url).startswith('http'):
                return np.zeros(2048)

            response = requests.get(str(url), headers=headers, timeout=3)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            tensor_img = transform(img).unsqueeze(0)

            with torch.no_grad():
                feat = resnet(tensor_img).squeeze().numpy()
                return feat
        except Exception:
            return np.zeros(2048)

    def fetch_in_batches(url_series, max_workers=16):
        print(f"Downloading images in parallel using {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_url, url_series))
        return np.vstack(results)

    print("Extracting ResNet Image Embeddings for X_train...")
    train_resnet = fetch_in_batches(X_train['image'])

    print("Extracting ResNet Image Embeddings for X_test...")
    test_resnet = fetch_in_batches(X_test['image'])

    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/train_resnet.npy", train_resnet)
    np.save("data/embeddings/test_resnet.npy", test_resnet)

    print("=================== RESNET EXTRACTION SUMMARY ===================")
    print(f"X_train ResNet Vector Shape : {train_resnet.shape}")
    print(f"X_test ResNet Vector Shape  : {test_resnet.shape}")
    print("\n ResNet Vision Embeddings successfully saved in 'data/embeddings/' directory!")

if __name__ == "__main__":
    extract_vision_features_fast()