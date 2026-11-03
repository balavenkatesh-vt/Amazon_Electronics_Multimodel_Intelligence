import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

def extract_bert_features():
    train_x_path = "data/splits/X_train.csv"
    test_x_path = "data/splits/X_test.csv"
    
    if not os.path.exists(train_x_path):
        print("Train split files not found! Complete previous steps first.")
        return

    print(" Loading Data Splits for BERT Embeddings...")
    X_train = pd.read_csv(train_x_path)
    X_test = pd.read_csv(test_x_path)

    # 1. Load Pretrained BERT Model & Tokenizer
    print(" Loading Pretrained Sentence-Transformers BERT Model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def get_embeddings(text_list, batch_size=64):
        embeddings = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size].tolist()
            # Clean missing text values
            batch_texts = [str(t) if pd.notna(t) else "" for t in batch_texts]
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                # Mean Pooling to get single vector per sentence
                token_embeddings = outputs[0]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                embeddings.append(pooled.numpy())
        return np.vstack(embeddings)

    print(" Extracting BERT Embeddings for X_train...")
    train_bert = get_embeddings(X_train['name'])
    
    print(" Extracting BERT Embeddings for X_test...")
    test_bert = get_embeddings(X_test['name'])

    # Save BERT Embeddings
    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/train_bert.npy", train_bert)
    np.save("data/embeddings/test_bert.npy", test_bert)

    print("=================== BERT EXTRACTION SUMMARY ===================")
    print(f"X_train BERT Vector Shape : {train_bert.shape}")
    print(f"X_test BERT Vector Shape  : {test_bert.shape}")
    print("\n BERT Embeddings successfully saved in 'data/embeddings/' directory!")

if __name__ == "__main__":
    extract_bert_features()