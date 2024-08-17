import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import numpy as np
import configparser
from sqlalchemy import create_engine
from tqdm import tqdm
import os

# 설정 파일 로드
def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

# 데이터 로드
def load_data_from_db(db_config):
    engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['db_name']}")
    query = "SELECT law_id, content FROM articles"
    df = pd.read_sql(query, engine)
    df = df.reset_index(drop=True)
    return df

class LawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = int(self.labels[item])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 디바이스 설정
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# 학습 함수
def train_epoch(model, data_loader, optimizer, device, epoch, num_epochs):
    model.train()
    losses = []
    total_batches = len(data_loader)

    progress_bar = tqdm(enumerate(data_loader), total=total_batches,
                        desc=f"Epoch {epoch}/{num_epochs}",
                        bar_format='{l_bar}{bar:30}{r_bar}')


    for batch_idx, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 진행률 표시
        progress = (batch_idx + 1) / total_batches
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}",
                                  "Progress": f"{'#' * int(progress * 30):{30}}"})

    return np.mean(losses)

#평가함수
def evaluate_model(model, data_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", bar_format='{l_bar}{bar:30}{r_bar}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(preds.cpu().tolist())

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    return accuracy, f1, precision, recall

# 메인 함수
def main():
    config = get_config()
    db_config = config['database']

    # 데이터 로드
    df = load_data_from_db(db_config)

    # 레이블 인코딩
    df['label'] = pd.Categorical(df['law_id']).codes

    # 데이터 분할
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 토크나이저 및 모델 초기화
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    num_classes = len(df['label'].unique())

    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=num_classes).to(device)

    # 데이터셋 및 데이터로더 생성
    max_len = 256
    """1차 512, 2,3차 256"""
    train_dataset = LawDataset(train_df['content'].values, train_df['label'].values, tokenizer, max_len)
    test_dataset = LawDataset(test_df['content'].values, test_df['label'].values, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 옵티마이저 설정 """추가학습으로 rl 축소/ 1차: 2e-5"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)


    ## 기존 모델 로드(추가학습)
    if os.path.exists('bert_law_classifier.pth'):
        model.load_state_dict(torch.load('bert_law_classifier.pth'))
        print("기존 모델을 로드했습니다.")
    else:
        print("새로운 모델로 시작합니다.")

    # 학습 루프
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1, num_epochs)
        accuracy, f1, precision, recall = evaluate_model(model, test_loader, device)

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test F1 Score: {f1:.4f}')
        print(f'Test Precision: {precision:.4f}')
        print(f'Test Recall: {recall:.4f}')

    # 모델 저장
    torch.save(model.state_dict(), 'bert_law_classifier.pth')
    print("모델이 저장되었습니다.")


if __name__ == "__main__":
    main()