import transformers
import sentencepiece
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


from datasets import Dataset

# SentencePiece 설치 확인
try:
    import sentencepiece
except ImportError:
    raise ImportError("You need to have SentencePiece installed to use MBart50TokenizerFast.")

from torch.utils.data import DataLoader, Dataset
import torch

class CustomTranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'source': self.data[idx]['en'], 'target': self.data[idx]['ko']}

# 임의의 데이터 생성
sample_data = [
    {'en': 'Hello', 'ko': '안녕'},
    {'en': 'I eat food', 'ko': '나 밥 먹다'},
    {'en': 'I love programming', 'ko': '나 프로그래밍 사랑해'},
    {'en': 'A group of people are competing for a book', 'ko': '사람 싸우다 책'}
]

custom_dataset = CustomTranslationDataset(sample_data)

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

def preprocess_data(example):
    input_text = example['source']
    target_text = example['target']
    return tokenizer(input_text, return_tensors='pt', padding=True, truncation=True), tokenizer(target_text, return_tensors='pt', padding=True, truncation=True)

train_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)
num_train_epochs = 50
learning_rate = 5e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for epoch in range(num_train_epochs):
    for batch in train_dataloader:
        inputs = tokenizer(batch['source'], return_tensors='pt', padding=True, truncation=True)
        labels = tokenizer(batch['target'], return_tensors='pt', padding=True, truncation=True)['input_ids']

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

article_en = "A group of people are competing for a book"
encoded_en = tokenizer(article_en, return_tensors="pt")
generated_tokens_en = model.generate(
    **encoded_en,
    forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]

)
translated_text_en = tokenizer.batch_decode(generated_tokens_en, skip_special_tokens=True)[0]
print("영어에서 한국어 번역:", translated_text_en)