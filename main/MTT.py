import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from pytorch_i3d import InceptionI3d
from keytotext import pipeline
import transformers
import sentencepiece
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


from datasets import Dataset

from konlpy.tag import Okt


# SentencePiece 설치 확인
try:
    import sentencepiece
except ImportError:
    raise ImportError("You need to have SentencePiece installed to use MBart50TokenizerFast.")


# MBart 모델과 토크나이저 로드
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# WLASL 클래스 인덱스와 단어를 매핑하는 사전 생성
def create_WLASL_dictionary():
    global wlasl_dict 
    wlasl_dict = {}
    
    with open('/home/plass-doogle/WLASL-Recognition-and-Translation/WLASL/I3D/preprocess/wlasl_class_list.txt') as file:
        for line in file:
            split_list = line.split()
            key = int(split_list[0])
            value = ' '.join(split_list[1:])
            wlasl_dict[key] = value


def extract_nouns_verbs(sentence):
    okt = Okt()
    words = okt.pos(sentence)
    extracted_words = [word for word, tag in words if tag in ['Noun', 'Verb']]
    return extracted_words

def load_model(weights, num_classes):
    # Inception 3D 모델 로드
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights, map_location='cpu'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    return i3d

def process_video(video_path, model, nlp, batch_size=40):
    global wlasl_dict 

    vidcap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []
    
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        frames.append(frame)

        if len(frames) == batch_size:
            prediction = run_on_tensor(model, frames)
            if prediction != " ":
                predictions.append(prediction)
            frames.pop(0)

    vidcap.release()

    # 예측된 클래스 인덱스를 단어로 변환하고 중복 제거
    unique_words = set()
    predicted_words = []
    for prediction in predictions:
        word = wlasl_dict.get(prediction, "Unknown")
        if word not in unique_words:
            predicted_words.append(word)
            unique_words.add(word)

    print("Predicted words:", predicted_words)

    if len(predicted_words) == 1:
        return predicted_words[0]

    sentence = nlp(predicted_words)
    return sentence


def preprocess_frame(frame):
    # 프레임 크기 조정 및 정규화
    sc = 224 / frame.shape[0]
    sx = 224 / frame.shape[1]
    frame = cv2.resize(frame, dsize=(0, 0), fx=sx, fy=sc)
    frame = (frame / 255.) * 2 - 1
    return frame

def run_on_tensor(model, frames):
    # 모델을 사용하여 프레임에서 수화 예측
    ip_tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    # 차원 전환: (t, h, w, c) -> (1, c, t, h, w)
    ip_tensor = ip_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    ip_tensor = ip_tensor.cuda()
    per_frame_logits = model(ip_tensor)
    t = ip_tensor.shape[2]  # 시간 차원(프레임 수) 설정
    predictions = F.upsample(per_frame_logits, t, mode='linear')
    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    arr = predictions.cpu().detach().numpy()[0]

    # 예측값 처리
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) > 0.5:
        return out_labels[0][-1]  # 최대 확률의 라벨 인덱스 반환
    else:
        return " "



def translate_to_korean(sentence, model, tokenizer):

    encoded_en = tokenizer(sentence, return_tensors="pt")
    generated_tokens_en = model.generate(
        **encoded_en,
        forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]
    )
    translated_text_en = tokenizer.batch_decode(generated_tokens_en, skip_special_tokens=True)[0]
    return translated_text_en


if __name__ == '__main__':
    # WLASL 사전 생성
    create_WLASL_dictionary()
#/home/plass-doogle/WLASL-Recognition-and-Translation/WLASL/I3D/vid/tesst.mp4
    video_path = '/home/plass-doogle/WLASL-Recognition-and-Translation/WLASL/I3D/vid/video (17).webm'
    weights = '/home/plass-doogle/WLASL-Recognition-and-Translation/WLASL/I3D/archived/asl300/FINAL_nslt_300_iters=2997_top1=56.14_top5=79.94_top10=86.98.pt'
    num_classes = 300  # 클래스 수

    i3d_model = load_model(weights, num_classes)
    nlp = pipeline("k2t-new")

    # 비디오 처리 및 문장 생성
    english_sentence = process_video(video_path, i3d_model, nlp)
    print("Generated English sentence: ", english_sentence)

    # 영어 문장을 한국어로 번역
    korean_translation = translate_to_korean(english_sentence, model, tokenizer)
    print("Translated Korean sentence: ", korean_translation)

    # 번역된 문장에서 명사와 동사 추출
    extracted_words = extract_nouns_verbs(korean_translation)
    print("Extracted words: ", ', '.join(extracted_words))
    