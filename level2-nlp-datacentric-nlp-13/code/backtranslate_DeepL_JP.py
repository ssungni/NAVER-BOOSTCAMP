import os
import time

import pandas as pd
import requests

# 경로 설정 추가
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "split_train_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "backtranslation_data")

# 입력 및 출력 파일 경로 수정
file_path = os.path.join(INPUT_DIR, "total_cleaned.csv")
output_path = os.path.join(OUTPUT_DIR, "backtranslated_DeepL_JP.csv")

DEEPL_API_KEY = " "
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

repeat = 1  # 각 데이터에 대해 역번역을 수행하는 횟수
SAVE_INTERVAL = 10  # 중간 저장할 데이터 개수
REQUEST_DELAY = 1  # API 요청 사이에 지연 추가 (단위: 초)


def back_translate(text):
    # 한국어 -> 일본어 번역
    params_ko_ja = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "source_lang": "KO",
        "target_lang": "JA",
    }
    response = requests.post(DEEPL_API_URL, data=params_ko_ja)
    response.raise_for_status()
    result = response.json()
    japanese_text = result["translations"][0]["text"]

    # 일본어 -> 한국어 번역
    params_ja_ko = {
        "auth_key": DEEPL_API_KEY,
        "text": japanese_text,
        "source_lang": "JA",
        "target_lang": "KO",
    }
    response = requests.post(DEEPL_API_URL, data=params_ja_ko)
    response.raise_for_status()
    result = response.json()
    korean_text = result["translations"][0]["text"]

    return korean_text


def post_process(df):
    replacements = {
        '"': "",
        ",": " ",
        "...": "…",
        "……": "…",
        "-": "·",
        " · ": "·",
        "· ": "·",
        " ·": "·",
        "  ": " ",
        "*": " ",
    }
    for old, new in replacements.items():
        df["text"] = df["text"].str.replace(old, new, regex=False)
    df["text"] = df["text"].str.strip()
    return df


def main():
    # 데이터 불러오기 경로 수정
    data = pd.read_csv(file_path)
    data = data[["ID", "text", "target"]]
    total_characters = data["text"].str.len().sum()

    print(f"원본 데이터 개수: {len(data)}")
    print(f"총 글자 수: {total_characters}")

    texts_to_translate = data["text"].tolist()

    print("back-translation을 시작합니다...")

    # 출력 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 기존 출력 파일이 있으면 덮어쓰기
    if os.path.exists(output_path):
        os.remove(output_path)

    for r in range(repeat):
        print(f"{r + 1}번째 증강을 시작합니다...")
        batch = []  # 중간 저장을 위한 배치 리스트

        for idx, text in enumerate(texts_to_translate):
            try:
                translated = back_translate(text)
            except Exception as e:
                print(f"ID {data.at[idx, 'ID']} 번역 중 오류 발생: {e}")
                translated = ""  # 오류 발생 시 빈 문자열로 대체

            batch.append(
                {
                    "ID": data.at[idx, "ID"],
                    "text": translated,
                    "target": data.at[idx, "target"],
                }
            )
            print(f"번역 완료: {idx + 1}/{len(texts_to_translate)}")
            time.sleep(REQUEST_DELAY)  # API 요청 사이에 지연 추가

            # SAVE_INTERVAL마다 중간 저장
            if (idx + 1) % SAVE_INTERVAL == 0:
                save_batch(batch, idx + 1)
                batch = []  # 배치 초기화

        # 마지막에 남은 데이터 저장
        if batch:
            save_batch(batch, len(texts_to_translate))

    # 후처리 적용
    print("후처리를 시작합니다...")

    # 생성된 CSV 파일 불러오기
    translated_data = pd.read_csv(output_path)

    # 후처리 함수 적용
    translated_data = post_process(translated_data)

    # 후처리된 데이터 다시 저장
    translated_data.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"후처리가 '{output_path}'에 성공적으로 적용되었습니다.")


def save_batch(batch, idx):
    batch_df = pd.DataFrame(batch)
    if not os.path.exists(output_path):
        # 첫 저장 시 헤더 포함
        batch_df.to_csv(output_path, index=False, encoding="utf-8-sig", mode="w")
    else:
        # 이후 저장 시 헤더 제외
        batch_df.to_csv(
            output_path, index=False, encoding="utf-8-sig", mode="a", header=False
        )
    print(f"{idx}개 데이터가 '{output_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()