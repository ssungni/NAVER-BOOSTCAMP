import os

import pandas as pd


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
    # BASE_DIR와 OUTPUT_DIR 설정
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "data")

    # 파일 경로 지정
    backtranslated_file = os.path.join(
        "backtranslation_data", "backtranslated_DeepL_JP.csv"
    )
    total_cleaned_file = os.path.join("split_train_data", "total_cleaned.csv")

    # 두 CSV 파일을 읽어 DataFrame으로 변환하고 post-processing 적용
    df_backtranslated = post_process(pd.read_csv(backtranslated_file))
    df_total_cleaned = post_process(pd.read_csv(total_cleaned_file))

    # 두 DataFrame을 하나로 병합합니다.
    merged_df = pd.concat([df_backtranslated, df_total_cleaned], ignore_index=True)

    # 중복 제거
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first")

    # ID 열을 기준으로 정렬합니다.
    merged_df = merged_df.sort_values("ID")

    # 병합된 DataFrame을 새로운 CSV 파일로 저장합니다.
    output_file = os.path.join(output_dir, "train.csv")

    # 파일이 이미 존재하면 덮어씁니다.
    merged_df.to_csv(output_file, index=False, mode="w")

    print(f"파일 병합 완료. '{output_file}'로 저장되었습니다.")
    print(f"총 행 수: {len(merged_df)}")


if __name__ == "__main__":
    main()