# Level 2: 주제 분류 프로젝트 (Data-Centric Topic Classification)

### 📝 Abstract
- 이 프로젝트는 네이버 부스트캠프 AI Tech 7기 NLP Level 2 기초 프로젝트 경진대회로, Dacon, Kaggle과 유사한 대회형 방식으로 진행되었다.
- Data-Centric Topic Classification는 주어진 뉴스 헤드라인으로부터 해당 뉴스의 주제를 `0`~`6`의 정수 레이블로 분류하는 것으로, 모델 구조의 변경 없이 Data-Centric 관점으로 해결함을 목표로 하였다.

<br>

## Project Leader Board 
- Public
<img width="700" alt="public_leader_board" src="https://github.com/user-attachments/assets/423cee6f-c285-4b18-888a-574d05ab2c15">

- Private
<img width="700" alt="private_leader_board" src="https://github.com/user-attachments/assets/03c1ab2d-250d-433c-bd9d-4f94f721fe74">

- [📈 Project Wrap-Up Report 살펴보기](https://github.com/user-attachments/files/17674538/DataCentric_13.Wrap-Up.pdf)

<br>

## 🧑🏻‍💻 Team Introduction & Members 

> Team Name : 스빈라킨스배 [ NLP 13조 ]

### 👨🏼‍💻 Members
권지수|김성은|김태원|이한서|정주현|
:-:|:-:|:-:|:-:|:-:
 <img src='https://github.com/user-attachments/assets/ab4b7189-ec53-41be-8569-f40619b596ce' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/49dc0e59-93ee-4e08-9126-4a3deca9d530' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/a15b0f0b-cd89-412b-9b3d-f59eb9787613' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/11b2ed88-bf94-4741-9df5-5eb2b9641a9b' height=125 width=100></img>|<img src='https://github.com/user-attachments/assets/3e2d2a7e-1c64-4cb7-97f6-a2865de0c594' height=125 width=100></img>
[Github](https://github.com/Kwon-Jisu)|[Github](https://github.com/ssungni)|[Github](https://github.com/chris40461)|[Github](https://github.com/beaver-zip)|[Github](https://github.com/peter520416)
<a href="mailto:wltn80609@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:sunny020111@ajou.ac.kr" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:chris40461@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:beaver.zip@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|<a href="mailto:peter520416@gmail.com" target="_blank"><img src="https://img.shields.io/badge/Gmail-EA4335?style&logo=Gmail&logoColor=white"/></a>|

### 🧑🏻‍🔧 Members' Role

| 이름 | 역할 |
| :---: | --- |
| **`권지수`** | **Text/Label Noise Split**, **Text Cleaning**, **Prompt-Based Generation** |
| **`김성은`** | **Text/Label Noise Split**, **Back-Translation**(Google Translate), **Prompt-Based Generation** |
| **`김태원`** | **EDA**, **Re-Labeling**, **Evol-Instruct LLM for Augmentation**  |
| **`이한서`** | **Text/Label Noise Split**, **Text Cleaning**, **Back-translation**(DeepL), **Prompt-Based Generation** |
| **`정주현`** | **Re-Labeling**, **ML Model Searching** |

<br>

## 🖥️ Project Introduction 


| **프로젝트 주제** |주어진 뉴스 헤드라인으로부터 해당 뉴스의 주제를 분류하되, 모델 구조의 변경 없이 Data-Centric 관점으로 해결|
| :---: | --- |
| **프로젝트 구현내용** | LLM을 활용해 데이터의 텍스트 노이즈와 라벨 노이즈를 분류한 뒤, LLM으로 텍스트를 정제하고 Cleanlab으로 라벨을 정제했다. 그 후 한국어-일본어로 역번역하여 데이터를 증강하고, 중복 데이터를 제거하여 최적의 학습 데이터를 생성한 뒤 분류를 진행했다.|
| **개발 환경** |**• `GPU` :** Tesla V100(32G RAM) 서버 4개 <br> **• `개발 Tool` :** Jupyter notebook, VS Code [서버 SSH연결]|
| **협업 환경** |**• `Github Repository` :** 데이터, 코드 버전 관리 및 작업 상황 공유 <br>**• `Notion` :** 역할 분담, 실험 가설 및 검증 결과 공유 <br>**• `SLACK, Zoom` :** 실시간 비대면 회의|

<br>

## 📁 Project Structure

### 🗂️ 디렉토리 구조 설명

**Code**
- Text/Label Noise Split: `filtering.ipynb`
- Text/Label Cleaning, Back-Translation, Baseline: `/code`

**Data**
- Noise/Cleaned: `/split_train_data`
- Back-Translation: `/backtranslation_data`
- Train: `/data`

### 📄 코드 설명

- `filtering.ipynb`: `train.csv`에서 Text/Label Noise를 구분해 `text_noise.csv`, `labe_noise.csv`를 생성
- `text_clean.py`: `text_noise.csv`의 text를 정제한 `text_cleaned.csv`를 생성
- `correct_label.py`: `label_noise.csv`의 label을 교정한 뒤, `text_cleaned.csv`와 합친 `merge_text_label_cleaned.csv`를 생성
- `total_clean.py`: `merge_text_label_clean.csv`의 label을 교정해 `total_cleaned.csv`를 생성
- `backtranslate_DeepL_JP.py`: `total_cleaned.csv`를 한-일-한 역번역해 `backtranslated_DeepL_JP.csv`를 생성
- `postprocess_and_merge.py`: `total_cleaned.csv`와 `backtranslated_DeepL_JP.csv`를 후처리하여 합친 `train.csv`를 생성
- `baseline_code.ipynb`: baseline code

```
📂 backtranslation_data
┃ ┗  backtranslated_DeepL_JP.csv
📂 code
┃ ┣  backtranslate_DeepL_JP.py
┃ ┣  baseline_code.ipynb
┃ ┣  correct_label.py
┃ ┣  postprocess_and_merge.py
┃ ┣  text_clean.py
┃ ┗  total_clean.py
📂 data
┃ ┗  train.csv
📂 split_train_data
┃ ┣  label_noise.csv
┃ ┣  merge_text_label_cleaned.csv
┃ ┣  text_cleaned.csv
┃ ┣  text_noise.csv
┃ ┗  total_cleaned.csv
┣ filtering.ipynb
┗ requirements.txt
 ```
<br>

## 📐 Project Ground Rule
> 팀 협업을 위해 프로젝트 관련 Ground Rule을 설정하여 프로젝트가 원활하게 돌아갈 수 있도록 규칙을 정했으며, 날짜 단위로 간략한 목표를 설정하여 협업을 원활하게 진행할 수 있도록 계획하여 진행했다.

- **`Server`**: 권지수, 김성은, 이한서, 정주현 캠퍼는 각자 서버를 생성해 모델 실험을 진행하고, 김태원 캠퍼는 서버가 유휴 상태인 서버에서 실험을 진행했다.
- **`Git`**: `exp` branch에 각자 폴더를 생성해 작업하고, 공통으로 사용할 파일은 main에 push 하는 방법으로 협업했다.
- **`Submission`**: 대회 마감 5일 전까지는 자유롭게 제출하고, 5일 전부터는 인당 2회씩 분배했다.
- **`Notion`**: 원활한 아이디어 브레인스토밍과 분업을 위해 회의를 할 경우 노션에 기록하며, 연구 및 실험결과의 기록을 공유했다.

<br>

## 🗓 Project Procedure: 총 2주 진행
- **1~5 일차**: 데이터 전처리 및 증강
- **6~11 일차**: `Cleanlab`을 활용한 데이터 노이즈 정제 및 데이터 생성 실험

<br>

## 💻 Getting Started

### ⚠️ How To Install Requirements
```bash
# 필요 라이브러리 설치
pip3 install -r requirements.txt
```

### ⌨️ How To Make Train Set
```bash
# label_noise.csv, text_noise.csv 파일 생성
filtering.ipynb

# prepare_train.sh 코드 실행 : Train dataset 을 만들기 위한 backtranslate_DeepL_JP.py,
# correct_label.py, postprocess_and_merge.py, text_clean.py, total_clean.py 실행
chmod +x prepare_train.sh # 권한 추가
./prepare_train.sh # 실행
```

### ⌨️ How To Test
```bash
# baseline code 실행
baseline_code.ipynb
```
