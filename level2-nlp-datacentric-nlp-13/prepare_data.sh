#!/bin/bash
'''
# 1. text_clean.py 실행
echo "1. text_clean.py 실행 중..."
python code/text_clean.py
if [ $? -ne 0 ]; then
    echo "text_clean.py 실행 실패!"
    exit 1
fi

# 2. correct_label.py 실행
echo "2. correct_label.py 실행 중..."
python code/correct_label.py
if [ $? -ne 0 ]; then
    echo "correct_label.py 실행 실패!"
    exit 1
fi
'''
# 3. total_clean.py 실행
echo "3. total_clean.py 실행 중..."
python code/total_clean.py
if [ $? -ne 0 ]; then
    echo "total_clean.py 실행 실패!"
    exit 1
fi

# 4. backtranslate_DeepL_JP.py 실행
echo "4. backtranslate_DeepL_JP.py 실행 중..."
python code/backtranslate_DeepL_JP.py
if [ $? -ne 0 ]; then
    echo "backtranslate_DeepL_JP.py 실행 실패!"
    exit 1
fi

# 5. postprocess_and_merge.py 실행
echo "5. postprocess_and_merge.py 실행 중..."
python code/postprocess_and_merge.py
if [ $? -ne 0 ]; then
    echo "postprocess_and_merge.py 실행 실패!"
    exit 1
fi

echo "모든 작업이 성공적으로 완료되었습니다."