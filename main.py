import fitz  # PyMuPDF, PDF 텍스트 추출
import re  # 정규 표현식
import os  # 파일 시스템
import matplotlib.pyplot as plt  # 시각화
import nltk  # 자연어 처리
from nltk.corpus import stopwords  # 불용어 목록
import ssl  # SSL
import pandas as pd  # pandas 추가

# 불용어 목록 다운로드
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')

def extract_word_counts_from_pdfs(pdf_paths):
    word_count = {}  # 단어 빈도 저장
    stop_words = set(stopwords.words('english'))  # 불용어 목록

    def is_person_name(word):
        return word[0].isupper()  # 대문자 시작은 이름으로 간주

    # 각 PDF 처리
    for pdf_path in pdf_paths:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_text = page.get_text()
            words = re.findall(r'[A-Za-z]+', page_text)

            for word in words:
                if len(word) > 1 and word not in stop_words and not is_person_name(word):
                    word_count[word] = word_count.get(word, 0) + 1

    return word_count

def get_pdf_paths(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.lower().endswith('.pdf')]

# PDF 경로 리스트 가져오기
pdf_paths = get_pdf_paths("pdfs")

# 단어 빈도 계산
word_counts = extract_word_counts_from_pdfs(pdf_paths)

# 빈도수 내림차순 정렬
sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

# DataFrame으로 상위 1000개 단어 저장
df = pd.DataFrame(sorted_word_counts[:1000], columns=['Word', 'Frequency'])
df.to_csv('word_frequencies.csv', index=False)

# 시각화
words, counts = zip(*sorted_word_counts[:100])
plt.figure(figsize=(12, 8))
plt.barh(words, counts, color='skyblue')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title('Words that frequently appear in the CSAT')
plt.yticks(fontsize=7)
plt.gca().invert_yaxis()
plt.xticks(range(0, max(counts)+20, 20))
plt.tight_layout()
plt.show()
