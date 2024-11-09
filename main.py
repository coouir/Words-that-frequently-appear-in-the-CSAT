import fitz  # PDF 텍스트 추출
import re  # 정규 표현식
import os  # 파일 시스템
import matplotlib.pyplot as plt  # 시각화
import nltk  # 자연어 처리
from nltk.corpus import stopwords  # 불용어 목록
from nltk.stem import WordNetLemmatizer  # 동사 원형화
import ssl  # SSL
import pandas as pd  # pandas

# 불용어 및 WordNet 리소스 다운로드
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('stopwords')
nltk.download('wordnet')  
nltk.download('omw-1.4')  

# Lemmatizer 초기화
lemmatizer = WordNetLemmatizer()

# PDF에서 단어 빈도 추출
def extract_word_counts_from_pdfs(pdf_paths):
    word_count = {}  # 단어 빈도 저장
    stop_words = set(stopwords.words('english'))  # 불용어 목록

    # 이름 확인 함수
    def is_person_name(word):
        return word[0].isupper()  # 대문자 시작은 이름으로 간주

    # PDF 처리
    for pdf_path in pdf_paths:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            page_text = page.get_text()
            words = re.findall(r'[A-Za-z]+', page_text)

            # 단어 원형화 및 빈도 카운트
            for word in words:
                if len(word) > 1 and word.lower() not in stop_words and not is_person_name(word):
                    lemmatized_word = lemmatizer.lemmatize(word.lower(), pos='v')  # 동사 원형화
                    word_count[lemmatized_word] = word_count.get(lemmatized_word, 0) + 1

    return word_count

# PDF 파일 경로 가져오기
def get_pdf_paths(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.lower().endswith('.pdf')]

# PDF 경로 리스트 가져오기
pdf_paths = get_pdf_paths("pdfs")

# 단어 빈도 계산
word_counts = extract_word_counts_from_pdfs(pdf_paths)

# 빈도수 내림차순 정렬
sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

# 상위 1000개 단어를 CSV로 저장
df = pd.DataFrame(sorted_word_counts[:1000], columns=['Word', 'Frequency'])
df.to_csv('word_frequencies.csv', index=False)

# 상위 100개 단어 시각화
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