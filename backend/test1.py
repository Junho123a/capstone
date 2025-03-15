import nltk

# 필수 데이터셋 및 모델 다운로드
nltk.download('punkt')           # 문장 및 단어 토큰화를 위한 토크나이저
nltk.download('stopwords')       # 불용어 목록
nltk.download('wordnet')         # 표제어 추출을 위한 WordNet 어휘 데이터베이스
nltk.download('averaged_perceptron_tagger_eng')  # 품사 태깅을 위한 태거
nltk.download('maxent_ne_chunker_tab')  # 개체명 인식 모델
nltk.download('words')           # 개체명 인식을 위한 단어 말뭉치
nltk.download('punkt_tab')
