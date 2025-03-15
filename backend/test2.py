import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# 기사 텍스트
article = """
The European Union has announced new sanctions against Russia over its invasion of Ukraine.
The sanctions target Russia's energy sector and several high-profile individuals close to President Vladimir Putin.
EU officials say the measures are designed to increase pressure on Moscow to end the conflict.
"""

# 문장 분리
sentences = sent_tokenize(article)

# 단어 분리 및 빈도 분석
words = word_tokenize(article)
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

fdist = FreqDist(filtered_words)
print(fdist.most_common(5))  # 가장 빈번한 5개 단어 출력
