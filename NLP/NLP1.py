import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------
# Download required NLTK data
# ------------------------------
nltk.download('punkt')
nltk.download('punkt_tab')   # ✅ REQUIRED FIX
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ------------------------------
# 1. Original Text
# ------------------------------
text = """I only have a day-off on Sunday, so I have only a little free time. Sunday is a wonderful day for me to spend time with my friend. One of the things I really enjoy doing on Sunday morning is to play chess. It is time to relax and talk about the events of the previous week and future plans.

On Sunday morning, I often sing with my friends at a karaoke restaurant and we all have a good time there. It is especially funny. When I get tired, I stop singing. Please don’t tell him I said this, but he is a very bad singer!

Once in a while, I go for a walk on Sundays with my friends. Sometimes, I just stay at home and listen to music, watch television, or read novels. Do you feel bored when you hear about my free time, teacher?"""

# ------------------------------
# 2. Text Preprocessing
# ------------------------------
# Lowercase
text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# Tokenization
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# POS tagging
pos_tags = pos_tag(stemmed_tokens)

# ------------------------------
# 3. Bag of Words
# ------------------------------
processed_text = " ".join(stemmed_tokens)

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform([processed_text])

bow_array = bow_matrix.toarray()[0]
feature_names = vectorizer.get_feature_names_out()
word_counts = dict(zip(feature_names, bow_array))

# ------------------------------
# 4. Print Results
# ------------------------------
print("Tokens:\n", tokens)
print("\nAfter Stopword Removal:\n", filtered_tokens)
print("\nAfter Stemming:\n", stemmed_tokens)
print("\nPOS Tags:\n", pos_tags)
print("\nWords with their counts (Bag of Words):\n")

for word, count in word_counts.items():
    print(f"{word}: {count}")
