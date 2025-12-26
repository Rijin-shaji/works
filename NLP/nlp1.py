
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

text = "Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence!"

text = text.lower()

text = text.translate(str.maketrans('', '', string.punctuation))

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

pos_tags = pos_tag(stemmed_tokens)

print("Original Text:")
print(text)
print("\nTokens:")
print(tokens)
print("\nAfter Stopword Removal:")
print(filtered_tokens)
print("\nAfter Stemming:")
print(stemmed_tokens)
print("\nPOS Tags:")
print(pos_tags)
