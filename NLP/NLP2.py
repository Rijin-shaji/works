import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

# Download tokenizer
nltk.download('punkt')

# Input sentence
text = "The quick brown fox jumps over the lazy dog"

# Tokenization
tokens = word_tokenize(text.lower())

# Unigram (1-gram)
unigrams = list(ngrams(tokens, 1))

# Bigram (2-gram)
bigrams = list(ngrams(tokens, 2))

# Trigram (3-gram)
trigrams = list(ngrams(tokens, 3))

# Print results
print("Tokens:")
print(tokens)

print("\nUnigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)
