import re
from collections import Counter
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api

# ---------------------------
# 0. NLTK downloads (run once)
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')

# ---------------------------
# 1. Load PDF (by pages)
# ---------------------------
pdf_path = "F:\harrypotter (1).pdf"   # your file
reader = PdfReader(pdf_path)

pages_text = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text is None:
        text = ""   # handle empty pages gracefully
    pages_text.append(text)

print(f"Loaded {len(pages_text)} pages from PDF.")

# ---------------------------
# 2. Preprocessing helper
# ---------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text, remove_stopwords=True):
    # lower, remove non-alphabetic (keeps words), tokenize
    text = text.lower()
    # keep basic punctuation removed
    text = re.sub(r'[_\\d]+',' ', text)               # drop numbers/underscores
    text = re.sub(r'[^\w\s]', ' ', text)             # drop punctuation
    tokens = word_tokenize(text)
    # keep alphabetic tokens only
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    return tokens

# Preprocess each page and build a version of pages suitable for TF-IDF
tokens_per_page = [preprocess_text(p) for p in pages_text]
pages_for_tfidf = [" ".join(tokens) for tokens in tokens_per_page]  # each page is a document

# ---------------------------
# 3. Generate n-grams and counts (global)
# ---------------------------
def ngram_counter(tokens_list, n):
    ctr = Counter()
    for tokens in tokens_list:
        ngs = ngrams(tokens, n)
        # join tokens to readable string for counting
        ngs_joined = (" ".join(ng) for ng in ngs)
        ctr.update(ngs_joined)
    return ctr

unigram_counts = ngram_counter(tokens_per_page, 1)
bigram_counts  = ngram_counter(tokens_per_page, 2)
trigram_counts = ngram_counter(tokens_per_page, 3)

print("\nTop 15 unigrams:")
print(unigram_counts.most_common(15))

print("\nTop 15 bigrams:")
print(bigram_counts.most_common(15))

print("\nTop 15 trigrams:")
print(trigram_counts.most_common(15))

# ---------------------------
# 4. TF-IDF across pages
# ---------------------------
tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=20000)
tfidf_matrix = tfidf_vectorizer.fit_transform(pages_for_tfidf)  # shape: (n_pages, n_terms)
print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")

# Example: top terms by average TF-IDF across pages
import numpy as np
tfidf_mean = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
terms = np.array(tfidf_vectorizer.get_feature_names_out())
top_idx = tfidf_mean.argsort()[::-1][:20]
print("\nTop TF-IDF terms (by average across pages):")
for i in top_idx:
    print(f"{terms[i]} : {tfidf_mean[i]:.4f}")

# ---------------------------
# 5. GloVe embeddings (average of word vectors)
# ---------------------------
# This will download GloVe the first time (~50-200MB depending on dim)
print("\nLoading GloVe vectors (this may take a while the first time)...")
glove = api.load("glove-wiki-gigaword-100")  # 100-dim vectors; change if you prefer 50/200/300

vector_size = glove.vector_size

def avg_glove_vector(tokens):
    vecs = []
    for t in tokens:
        if t in glove:
            vecs.append(glove[t])
    if len(vecs) == 0:
        return np.zeros(vector_size, dtype=float)
    return np.mean(vecs, axis=0)

# Compute embeddings per page
glove_per_page = np.vstack([avg_glove_vector(tokens) for tokens in tokens_per_page])  # shape: (n_pages, dim)
print(f"GloVe per-page shape: {glove_per_page.shape}")

# And an overall document vector (average of page vectors or of all tokens)
all_tokens = [t for toks in tokens_per_page for t in toks]
doc_glove_vector = avg_glove_vector(all_tokens)
print(f"Document-level GloVe vector shape: {doc_glove_vector.shape}")

# ---------------------------
# 6. Example: combine TF-IDF + Glove for a downstream model
# ---------------------------
# As an example, create a DataFrame with page-level features: page_index, glove_emb, and optionally top tfidf features
page_df = pd.DataFrame({
    'page_text': pages_text,
    'page_tokens': [" ".join(toks) for toks in tokens_per_page]
})

# Add flattened glove columns (g_0 ... g_99)
for i in range(vector_size):
    page_df[f"g_{i}"] = glove_per_page[:, i]

print("\nSample page dataframe columns:", page_df.columns[:10])
print("Ready to use page_df (glove columns) and tfidf_matrix for modeling.")