import PyPDF2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import string
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# ------------------------------
# 1. Read PDF File
# ------------------------------
pdf_path = "F:\harrypotter (1).pdf"

text = ""
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()

# ------------------------------
# 2. Text Preprocessing
# ------------------------------
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in text.split() if word not in stop_words]

processed_text = " ".join(filtered_words)

# ------------------------------
# 3. Generate Word Cloud
# ------------------------------
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(processed_text)

# ------------------------------
# 4. Display Word Cloud
# ------------------------------
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
