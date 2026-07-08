import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

def read_htm(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator=" ")

text_2024 = read_htm("intc-20241228.html")
text_2025 = read_htm("intc-20251227.html")

CUSTOM_STOPWORDS = {
    "also", "may", "will", "can", "including", "based", "used", "use",
    "using", "one", "two", "three", "see", "per", "net", "item", "table",
    "form", "year", "fiscal", "annual", "report", "company", "corporation",
    "inc", "corp", "ltd", "page", "total", "period", "quarter", "noted",
    "intel", "the", "and", "for", "our", "are", "was", "were", "have",
    "has", "had", "with", "that", "this", "from", "not", "but", "its",
    "been", "which"
}

ALL_STOP = set(stopwords.words("english")) | CUSTOM_STOPWORDS

def clean_and_count(text, top_n=30):
    text  = text.lower()
    text  = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in ALL_STOP and len(w) > 2]
    return Counter(words).most_common(top_n)

top_2024 = clean_and_count(text_2024, 30)
top_2025 = clean_and_count(text_2025, 30)

df_2024 = pd.DataFrame(top_2024, columns=["Word", "Freq_FY2024"])
df_2025 = pd.DataFrame(top_2025, columns=["Word", "Freq_FY2025"])

comparison = pd.merge(df_2024, df_2025, on="Word", how="outer").fillna(0)
comparison[["Freq_FY2024", "Freq_FY2025"]] = comparison[["Freq_FY2024", "Freq_FY2025"]].astype(int)
comparison = comparison.sort_values("Freq_FY2024", ascending=False).reset_index(drop=True)

print("High-Frequency Word Comparison Table (Python):")
print(comparison.to_string(index=False))

comparison.to_csv("word_frequency_comparison_Python.csv", index=False)
print("\nSaved: word_frequency_comparison_Python.csv")

def make_wordcloud(freq_list, title, filename, colormap="Dark2"):
    wc = WordCloud(width=800, height=600, background_color="white",
                   colormap=colormap, max_words=100)
    wc.generate_from_frequencies(dict(freq_list))
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

make_wordcloud(top_2024, "Intel 10-K FY2024 – Word Cloud (Python)", "wordcloud_FY2024_Python.png", "Dark2")
make_wordcloud(top_2025, "Intel 10-K FY2025 – Word Cloud (Python)", "wordcloud_FY2025_Python.png", "Set1")