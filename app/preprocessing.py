# Cell 1: Imports & Setup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from collections import Counter
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Cell 2: Load & Initial Exploration
df = pd.read_csv('data/complaints_10k.csv')  # Adjust path if needed
print(f"Dataset shape: {df.shape}")
print(df.head(3))
print("\nMissing values:\n", df.isnull().sum())

# Focus on key columns for classification
df_subset = df[['Consumer complaint narrative', 'Issue', 'Product', 'Company']].copy()
df_subset = df_subset.dropna(subset=['Consumer complaint narrative', 'Issue'])  # Drop rows without narrative or label
print(f"Cleaned shape: {df_subset.shape}")

# Cell 3: Text Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Lowercase & remove non-alpha
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    # Tokenize & remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

df_subset['processed_narrative'] = df_subset['Consumer complaint narrative'].apply(preprocess_text)
print("Sample preprocessed:\n", df_subset['processed_narrative'].head())

# Cell 4: EDA - Class Distribution (Multiclass Target: Issue)
plt.figure(figsize=(12, 6))
issue_counts = df_subset['Issue'].value_counts().head(10)
sns.barplot(x=issue_counts.values, y=issue_counts.index, palette='viridis')
plt.title('Top 10 Complaint Issues (Multiclass Labels)')
plt.xlabel('Count')
plt.show()

# Narrative length stats
df_subset['narrative_len'] = df_subset['processed_narrative'].str.len()
plt.figure(figsize=(8, 4))
sns.histplot(df_subset['narrative_len'], bins=50, kde=True)
plt.title('Distribution of Preprocessed Narrative Lengths')
plt.xlabel('Length')
plt.show()

# Cell 5: Word Cloud for Top Issue (e.g., Credit Reporting)
top_issue = 'Incorrect information on your report'  # Most common
top_narratives = ' '.join(df_subset[df_subset['Issue'] == top_issue]['processed_narrative'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(top_narratives)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f'Word Cloud: {top_issue}')
plt.show()

# Cell 6: Prepare for Modeling (Split Data)
X = df_subset['processed_narrative']
y = df_subset['Issue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)
print(f"Train classes: {Counter(y_train)}")
print("Ready for classification! Next: Baseline model (e.g., TF-IDF + SVM).")