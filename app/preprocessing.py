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
import yaml

# It's better to ensure these are downloaded once as part of setup,
# but for now we'll leave them here for simplicity.
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# Cell 3: Text Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, min_token_length=3):
    """
    Cleans and preprocesses raw text by:
    1. Lowercasing
    2. Removing non-alphabetic characters
    3. Tokenizing
    4. Removing stopwords
    5. Lemmatizing
    """
    if pd.isna(text):
        return ""
    # Lowercase & remove non-alpha
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    # Tokenize & remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) >= min_token_length]
    return ' '.join(tokens)

def main():
    """Main function to run the data loading, preprocessing, and EDA."""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Cell 2: Load & Initial Exploration
    df = pd.read_csv(config['data']['raw_data_path'])
    print(f"Dataset shape: {df.shape}")
    print(df.head(3))
    print("\nMissing values:\n", df.isnull().sum())

    # Focus on key columns for classification
    narrative_col = config['data']['narrative_column']
    target_col = config['data']['target_column']
    df_subset = df[[narrative_col, target_col, 'Product', 'Company']].copy()
    df_subset = df_subset.dropna(subset=[narrative_col, target_col])
    print(f"Cleaned shape: {df_subset.shape}")

    min_token_len = config['preprocessing']['min_token_length']
    df_subset['processed_narrative'] = df_subset[narrative_col].apply(
        lambda x: preprocess_text(x, min_token_length=min_token_len)
    )
    print("Sample preprocessed:\n", df_subset['processed_narrative'].head())

    # Cell 4: EDA - Class Distribution (Multiclass Target: Issue)
    plt.figure(figsize=(12, 6))
    issue_counts = df_subset[target_col].value_counts().head(10)
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
    # Note: This part is for EDA and uses a hardcoded value for the top issue.
    # This is acceptable for exploratory analysis.
    top_issue = 'Incorrect information on your report'
    top_narratives = ' '.join(df_subset[df_subset[target_col] == top_issue]['processed_narrative'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(top_narratives)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {top_issue}')
    plt.show()

    # Cell 6: Prepare for Modeling (Split Data)
    X = df_subset['processed_narrative']
    y = df_subset[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config['data']['test_split_size'],
        random_state=config['training']['random_seed']
    )
    print(f"Train classes: {Counter(y_train)}")
    print("Ready for classification! Next: Baseline model (e.g., TF-IDF + SVM).")

if __name__ == '__main__':
    main()