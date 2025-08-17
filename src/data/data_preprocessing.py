import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any, Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging for the module
logging.basicConfig(
    filename="logs/preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logging.info("NLTK resources downloaded successfully")
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {e}")
    raise

def lemmatization(text: str) -> str:
    """
    Lemmatize each word in the text.
    Args:
        text (str): Input text.
    Returns:
        str: Lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def remove_stop_words(text: str) -> str:
    """
    Remove stop words from the text.
    Args:
        text (str): Input text.
    Returns:
        str: Text without stop words.
    """
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in str(text).split() if word not in stop_words]
    return " ".join(filtered)

def removing_numbers(text: str) -> str:
    """
    Remove all digits from the text.
    Args:
        text (str): Input text.
    Returns:
        str: Text without digits.
    """
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    """
    Convert all words in the text to lowercase.
    Args:
        text (str): Input text.
    Returns:
        str: Lowercased text.
    """
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text: str) -> str:
    """
    Remove punctuations and extra whitespace from the text.
    Args:
        text (str): Input text.
    Returns:
        str: Text without punctuations.
    """
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    """
    Remove URLs from the text.
    Args:
        text (str): Input text.
    Returns:
        str: Text without URLs.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    """
    Set text to NaN if sentence has fewer than 3 words.
    Args:
        df (pd.DataFrame): DataFrame with 'text' column.
    """
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the 'content' column of the DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    try:
        df.content = df.content.apply(lower_case)
        df.content = df.content.apply(remove_stop_words)
        df.content = df.content.apply(removing_numbers)
        df.content = df.content.apply(removing_punctuations)
        df.content = df.content.apply(removing_urls)
        df.content = df.content.apply(lemmatization)
        logging.info("Text normalization completed")
        return df
    except Exception as e:
        logging.error(f"Text normalization failed: {e}")
        raise

def normalized_sentence(sentence: str) -> str:
    """
    Apply all preprocessing steps to a single sentence.
    Args:
        sentence (str): Input sentence.
    Returns:
        str: Normalized sentence.
    """
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Sentence normalization failed: {e}")
        raise

def main() -> None:
    """
    Main function to orchestrate data preprocessing steps.
    """
    try:
        # Load raw train and test data
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logging.info("Raw train and test data loaded")

        # Normalize train and test data
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        # Save processed data to CSV files
        os.makedirs("data/processed", exist_ok=True)  # Ensure the directory exists
        train_data.to_csv("data/processed/train.csv", index=False)
        test_data.to_csv("data/processed/test.csv", index=False)
        logging.info("Processed train and test data saved to data/processed")
    except Exception as e:
        logging.error(f"Data preprocessing pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()