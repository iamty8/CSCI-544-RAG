import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class Preprocessor:
    @staticmethod
    def remove_urls(text):
        """Remove URLs from the input text."""
        return re.sub(r'http\S+', '', text)

    @staticmethod
    def normalize_text(text):
        """
        Apply lowercasing, Unicode normalization, and punctuation removal.
        """
        text = Preprocessor.remove_urls(text)    # Remove URLs
        text = text.lower()                        # Lowercase text
        text = unicodedata.normalize('NFKD', text) # Normalize Unicode characters
        text = text.encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
        return text

    @staticmethod
    def tokenize_text(text):
        """Tokenize text into words."""
        return word_tokenize(text)

    @staticmethod
    def remove_stopwords(tokens):
        """Remove common English stopwords from a list of tokens."""
        return [token for token in tokens if token not in stop_words]

    @staticmethod
    def lemmatize_tokens(tokens):
        """Lemmatize each token to its base form."""
        return [lemmatizer.lemmatize(token) for token in tokens]

    @staticmethod
    def preprocess_text_for_sparse_methods(text):
        """
        Run the complete preprocessing pipeline for sparse methods:
          1. Normalize text (remove URLs, lowercase, Unicode normalization, remove punctuation)
          2. Tokenize
          3. Remove stopwords
          4. Lemmatize tokens
        Returns a list of processed tokens.
        """
        normalized = Preprocessor.normalize_text(text)
        tokens = Preprocessor.tokenize_text(normalized)
        tokens = Preprocessor.remove_stopwords(tokens)
        tokens = Preprocessor.lemmatize_tokens(tokens)
        return tokens

    @staticmethod
    def preprocess_text_for_dense_methods(text):
        """
        Preprocessing pipeline for dense methods:
        For dense retrieval, it's beneficial to retain natural language context.
        This method performs minimal processing:
          1. Remove URLs
          2. Lowercase and perform Unicode normalization
        Returns the cleaned text as a string (with punctuation and stopwords preserved).
        """
        # Remove URLs
        text = Preprocessor.remove_urls(text)
        # Lowercase text
        text = text.lower()
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text
