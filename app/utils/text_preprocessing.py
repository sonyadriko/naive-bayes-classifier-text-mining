"""Text preprocessing module for Indonesian sentiment analysis.

This module provides a complete preprocessing pipeline for Indonesian text,
including case folding, cleaning, tokenization, stopword removal, normalization,
and stemming using Sastrawi.
"""

import re
import string
from pathlib import Path
from typing import Callable


class TextPreprocessor:
    """Text preprocessor for Indonesian language.

    Pipeline:
    1. Case Folding - Convert to lowercase
    2. Cleaning - Remove URLs, mentions, special chars, numbers
    3. Tokenization - Split into words
    4. Stopword Removal - Remove common Indonesian words
    5. Normalization - Convert informal words to formal (e.g., "yg" -> "yang")
    6. Stemming - Reduce words to root form using Sastrawi

    Example:
        ```python
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Aplikasinya SANGAT bagus!!")
        # Returns: "aplikasi bagus"
        ```
    """

    # Indonesian stopwords (common words to remove)
    INDONESIAN_STOPWORDS = {
        "ada", "adalah", "adanya", "akankah", "akan", "akanlah", "aku", "akulah",
        "amat", "amatlah", "anda", "andalah", "antar", "antara", "antaranya",
        "apakah", "apakah", "apatah", "atau", "ataukah", "ataupun", "bagai",
        "bagaimana", "bagaimanakah", "bagaimanapun", "bagi", "bagian", "bahkan",
        "bahwa", "bahwasanya", "bukan", "bukankah", "bukanlah", "bukanlah",
        "juga", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kamu",
        "kamulah", "kamu", "kan", "kapan", "kapankah", "kapanpun", "karena",
        "karenanya", "ke", "kecil", "kemudian", "kenapa", "kepada", "kepadanya",
        "ketika", "khususnya", "kini", "kita", "kitalah", "lagi", "lagian",
        "lah", "lain", "lainnya", "lalai", "lama", "lamanya", "lebih", "luar",
        "macam", "maka", "makanya", "makin", "malah", "malahan", "mampu",
        "mampukah", "mana", "manakah", "manalagi", "masih", "masihkah", "masing",
        "mau", "maukah", "melainkan", "melalui", "memang", "mengapa", "mereka",
        "mereka", "merekalah", "merupakan", "meski", "meskipun", "mungkin",
        "mungkinkah", "nah", "namun", "nanti", "nantinya", "nyaris", "oleh",
        "olehnya", "pada", "padahal", "padanya", "paling", "pantas", "para",
        "pasti", "pastilah", "per", "pernah", "pukul", "saja", "sajakah",
        "saling", "sama", "sama", "sampaikan", "sangat", "sangatlah", "saya",
        "sayalah", "se", "sebab", "sebabnya", "sebagai", "sebagaimana",
        "sebagainya", "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya",
        "sebenarnya", "sebuah", "sedapat", "sedikit", "sedikitlah", "segala",
        "segalanya", "segera", "seharusnya", "sehingga", "seingat", "sejak",
        "sejauh", "sejenak", "sekadar", "sekadarnya", "sekali", "sekalian",
        "sekaligus", "sekalipun", "sekarang", "seketika", "sekiranya", "sekitar",
        "sela", "selain", "selaku", "selalu", "selama", "selamanya", "selaras",
        "selepas", "selnya", "semakin", "semakin", "sementara", "sempat", "semua",
        "semua", "semualah", "semua", "semestinya", "sendiri", "sendirian",
        "seolah", "seorang", "sepanjang", "seperti", "sepertinya", "sering",
        "seringnya", "serupa", "sesaat", "sesama", "sesegera", "seseorang",
        "sesuatu", "sesuatunya", "sesudah", "sesudahnya", "setelah", "setiap",
        "sewaktu", "siapakah", "sini", "sinilah", "soal", "suatu", "sudah",
        "sudahlah", "supaya", "tadi", "tadinya", "tak", "tanpa", "tapi",
        "telah", "tentang", "tentu", "tentulah", "tentunya", "terdiri", "terhadap",
        "terhadapnya", "terlalu", "terlebih", "tersebut", "tersebutlah", "tersurat",
        "tertuju", "tetapi", "tiap", "tidak", "tidakkah", "tidaklah", "toh",
        "waduh", "wah", "wahai", "walau", "walaupun", "wong", "yaitu", "yaitulah",
        "yang",
    }

    def __init__(
        self,
        use_stopwords: bool = True,
        use_stemming: bool = True,
        use_normalization: bool = True,
        custom_stopwords: set[str] | None = None,
        normalization_path: str | None = None,
    ) -> None:
        """Initialize text preprocessor.

        Args:
            use_stopwords: Whether to remove stopwords.
            use_stemming: Whether to apply stemming (requires Sastrawi).
            use_normalization: Whether to normalize informal words.
            custom_stopwords: Additional stopwords to remove.
            normalization_path: Path to normalization dictionary file.
        """
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self.use_normalization = use_normalization

        # Merge default and custom stopwords
        self.stopwords = self.INDONESIAN_STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        # Load normalization dictionary
        self.normalization_dict: dict[str, str] = {}
        if use_normalization:
            self._load_normalization_dict(normalization_path)

        # Initialize stemmer (lazy loading)
        self._stemmer: Callable[[str], str] | None = None

    def _load_normalization_dict(self, path: str | None = None) -> None:
        """Load normalization dictionary from file.

        Args:
            path: Path to normalization file (format: "informal,formal" per line).
        """
        if path is None:
            path = Path(__file__).parent.parent.parent / "data" / "dictionaries" / "normalisasi.txt"

        norm_path = Path(path)
        if norm_path.exists():
            try:
                with open(norm_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "," in line:
                            informal, formal = line.split(",", 1)
                            self.normalization_dict[informal.strip()] = formal.strip()
            except Exception as e:
                print(f"Warning: Failed to load normalization dict: {e}")

    def _get_stemmer(self) -> Callable[[str], str]:
        """Get Sastrawi stemmer (lazy loading).

        Returns:
            Stemmer function or identity if Sastrawi not available.
        """
        if self._stemmer is None:
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                self._stemmer = stemmer.stem
            except ImportError:
                # Sastrawi not installed, return identity function
                self._stemmer = lambda x: x
                print("Warning: Sastrawi not installed. Stemming disabled.")

        return self._stemmer

    def case_folding(self, text: str) -> str:
        """Convert text to lowercase.

        Args:
            text: Input text.

        Returns:
            Lowercase text.
        """
        return text.lower()

    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, mentions, special chars, and numbers.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+|https?\.\S+", "", text)

        # Remove mentions and hashtags
        text = re.sub(r"@\w+|#\w+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        return text.split()

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords from tokens.

        Args:
            tokens: List of tokens.

        Returns:
            Filtered tokens without stopwords.
        """
        if not self.use_stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]

    def normalize_words(self, tokens: list[str]) -> list[str]:
        """Normalize informal words to formal Indonesian.

        Args:
            tokens: List of tokens.

        Returns:
            Normalized tokens.
        """
        if not self.use_normalization or not self.normalization_dict:
            return tokens
        return [self.normalization_dict.get(t, t) for t in tokens]

    def stem_words(self, tokens: list[str]) -> list[str]:
        """Stem words to root form.

        Args:
            tokens: List of tokens.

        Returns:
            Stemmed tokens.
        """
        if not self.use_stemming:
            return tokens

        stemmer = self._get_stemmer()
        return [stemmer(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        """Run full preprocessing pipeline.

        Args:
            text: Input text.

        Returns:
            Preprocessed text (joined tokens).
        """
        # Step 1: Case folding
        text = self.case_folding(text)

        # Step 2: Clean text
        text = self.clean_text(text)

        # Step 3: Tokenize
        tokens = self.tokenize(text)

        # Skip if no tokens
        if not tokens:
            return ""

        # Step 4: Normalize (before stopword removal for better matching)
        tokens = self.normalize_words(tokens)

        # Step 5: Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Step 6: Stem
        tokens = self.stem_words(tokens)

        return " ".join(tokens)

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Preprocess multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of preprocessed texts.
        """
        return [self.preprocess(text) for text in texts]

    def get_vocabulary(self, texts: list[str]) -> set[str]:
        """Get unique words from preprocessed texts.

        Args:
            texts: List of input texts.

        Returns:
            Set of unique words (vocabulary).
        """
        preprocessed = [self.preprocess(text) for text in texts]
        vocab = set()
        for text in preprocessed:
            if text:
                vocab.update(text.split())
        return vocab

    def add_custom_stopwords(self, words: list[str] | set[str]) -> None:
        """Add custom stopwords.

        Args:
            words: Words to add to stopwords.
        """
        self.stopwords.update(words)

    def load_stopwords_from_file(self, filepath: str) -> None:
        """Load stopwords from file (one word per line).

        Args:
            filepath: Path to stopwords file.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                words = {line.strip() for line in f if line.strip()}
            self.stopwords.update(words)
        except Exception as e:
            print(f"Warning: Failed to load stopwords from {filepath}: {e}")


class BagOfWords:
    """Bag of Words feature extractor for text classification.

    Converts text documents to numerical feature vectors using vocabulary.
    """

    def __init__(self, min_freq: int = 1, max_features: int | None = None) -> None:
        """Initialize Bag of Words.

        Args:
            min_freq: Minimum frequency for a word to be included.
            max_features: Maximum number of features (most frequent).
        """
        self.min_freq = min_freq
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str], preprocessor: TextPreprocessor | None = None) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of preprocessed texts.
            preprocessor: Optional preprocessor to apply first.
        """
        # Preprocess if needed
        if preprocessor:
            texts = preprocessor.preprocess_batch(texts)

        # Count word frequencies
        word_freq: dict[str, int] = {}
        doc_freq: dict[str, int] = {}

        for text in texts:
            if not text:
                continue
            words = set(text.split())
            for word in words:
                doc_freq[word] = doc_freq.get(word, 0) + 1

            words_list = text.split()
            for word in words_list:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Filter by minimum frequency
        filtered = {w: f for w, f in word_freq.items() if f >= self.min_freq}

        # Sort by frequency and limit features
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        # Build vocabulary
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}

        # Calculate IDF (for TF-IDF)
        n_docs = len([t for t in texts if t])
        for word in self.vocabulary:
            self.idf[word] = __import__("math").log(n_docs / (doc_freq.get(word, 0) + 1))

    def transform(self, texts: list[str], use_tf_idf: bool = False) -> list[dict[str, int | float]]:
        """Transform texts to feature vectors.

        Args:
            texts: List of preprocessed texts.
            use_tf_idf: Use TF-IDF weighting instead of count.

        Returns:
            List of feature dictionaries (word -> count/weight).
        """
        if not self.vocabulary:
            raise ValueError("Vocabulary not fitted. Call fit() first.")

        features = []
        for text in texts:
            if not text:
                features.append({word: 0 for word in self.vocabulary})
                continue

            words = text.split()
            word_counts: dict[str, int] = {}

            for word in words:
                if word in self.vocabulary:
                    word_counts[word] = word_counts.get(word, 0) + 1

            # Convert to TF-IDF if requested
            if use_tf_idf:
                feature_dict = {
                    word: (count / len(words)) * self.idf.get(word, 1)
                    for word, count in word_counts.items()
                }
                # Fill missing words with 0
                for word in self.vocabulary:
                    if word not in feature_dict:
                        feature_dict[word] = 0.0
            else:
                # Fill missing words with 0
                feature_dict = {word: word_counts.get(word, 0) for word in self.vocabulary}

            features.append(feature_dict)

        return features

    def fit_transform(self, texts: list[str], preprocessor: TextPreprocessor | None = None) -> list[dict[str, int]]:
        """Fit vocabulary and transform texts.

        Args:
            texts: List of texts.
            preprocessor: Optional preprocessor.

        Returns:
            List of feature dictionaries.
        """
        self.fit(texts, preprocessor)
        return self.transform(texts)

    def get_vocabulary_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of words in vocabulary.
        """
        return len(self.vocabulary)

    def get_vocabulary(self) -> list[str]:
        """Get vocabulary words.

        Returns:
            List of vocabulary words.
        """
        return list(self.vocabulary.keys())
