"""
Offline text retrieval using BM25 or TF-IDF with optional enhancements.

This module provides text retrieval functionality for RAG (Retrieval-Augmented Generation)
pipelines. It supports both basic retrieval and enhanced retrieval with:

1. QUERY EXPANSION: Map symptoms to component/cause keywords
   - "no steam" -> "no steam valve blocked pressure wand"
   - Configurable symptom-to-cause dictionary

2. CONTEXT WINDOWS: Include N chunks before/after each retrieved chunk
   - Maintains document order (not relevance order)
   - Removes duplicates when windows overlap

3. BETTER TOKENIZATION: Stop word removal and basic stemming
   - Pure Python implementation (no external dependencies)
   - Optional features for flexibility

4. QUALITY METRICS: Scores, thresholds, and debugging
   - Return scores alongside chunks
   - Filter by minimum score threshold
   - Debug logging for retrieval quality analysis

No external dependencies required for core functionality.
Falls back from BM25 to TF-IDF if rank_bm25 not installed.

Example usage:
    # Basic usage (backwards compatible)
    chunks = retrieve_chunks(query, manual_chunks, top_k=6)

    # With all enhancements
    chunks = retrieve_chunks_enhanced(
        query="lights off",
        chunks=manual_chunks,
        top_k=8,
        enable_expansion=True,
        window_size=1,
        return_scores=True
    )

    # Debugging low retrieval quality
    results = retrieve_chunks_enhanced(
        query="thermal fuse",
        chunks=manual_chunks,
        top_k=10,
        min_score=0.1,
        return_scores=True,
        enable_logging=True
    )
    for chunk, score in results:
        print(f"{score:.3f}: {chunk[:80]}...")
"""

from typing import List, Tuple, Dict, Optional, Union, Set
import re
import math
from collections import Counter


# ================= Stop Words (English) =================
# Common English stop words that don't contribute to retrieval quality.
# Removing these improves precision by focusing on content words.

ENGLISH_STOP_WORDS: Set[str] = {
    # Articles
    "a", "an", "the",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    # Verbs (common)
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "should", "could", "ought", "might", "must", "shall", "will", "can",
    # Prepositions
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    # Conjunctions
    "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "also", "now", "any", "all",
}


# ================= Symptom-to-Cause Expansion Dictionary =================
# Maps common appliance symptoms to related component/cause keywords.
# This improves recall by expanding queries with domain-specific terms.
# Users can extend this dictionary for their specific domain.

DEFAULT_SYMPTOM_EXPANSION: Dict[str, List[str]] = {
    # Steam/pressure issues
    "no steam": ["valve", "blocked", "pressure", "wand", "boiler", "thermostat"],
    "low steam": ["valve", "pressure", "scale", "buildup", "descale"],
    "steam leak": ["gasket", "seal", "valve", "o-ring", "crack"],

    # Water issues
    "no water": ["pump", "inlet", "valve", "filter", "blocked", "clogged"],
    "water leak": ["hose", "seal", "gasket", "pump", "connection", "crack"],
    "not draining": ["pump", "drain", "filter", "blockage", "clogged", "hose"],
    "won't drain": ["pump", "drain", "filter", "blockage", "clogged", "hose"],

    # Heating issues
    "not heating": ["element", "thermostat", "thermal", "fuse", "relay", "control"],
    "won't heat": ["element", "thermostat", "thermal", "fuse", "relay", "control"],
    "overheating": ["thermostat", "thermal", "fuse", "sensor", "control", "relay"],
    "cold": ["element", "thermostat", "thermal", "fuse", "heating"],

    # Power issues
    "no power": ["fuse", "switch", "cord", "plug", "relay", "circuit", "breaker"],
    "won't turn on": ["power", "switch", "fuse", "cord", "relay", "control"],
    "turns off": ["thermal", "fuse", "overload", "circuit", "safety"],
    "lights off": ["bulb", "led", "power", "control", "board", "fuse"],

    # Motor issues
    "not spinning": ["motor", "belt", "capacitor", "bearing", "brush"],
    "won't spin": ["motor", "belt", "capacitor", "bearing", "brush", "clutch"],
    "loud noise": ["bearing", "motor", "pump", "fan", "belt", "drum"],
    "vibration": ["balance", "bearing", "mount", "shock", "spring", "drum"],

    # Display/control issues
    "error code": ["sensor", "control", "board", "module", "pcb"],
    "display blank": ["power", "board", "control", "fuse", "connection"],
    "buttons not working": ["control", "board", "membrane", "switch", "pcb"],

    # Door issues
    "door won't open": ["latch", "lock", "interlock", "solenoid", "handle"],
    "door won't close": ["latch", "hinge", "strike", "alignment", "seal"],

    # Cooling issues (refrigerators, AC)
    "not cooling": ["compressor", "fan", "thermostat", "refrigerant", "coil"],
    "frost buildup": ["defrost", "heater", "timer", "thermostat", "seal"],
    "ice maker": ["valve", "thermostat", "motor", "arm", "water"],
}


# ================= Basic Stemming Rules =================
# Simple suffix-stripping stemmer (Porter-lite).
# Pure Python implementation - no external dependencies.

STEMMING_RULES: List[Tuple[str, str]] = [
    # Verb forms
    ("ying", "y"),      # drying -> dry
    ("bing", "b"),      # rubbing -> rub (handle doubling)
    ("ning", "n"),      # running -> run
    ("ting", "t"),      # hitting -> hit
    ("ping", "p"),      # stopping -> stop
    ("ming", "m"),      # swimming -> swim
    ("ging", "g"),      # logging -> log
    ("ding", "d"),      # adding -> add
    ("ing", ""),        # heating -> heat
    ("ied", "y"),       # carried -> carry
    ("ed", ""),         # blocked -> block
    ("es", ""),         # switches -> switch
    ("s", ""),          # pumps -> pump

    # Noun forms
    ("tion", "t"),      # connection -> connect
    ("sion", "s"),      # expansion -> expans
    ("ment", ""),       # replacement -> replace
    ("ness", ""),       # darkness -> dark
    ("ity", ""),        # capacity -> capac
    ("er", ""),         # heater -> heat
    ("or", ""),         # motor -> mot (careful - short words)
    ("ly", ""),         # quickly -> quick
    ("al", ""),         # thermal -> therm
]

# Minimum word length after stemming (avoid over-stemming short words)
MIN_STEM_LENGTH = 3


# ================= Tokenization Functions =================

def _tokenize(text: str) -> List[str]:
    """
    Simple tokenization: lowercase + split on non-word chars.

    This is the original tokenizer, preserved for backwards compatibility.

    Args:
        text: Input text

    Returns:
        List of lowercase tokens
    """
    return re.findall(r'\w+', text.lower())


def _stem_word(word: str) -> str:
    """
    Apply simple suffix-stripping stemming to a word.

    Uses a Porter-lite approach with common English suffixes.
    Pure Python implementation - no external dependencies.

    Args:
        word: Word to stem (should be lowercase)

    Returns:
        Stemmed word
    """
    if len(word) <= MIN_STEM_LENGTH:
        return word  # Don't stem short words

    for suffix, replacement in STEMMING_RULES:
        if word.endswith(suffix):
            stemmed = word[:-len(suffix)] + replacement
            # Only accept if result is long enough
            if len(stemmed) >= MIN_STEM_LENGTH:
                return stemmed

    return word


def _tokenize_enhanced(
    text: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = True
) -> List[str]:
    """
    Enhanced tokenization with optional stop word removal and stemming.

    This improves retrieval quality by:
    1. Removing common English stop words (the, a, is, etc.)
    2. Applying basic stemming (heating -> heat, pumps -> pump)

    Both features are optional and can be disabled for specific use cases.

    Args:
        text: Input text to tokenize
        remove_stopwords: Whether to remove English stop words
        apply_stemming: Whether to apply basic stemming

    Returns:
        List of processed tokens
    """
    # Basic tokenization
    tokens = re.findall(r'\w+', text.lower())

    # Remove stop words (optional)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    # Apply stemming (optional)
    if apply_stemming:
        tokens = [_stem_word(t) for t in tokens]

    return tokens


# ================= Query Expansion =================

def expand_diagnostic_query(
    query: str,
    expansion_dict: Optional[Dict[str, List[str]]] = None,
    max_expansions: int = 6
) -> str:
    """
    Expand a diagnostic query with symptom-to-cause keywords.

    This improves recall by adding related technical terms to queries.
    For example: "no steam" -> "no steam valve blocked pressure wand"

    The expansion is performed by:
    1. Looking for known symptom patterns in the query
    2. Adding related component/cause keywords
    3. Limiting total expansions to avoid query drift

    Args:
        query: Original search query
        expansion_dict: Custom symptom-to-cause dictionary
                       (defaults to DEFAULT_SYMPTOM_EXPANSION)
        max_expansions: Maximum number of keywords to add

    Returns:
        Expanded query string

    Example:
        >>> expand_diagnostic_query("no steam")
        "no steam valve blocked pressure wand boiler thermostat"
    """
    if expansion_dict is None:
        expansion_dict = DEFAULT_SYMPTOM_EXPANSION

    query_lower = query.lower()
    expansions: List[str] = []

    # Check each symptom pattern
    for symptom, keywords in expansion_dict.items():
        if symptom in query_lower:
            # Add keywords that aren't already in the query
            for keyword in keywords:
                if keyword not in query_lower and keyword not in expansions:
                    expansions.append(keyword)
                    if len(expansions) >= max_expansions:
                        break

        if len(expansions) >= max_expansions:
            break

    if expansions:
        return f"{query} {' '.join(expansions)}"

    return query


# ================= Context Window Functions =================

def _add_context_windows(
    retrieved_indices: List[int],
    total_chunks: int,
    window_size: int = 1
) -> List[int]:
    """
    Expand retrieved chunk indices to include surrounding context.

    This improves answer quality by including N chunks before and after
    each retrieved chunk, maintaining document continuity.

    The function:
    1. Adds window_size chunks before and after each retrieved index
    2. Removes duplicates (when windows overlap)
    3. Sorts by document order (not relevance order)

    Args:
        retrieved_indices: List of retrieved chunk indices (by relevance)
        total_chunks: Total number of chunks in the document
        window_size: Number of chunks to include before/after each hit

    Returns:
        Expanded list of indices in document order (deduplicated)

    Example:
        >>> _add_context_windows([5, 10], total_chunks=20, window_size=1)
        [4, 5, 6, 9, 10, 11]
    """
    if window_size <= 0:
        # No context windows - return original indices in document order
        return sorted(set(retrieved_indices))

    expanded: Set[int] = set()

    for idx in retrieved_indices:
        # Add chunks before
        for offset in range(-window_size, window_size + 1):
            neighbor_idx = idx + offset
            # Bounds checking
            if 0 <= neighbor_idx < total_chunks:
                expanded.add(neighbor_idx)

    # REPRODUCIBILITY: Sort by document order for deterministic output
    return sorted(expanded)


# ================= Core Retrieval Functions (Backwards Compatible) =================

def retrieve_top_k_tfidf(
    query: str,
    chunks: List[str],
    top_k: int = 6
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k chunks using TF-IDF cosine similarity.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending
    """
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    chunk_tokens = [_tokenize(ch) for ch in chunks]

    # Build vocabulary
    vocab = set()
    for tokens in chunk_tokens + [query_tokens]:
        vocab.update(tokens)
    vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Document frequency
    df = Counter()
    for tokens in chunk_tokens:
        for word in set(tokens):
            df[word] += 1

    N = len(chunks)

    def tfidf_vector(tokens: List[str]) -> List[float]:
        vec = [0.0] * len(vocab)
        tf = Counter(tokens)
        for word, count in tf.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                idf = math.log((N + 1) / (df.get(word, 0) + 1)) + 1
                vec[idx] = count * idf
        return vec

    query_vec = tfidf_vector(query_tokens)
    chunk_vecs = [tfidf_vector(tokens) for tokens in chunk_tokens]

    # Cosine similarity
    def cosine(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    scores = [(i, cosine(query_vec, ch_vec)) for i, ch_vec in enumerate(chunk_vecs)]
    # REPRODUCIBILITY: Stable sort by score (descending) then chunk_index (ascending)
    # This ensures deterministic ordering when scores are tied
    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores[:top_k]


def retrieve_top_k_bm25(
    query: str,
    chunks: List[str],
    top_k: int = 6
) -> List[Tuple[int, float]]:
    """
    Retrieve top-k chunks using BM25 (if rank_bm25 is installed).

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

        tokenized_chunks = [_tokenize(ch) for ch in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        # REPRODUCIBILITY: Stable sort by score (descending) then chunk_index (ascending)
        # This ensures deterministic ordering when scores are tied
        indexed_scores.sort(key=lambda x: (-x[1], x[0]))

        return indexed_scores[:top_k]
    except ImportError:
        # Fallback to TF-IDF (always available, no external dependencies)
        return retrieve_top_k_tfidf(query, chunks, top_k)


def retrieve_chunks(
    query: str,
    chunks: List[str],
    top_k: int = 6,
    method: str = "bm25"
) -> List[str]:
    """
    Retrieve top-k relevant chunks for a query.

    This is the original retrieval function, preserved for backwards compatibility.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve
        method: 'bm25' (tries BM25, falls back to TF-IDF) or 'tfidf'

    Returns:
        List of retrieved chunks (in relevance order)
    """
    if not chunks:
        return []

    if method == "bm25":
        results = retrieve_top_k_bm25(query, chunks, top_k)
    else:
        results = retrieve_top_k_tfidf(query, chunks, top_k)

    # Return chunks in relevance order
    return [chunks[idx] for idx, _ in results]


# ================= Enhanced Retrieval Functions =================

def _retrieve_top_k_tfidf_enhanced(
    query: str,
    chunks: List[str],
    top_k: int = 6,
    remove_stopwords: bool = True,
    apply_stemming: bool = True
) -> List[Tuple[int, float]]:
    """
    Enhanced TF-IDF retrieval with better tokenization.

    Uses enhanced tokenization for improved retrieval quality while
    maintaining the same TF-IDF algorithm.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve
        remove_stopwords: Whether to remove English stop words
        apply_stemming: Whether to apply basic stemming

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending
    """
    if not chunks:
        return []

    # Use enhanced tokenization
    query_tokens = _tokenize_enhanced(query, remove_stopwords, apply_stemming)
    chunk_tokens = [_tokenize_enhanced(ch, remove_stopwords, apply_stemming) for ch in chunks]

    # Handle edge case: query has no tokens after processing
    if not query_tokens:
        # Fall back to basic tokenization
        query_tokens = _tokenize(query)
        chunk_tokens = [_tokenize(ch) for ch in chunks]

    # Build vocabulary
    vocab = set()
    for tokens in chunk_tokens + [query_tokens]:
        vocab.update(tokens)
    vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    # Document frequency
    df = Counter()
    for tokens in chunk_tokens:
        for word in set(tokens):
            df[word] += 1

    N = len(chunks)

    def tfidf_vector(tokens: List[str]) -> List[float]:
        vec = [0.0] * len(vocab)
        tf = Counter(tokens)
        for word, count in tf.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                idf = math.log((N + 1) / (df.get(word, 0) + 1)) + 1
                vec[idx] = count * idf
        return vec

    query_vec = tfidf_vector(query_tokens)
    chunk_vecs = [tfidf_vector(tokens) for tokens in chunk_tokens]

    # Cosine similarity
    def cosine(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    scores = [(i, cosine(query_vec, ch_vec)) for i, ch_vec in enumerate(chunk_vecs)]

    # REPRODUCIBILITY: Stable sort by score (descending) then chunk_index (ascending)
    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores[:top_k]


def _retrieve_top_k_bm25_enhanced(
    query: str,
    chunks: List[str],
    top_k: int = 6,
    remove_stopwords: bool = True,
    apply_stemming: bool = True
) -> List[Tuple[int, float]]:
    """
    Enhanced BM25 retrieval with better tokenization.

    Uses enhanced tokenization for improved retrieval quality while
    maintaining the BM25 algorithm.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of chunks to retrieve
        remove_stopwords: Whether to remove English stop words
        apply_stemming: Whether to apply basic stemming

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]

        # Use enhanced tokenization
        tokenized_chunks = [_tokenize_enhanced(ch, remove_stopwords, apply_stemming) for ch in chunks]
        query_tokens = _tokenize_enhanced(query, remove_stopwords, apply_stemming)

        # Handle edge case: query has no tokens after processing
        if not query_tokens:
            query_tokens = _tokenize(query)
            tokenized_chunks = [_tokenize(ch) for ch in chunks]

        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query_tokens)

        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        # REPRODUCIBILITY: Stable sort by score (descending) then chunk_index (ascending)
        indexed_scores.sort(key=lambda x: (-x[1], x[0]))

        return indexed_scores[:top_k]
    except ImportError:
        # Fallback to enhanced TF-IDF
        return _retrieve_top_k_tfidf_enhanced(
            query, chunks, top_k, remove_stopwords, apply_stemming
        )


def retrieve_chunks_enhanced(
    query: str,
    chunks: List[str],
    top_k: int = 6,
    method: str = "bm25",
    # Query expansion options
    enable_expansion: bool = False,
    expansion_dict: Optional[Dict[str, List[str]]] = None,
    max_expansions: int = 6,
    # Tokenization options
    remove_stopwords: bool = True,
    apply_stemming: bool = True,
    # Context window options
    window_size: int = 0,
    # Quality/debug options
    min_score: float = 0.0,
    return_scores: bool = False,
    enable_logging: bool = False
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Enhanced retrieval with query expansion, context windows, and quality metrics.

    This is the main enhanced retrieval function that combines all improvements:

    1. QUERY EXPANSION: Automatically expands symptom queries with related keywords
       - "no steam" -> "no steam valve blocked pressure wand"
       - Configurable via expansion_dict parameter

    2. BETTER TOKENIZATION: Stop word removal and stemming for improved matching
       - "The heating element is blocked" -> ["heat", "element", "block"]
       - Both features optional via remove_stopwords and apply_stemming

    3. CONTEXT WINDOWS: Include surrounding chunks for continuity
       - window_size=1 includes 1 chunk before and after each retrieved chunk
       - Duplicates removed, results in document order

    4. QUALITY METRICS: Scores, thresholds, and debugging
       - Filter low-quality matches with min_score
       - Return scores for analysis with return_scores=True
       - Debug output with enable_logging=True

    Args:
        query: Search query
        chunks: List of text chunks to search
        top_k: Number of chunks to retrieve (before context expansion)
        method: 'bm25' (default) or 'tfidf'

        enable_expansion: Whether to expand query with related keywords
        expansion_dict: Custom symptom-to-keyword dictionary
        max_expansions: Maximum keywords to add during expansion

        remove_stopwords: Remove common English stop words
        apply_stemming: Apply basic suffix-stripping stemming

        window_size: Number of chunks before/after each hit to include (0=disabled)

        min_score: Minimum score threshold (0.0 to disable filtering)
        return_scores: Return (chunk, score) tuples instead of just chunks
        enable_logging: Print debug information about retrieval

    Returns:
        If return_scores=False: List of retrieved chunks
        If return_scores=True: List of (chunk, score) tuples

    Example:
        # Basic enhanced retrieval
        chunks = retrieve_chunks_enhanced(
            query="no steam",
            chunks=manual_chunks,
            top_k=5,
            enable_expansion=True
        )

        # Full debugging mode
        results = retrieve_chunks_enhanced(
            query="thermal fuse",
            chunks=manual_chunks,
            top_k=10,
            min_score=0.1,
            return_scores=True,
            enable_logging=True
        )
    """
    if not chunks:
        return [] if not return_scores else []

    # Step 1: Query expansion (optional)
    processed_query = query
    if enable_expansion:
        processed_query = expand_diagnostic_query(query, expansion_dict, max_expansions)
        if enable_logging and processed_query != query:
            print(f"[RETRIEVAL] Query expanded: '{query}' -> '{processed_query}'")

    # Step 2: Retrieve with enhanced tokenization
    if method == "bm25":
        results = _retrieve_top_k_bm25_enhanced(
            processed_query, chunks, top_k, remove_stopwords, apply_stemming
        )
    else:
        results = _retrieve_top_k_tfidf_enhanced(
            processed_query, chunks, top_k, remove_stopwords, apply_stemming
        )

    # Step 3: Apply minimum score threshold
    if min_score > 0.0:
        original_count = len(results)
        results = [(idx, score) for idx, score in results if score >= min_score]
        if enable_logging and len(results) < original_count:
            print(f"[RETRIEVAL] Filtered {original_count - len(results)} chunks below min_score={min_score}")

    # Step 4: Debug logging
    if enable_logging:
        print(f"[RETRIEVAL] Method: {method}, Top-K: {top_k}, Results: {len(results)}")
        if results:
            print(f"[RETRIEVAL] Score range: {results[-1][1]:.4f} - {results[0][1]:.4f}")
            for idx, score in results[:3]:  # Show top 3
                preview = chunks[idx][:60].replace('\n', ' ')
                print(f"[RETRIEVAL]   {score:.4f}: {preview}...")

    # Step 5: Apply context windows
    if window_size > 0 and results:
        retrieved_indices = [idx for idx, _ in results]
        expanded_indices = _add_context_windows(retrieved_indices, len(chunks), window_size)

        if enable_logging:
            print(f"[RETRIEVAL] Context window expanded {len(retrieved_indices)} -> {len(expanded_indices)} chunks")

        # Build score map for expanded chunks
        # Original chunks keep their scores, context chunks get score of their nearest neighbor
        score_map: Dict[int, float] = {}
        for idx, score in results:
            score_map[idx] = score

        # Assign scores to context chunks (use max score of adjacent retrieved chunks)
        for idx in expanded_indices:
            if idx not in score_map:
                # Find nearest retrieved chunk's score
                nearest_score = 0.0
                for orig_idx, score in results:
                    if abs(idx - orig_idx) <= window_size:
                        nearest_score = max(nearest_score, score)
                score_map[idx] = nearest_score

        # Return in document order with scores
        if return_scores:
            return [(chunks[idx], score_map[idx]) for idx in expanded_indices]
        else:
            return [chunks[idx] for idx in expanded_indices]

    # No context windows - return in relevance order
    if return_scores:
        return [(chunks[idx], score) for idx, score in results]
    else:
        return [chunks[idx] for idx, _ in results]


# ================= Utility Functions =================

def get_expansion_keywords(symptom: str) -> List[str]:
    """
    Get expansion keywords for a given symptom.

    Utility function to inspect what keywords would be added for a symptom.
    Useful for debugging and extending the expansion dictionary.

    Args:
        symptom: Symptom string to look up

    Returns:
        List of expansion keywords (empty if no match)

    Example:
        >>> get_expansion_keywords("no steam")
        ['valve', 'blocked', 'pressure', 'wand', 'boiler', 'thermostat']
    """
    symptom_lower = symptom.lower()
    for key, keywords in DEFAULT_SYMPTOM_EXPANSION.items():
        if key in symptom_lower:
            return keywords
    return []


def add_custom_expansion(symptom: str, keywords: List[str]) -> None:
    """
    Add a custom symptom-to-keyword mapping to the expansion dictionary.

    Modifies the global DEFAULT_SYMPTOM_EXPANSION dictionary.
    Use this to extend the expansion dictionary for domain-specific terms.

    Args:
        symptom: Symptom phrase (will be lowercased)
        keywords: List of related keywords

    Example:
        >>> add_custom_expansion("coffee grounds in cup", ["filter", "basket", "seal", "portafilter"])
    """
    DEFAULT_SYMPTOM_EXPANSION[symptom.lower()] = keywords


def analyze_retrieval_quality(
    query: str,
    chunks: List[str],
    top_k: int = 10,
    method: str = "bm25"
) -> Dict[str, any]:
    """
    Analyze retrieval quality for debugging purposes.

    Returns detailed statistics about retrieval results including
    score distribution and token analysis.

    Args:
        query: Search query
        chunks: List of text chunks
        top_k: Number of results to analyze
        method: 'bm25' or 'tfidf'

    Returns:
        Dictionary with analysis results:
        - query_tokens: Tokenized query
        - results: List of (index, score, preview) tuples
        - score_stats: min, max, mean, std of scores
        - zero_score_count: Number of chunks with zero score
    """
    if not chunks:
        return {"error": "No chunks provided"}

    # Get all scores (not just top-k)
    if method == "bm25":
        all_results = retrieve_top_k_bm25(query, chunks, len(chunks))
    else:
        all_results = retrieve_top_k_tfidf(query, chunks, len(chunks))

    # Extract scores
    scores = [score for _, score in all_results]

    # Calculate statistics
    score_stats = {
        "min": min(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5 if scores else 0.0
    }

    # Top results with previews
    top_results = [
        {
            "index": idx,
            "score": score,
            "preview": chunks[idx][:100].replace('\n', ' ')
        }
        for idx, score in all_results[:top_k]
    ]

    return {
        "query_tokens": _tokenize(query),
        "query_tokens_enhanced": _tokenize_enhanced(query),
        "total_chunks": len(chunks),
        "results": top_results,
        "score_stats": score_stats,
        "zero_score_count": sum(1 for s in scores if s == 0.0),
        "above_threshold_count": {
            "0.1": sum(1 for s in scores if s >= 0.1),
            "0.2": sum(1 for s in scores if s >= 0.2),
            "0.3": sum(1 for s in scores if s >= 0.3),
        }
    }
