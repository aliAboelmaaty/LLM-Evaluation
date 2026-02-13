"""
Metrics Module

Comprehensive set of metrics for evaluating LLM outputs across different domains.

Metrics Categories:
1. RAG-specific: citation_coverage, citation_correctness, context_precision/recall, faithfulness
2. Quality: answer_correctness (with ground truth)
3. Readability: fkgl (Flesch-Kincaid Grade Level)
4. Hallucination: hallucination_rate
5. Domain-specific: scenario_identification_rate, property_identification_rate
6. Consistency: output_consistency
7. OCR quality: cer, wer
"""

from typing import List, Dict, Any, Optional, Set
import re
import json


# ================= Text Processing Utilities =================

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, filtering out structural elements.

    Handles:
    - Standard sentence boundaries (.!?)
    - Citations after periods: "...text. [1, 2]" or "...text). [1]"
    - Paragraph breaks (double newlines) as sentence boundaries
    - Structured text with labels like "Problem:", "Solution:"
    - Bold labels like "**Problem:** content here" - extracts the content

    Excludes:
    - Markdown headers (lines starting with # for headings)
    - Pure blockquotes (lines starting with > )
    - Very short segments (< 10 chars)
    - Lines that are just labels (ending with :)
    """
    text = (text or "").strip()
    if not text:
        return []

    # Step 1: Split on paragraph breaks first (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)

    all_sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Step 2: Split on sentence boundaries using findall to KEEP citations
        # Match: content ending with .!? optionally followed by ) and citations
        # This regex captures complete sentences INCLUDING their trailing citations
        sentence_pattern = r'[^.!?]*[.!?](?:\)?(?:\s*\[\d+(?:[,\s]*\d+)*\])*)?'
        matches = re.findall(sentence_pattern, para)

        # If no matches (e.g., paragraph without sentence-ending punctuation), use whole paragraph
        if not matches:
            matches = [para]

        for p in matches:
            p = p.strip()
            if not p:
                continue

            # Skip markdown heading lines (e.g., "## Title", "### Section")
            if p.startswith('#'):
                continue

            # Skip pure blockquotes
            if p.startswith('> '):
                continue

            # Handle bold labels like "**Problem:** content here [1]"
            # Extract the content AFTER the label instead of discarding
            if p.startswith('**'):
                # Pattern: **Label:** content OR **Label** content
                bold_label_match = re.match(r'\*\*[^*]+\*\*:?\s*', p)
                if bold_label_match:
                    # Extract content after the bold label
                    content_after_label = p[bold_label_match.end():].strip()
                    if content_after_label and len(content_after_label) >= 10:
                        p = content_after_label
                    else:
                        # Pure header with no content, skip it
                        continue
                else:
                    # Malformed bold, skip
                    continue

            # Skip very short segments (likely labels or formatting)
            if len(p) < 10:
                continue

            # Skip lines that are ONLY a label (like "Problem:" alone)
            # But keep lines that have content after the label
            if p.endswith(':') and len(p.split()) <= 4:
                continue

            all_sentences.append(p)

    return all_sentences


def _tokens(text: str) -> List[str]:
    """Extract tokens from text"""
    return re.findall(r"[A-Za-z0-9\u0027]+", (text or "").lower())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cosine_tfidf(a: str, b: str) -> Optional[float]:
    """Calculate TF-IDF cosine similarity between two texts"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return None

    if not a.strip() or not b.strip():
        return 0.0

    vect = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vect.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


# ================= Citation Standards =================

# Standard instruction to inject into all LLM prompts
# This ensures consistent citation format across all models
CITATION_STANDARD_INSTRUCTION = """
CRITICAL REQUIREMENTS:

LANGUAGE: You MUST respond in English only. Even if the source documents are in German, Spanish, French, or any other language, your response MUST be written entirely in English. Translate any relevant information into English.

CITATIONS:
- Every statement derived from the provided context MUST include a citation
- Use ONLY this format: [1], [2], [3], etc.
- Do NOT use: (p. 5), (Source 1), (Manual), or any other format
- Place citations at the end of the sentence or claim
- Multiple sources: [1, 2] or [1][2]

Example:
"The pump is blocked due to debris accumulation [1]. This requires disassembly [2]."
"""


def get_citation_instruction() -> str:
    """
    Returns the standard citation instruction to inject into prompts.

    Usage in your experiment:
        prompt = f"{question}\\n\\n{get_citation_instruction()}"
    """
    return CITATION_STANDARD_INSTRUCTION


# ================= Citation & Grounding Metrics =================

# STRICT CITATION PATTERN
# Only matches: [1], [1,2], [1, 2, 3] - the ONLY format we instruct models to use
# Excludes: (p. 5), [Source 1], [[1]] - these formats are NOT requested in prompts
# Uses negative lookbehind/lookahead to exclude double brackets [[1]]
_CITATION_PAT = re.compile(r'(?<!\[)\[\s*\d+(?:\s*,\s*\d+)*\s*\](?!\])')


# ================= Citation Normalization =================

def normalize_citations(text: str) -> str:
    """
    Normalize various citation formats to standard [1] format.

    This pre-processing step ensures legacy outputs or non-compliant
    models can still be evaluated fairly.

    Conversions:
    - (p. 5) -> [5]
    - (Page 12) -> [12]
    - [Source 1] -> [1]
    - [Doc 2] -> [2]
    - [[3]] -> [3]

    Args:
        text: Raw LLM output

    Returns:
        Normalized text with standard [1] format
    """
    if text is None:
        return ""

    # Convert (p. 5) or (Page 12) -> [5], [12]
    text = re.sub(
        r'\(\s*(?:p\.|pp\.|page|pages)\s*(\d+)\s*\)',
        r'[\1]',
        text,
        flags=re.IGNORECASE
    )

    # Convert (Section 3) or (Sec. 4) -> [3], [4]
    text = re.sub(
        r'\(\s*(?:sec\.|section)\s*(\d+)\s*\)',
        r'[\1]',
        text,
        flags=re.IGNORECASE
    )

    # Convert (Source 1) or (Ref 1) -> [1]
    text = re.sub(
        r'\(\s*(?:source|ref|reference|fig\.|figure)\s*(\d+)\s*\)',
        r'[\1]',
        text,
        flags=re.IGNORECASE
    )

    # Convert [Source 1] or [Doc 2] or [Ref 3] -> [1], [2], [3]
    text = re.sub(
        r'\[\s*(?:source|doc|document|ref|reference)\s*(\d+)\s*\]',
        r'[\1]',
        text,
        flags=re.IGNORECASE
    )

    # Convert [[1]] -> [1]
    text = re.sub(r'\[\[\s*(\d+)\s*\]\]', r'[\1]', text)

    return text


def _best_overlap_score(span: str, contexts: List[str]) -> float:
    """Calculate best overlap score between span and any context"""
    span_tokens = _tokens(span)
    if not span_tokens:
        return 0.0

    best = 0.0
    for c in contexts:
        ctx_tokens = _tokens(c)
        if not ctx_tokens:
            continue

        span_set, ctx_set = set(span_tokens), set(ctx_tokens)
        inter = len(span_set & ctx_set)
        prec = inter / len(span_set) if span_set else 0.0
        rec = inter / len(ctx_set) if ctx_set else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        # Character-level overlap
        char_overlap = _jaccard(
            set(re.findall(r"..", " ".join(span_tokens))),
            set(re.findall(r"..", " ".join(ctx_tokens))),
        )

        score = 0.7 * f1 + 0.3 * char_overlap
        best = max(best, score)

    return best


def citation_coverage(
    answer: str,
    contexts: Optional[List[str]] = None,
    overlap_threshold: float = 0.18,
) -> float:
    """
    Fraction of sentences with EXPLICIT citations.

    Uses 3-layer standardization:
    1. Normalizes citations to [1] format (catches legacy formats)
    2. Detects using universal pattern (catches all valid structures)
    3. Maintains strict discipline (no implicit grounding fallback)

    Args:
        answer: LLM answer text
        contexts: List of context documents (no longer used for implicit scoring)
        overlap_threshold: Deprecated parameter (kept for backward compatibility)

    Returns:
        Coverage score between 0.0 and 1.0
    """
    # Layer 3: Normalize first (handles legacy/non-compliant formats)
    answer = normalize_citations(answer)

    sents = _split_sentences(answer)
    if not sents:
        return 0.0

    # Count only explicit citations (Layer 1 pattern catches all valid formats)
    explicit = [s for s in sents if _CITATION_PAT.search(s)]
    return len(explicit) / len(sents)


def _citation_correctness_with_seed(
    answer: str,
    contexts: List[str],
    threshold: float = 0.18,
    sample_k: int = 10,
    seed: int = 42,
) -> float:
    """
    Check if EXPLICITLY cited sentences are actually grounded in contexts.

    Uses 3-layer standardization to ensure fair evaluation across all
    citation formats.

    Args:
        answer: LLM answer text
        contexts: List of context documents
        threshold: Minimum overlap to consider grounded
        sample_k: Maximum sentences to sample for efficiency
        seed: Random seed for sampling (for reproducibility)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    # Layer 3: Normalize first (handles legacy/non-compliant formats)
    answer = normalize_citations(answer)

    # Extract only sentences with explicit citations (Layer 1 pattern)
    sents = [s for s in _split_sentences(answer) if _CITATION_PAT.search(s)]

    if not contexts:
        return 0.0

    # If no citations found, return 0.0 (no implicit fallback)
    if not sents:
        return 0.0

    # Sample sentences if too many (use provided seed for reproducibility)
    import random
    rnd = random.Random(seed)
    sample = sents if len(sents) <= sample_k else rnd.sample(sents, sample_k)

    # Check how many cited sentences are grounded in contexts
    ok = 0
    for s in sample:
        if _best_overlap_score(s, contexts) >= threshold:
            ok += 1

    return ok / len(sample) if sample else 0.0




def faithfulness(
    answer: str,
    contexts: List[str],
    seed: int = 42,
    embedder: Optional[Any] = None
) -> float:
    """
    Semantic grounding score. Uses Embeddings if available, falls back to Overlap.

    Measures how well each sentence in the answer is grounded in the contexts.
    Uses semantic similarity (cosine) when embedder is provided, otherwise
    falls back to token overlap.

    Args:
        answer: LLM answer text
        contexts: List of context documents
        seed: Random seed for reproducibility (currently unused but standardized)
        embedder: Optional sentence transformer model for semantic similarity

    Returns:
        Faithfulness score between 0.0 and 1.0
    """
    sents = _split_sentences(answer)
    if not sents or not contexts:
       return 0.0

    # MODE A: Smart Semantic Check (If embedder is passed)
    if embedder is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # 1. Encode
            sent_embeddings = embedder.encode(sents)
            ctx_embeddings = embedder.encode(contexts)

            # 2. Similarity Matrix (Sentence vs Context)
            sim_matrix = cosine_similarity(sent_embeddings, ctx_embeddings)

            # 3. Best match for each sentence
            max_scores = sim_matrix.max(axis=1)

            # 4. Average
            return float(max_scores.mean())
        except Exception:
            pass  # Fallback if anything fails

    # MODE B: Strict Overlap Check (Fallback)
    scores = [_best_overlap_score(s, contexts) for s in sents]
    return float(sum(scores) / len(scores))


# ================= Context Quality Metrics =================

def _is_context_used(unit_text: str, answer: str, q: str) -> float:
    """Check if a context unit is used in the answer"""
    return _best_overlap_score(unit_text, [answer + "\n" + q])


def context_precision(
    contexts: List[str],
    question: str,
    answer: str,
    use_threshold: float = 0.18,
) -> float:
    """
    Precision of retrieved context.

    What fraction of retrieved contexts are actually used?

    Args:
        contexts: Retrieved context documents
        question: Original question
        answer: LLM answer
        use_threshold: Minimum overlap to consider "used"

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not contexts:
        return 0.0

    used_flags = [
        1 if _is_context_used(c, answer, question) >= use_threshold else 0
        for c in contexts
    ]
    precision = sum(used_flags) / len(contexts) if contexts else 0.0
    return precision


def context_recall(
    contexts: List[str],
    question: str,
    answer: str,
    use_threshold: float = 0.18,
) -> float:
    """
    Recall of retrieved context.

    How much of the answer is covered by the retrieved contexts?

    Args:
        contexts: Retrieved context documents
        question: Original question
        answer: LLM answer
        use_threshold: Minimum overlap to consider "used"

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not contexts:
        return 0.0

    used_flags = [
        1 if _is_context_used(c, answer, question) >= use_threshold else 0
        for c in contexts
    ]

    used_text = " ".join([c for c, f in zip(contexts, used_flags) if f])
    all_text = " ".join(contexts)

    cover_used = _best_overlap_score(answer, [used_text]) if used_text else 0.0
    cover_all = _best_overlap_score(answer, [all_text]) or 1e-9

    recall = min(1.0, cover_used / cover_all) if cover_all > 0 else 0.0
    return recall


# ================= Answer Quality Metrics =================

def _extract_diagnosis_section(answer: str) -> str:
    """
    Extract the diagnosis section from a structured LLM answer.

    Looks for patterns like:
    - **Diagnosis:** <text>
    - Diagnosis: <text>
    - ## Diagnosis

    Args:
        answer: LLM answer text

    Returns:
        Extracted diagnosis text (or empty string if not found)
    """
    lines = answer.split('\n')
    diagnosis_text = []
    in_diagnosis = False

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line starts a diagnosis section
        if 'diagnosis' in line_lower and ':' in line:
            # Extract text after "Diagnosis:"
            parts = line.split(':', 1)
            if len(parts) > 1:
                diagnosis_text.append(parts[1].strip())
            in_diagnosis = True
            continue

        # Check for markdown heading
        if line_lower.startswith('#') and 'diagnosis' in line_lower:
            in_diagnosis = True
            continue

        # Stop at next section heading
        if in_diagnosis and (line.startswith('#') or (line.endswith(':') and len(line.split()) <= 4)):
            break

        # Collect diagnosis text
        if in_diagnosis and line.strip():
            diagnosis_text.append(line.strip())

    return ' '.join(diagnosis_text)


def _normalize_diagnosis(text: str) -> str:
    """
    Normalize diagnosis text for comparison.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace

    Args:
        text: Diagnosis text

    Returns:
        Normalized text
    """
    import string
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def _answer_correctness_with_embedder(
    llm_answer: str,
    ground_truth: Dict[str, Any],
    embedder: Optional[Any] = None
) -> float:
    """
    DIAGNOSIS-GROUNDED METRIC - Compare extracted diagnosis with ground truth.

    This metric specifically evaluates diagnostic capability by:
    1. Extracting the "Diagnosis" section from the LLM answer
    2. Comparing only that section to ground_truth["diagnosis"]
    3. Using exact/normalized match as primary score
    4. Falling back to semantic similarity for partial credit

    For diagnosis tasks:
        ground_truth = {"diagnosis": "Pump blocked", "root_cause": "Foreign object"}
        Extracts "Diagnosis:" section from llm_answer, compares with "Pump blocked"

    For other tasks (repurposing, ML recommendation):
        Falls back to keyword matching across all ground truth fields

    Args:
        llm_answer: LLM's answer
        ground_truth: Dictionary with expected outputs (must include "diagnosis" key for diagnosis tasks)
        embedder: Pre-loaded SentenceTransformer model (optional, improves performance)

    Returns:
        Correctness score between 0.0 and 1.0
    """
    if not llm_answer or not ground_truth:
        return 0.0

    # Diagnosis task: extract and compare diagnosis section
    if "diagnosis" in ground_truth:
        gt_diagnosis = str(ground_truth["diagnosis"]).strip()
        if not gt_diagnosis:
            return 0.0

        # Extract diagnosis from answer
        extracted_diagnosis = _extract_diagnosis_section(llm_answer)

        # If extraction failed, fall back to searching whole answer
        if not extracted_diagnosis:
            extracted_diagnosis = llm_answer

        # Normalize both for comparison
        norm_extracted = _normalize_diagnosis(extracted_diagnosis)
        norm_gt = _normalize_diagnosis(gt_diagnosis)

        # 1. Exact match after normalization
        if norm_extracted == norm_gt:
            return 1.0

        # 2. Check if ground truth is substring of extracted (partial match)
        if norm_gt in norm_extracted:
            return 0.9

        # 3. Token overlap (precision-focused)
        extracted_tokens = set(_tokens(extracted_diagnosis))
        gt_tokens = set(_tokens(gt_diagnosis))

        if not gt_tokens:
            return 0.0

        # Precision: what fraction of extracted tokens are in ground truth?
        # Recall: what fraction of ground truth tokens are in extracted?
        precision = len(extracted_tokens & gt_tokens) / len(extracted_tokens) if extracted_tokens else 0.0
        recall = len(extracted_tokens & gt_tokens) / len(gt_tokens) if gt_tokens else 0.0

        # F1 score (balanced precision and recall)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 4. Semantic similarity as fallback (use shared embedder if available)
        if embedder is not None:
            try:
                embeddings = embedder.encode([extracted_diagnosis, gt_diagnosis])
                from sklearn.metrics.pairwise import cosine_similarity
                semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

                # Weighted combination: 60% token overlap, 40% semantic similarity
                return 0.3 * f1 + 0.7 * semantic_sim
            except Exception:
                # Fall back to token overlap if embedding fails
                return f1
        else:
            # No embedder available, use token overlap only
            return f1

    # Non-diagnosis tasks: keyword matching across all ground truth fields
    gt_keywords = set()
    for v in ground_truth.values():
        if isinstance(v, str):
            gt_keywords.update(_tokens(v))
        elif isinstance(v, list):
            for item in v:
                gt_keywords.update(_tokens(str(item)))

    answer_tokens = set(_tokens(llm_answer))

    if not gt_keywords:
        return 0.0

    # Jaccard similarity
    return _jaccard(answer_tokens, gt_keywords)

# ================= Answer Correctness with Weights =================

def answer_correctness(
    answer: str,
    ground_truth: str,
    embedder=None,
    weight_f1: float = 0.50,
    weight_semantic: float = 0.50
) -> float:
    """
    Weighted combination of F1 score and semantic similarity.

    Equal 50/50 weighting balances:
    - F1 Score (50%): Token-level precision critical for component identification
      in technical diagnosis (e.g., "thermal fuse" vs "heating element" are different)
    - Semantic Similarity (50%): Captures conceptual correctness for valid paraphrases
      (e.g., "pump blocked" vs "pump obstruction" are equivalent)

    This weighting is appropriate for safety-critical domains where both
    precise terminology and semantic understanding are equally important.

    Adapted from RAGAS framework (Es et al., 2023) with equal weighting
    for balanced evaluation in technical diagnosis tasks.

    Args:
        answer: LLM-generated answer text
        ground_truth: Expected answer text (string form)
        embedder: Optional sentence transformer for semantic similarity
        weight_f1: Weight for F1 score (default: 0.50)
        weight_semantic: Weight for semantic similarity (default: 0.50)

    Returns:
        Weighted correctness score between 0.0 and 1.0
    """
    if not answer or not ground_truth:
        return 0.0

    # Ensure ground_truth is a string
    if isinstance(ground_truth, dict):
        # Extract diagnosis field if it's a dict
        ground_truth = str(ground_truth.get("diagnosis", ground_truth))
    ground_truth = str(ground_truth)

    # Calculate F1 score (token-level)
    answer_tokens = set(_tokens(answer))
    gt_tokens = set(_tokens(ground_truth))

    if not gt_tokens:
        return 0.0

    precision = len(answer_tokens & gt_tokens) / len(answer_tokens) if answer_tokens else 0.0
    recall = len(answer_tokens & gt_tokens) / len(gt_tokens) if gt_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate semantic similarity
    semantic_sim = 0.0
    if embedder is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            embeddings = embedder.encode([answer, ground_truth])
            semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        except Exception:
            # Fallback to TF-IDF if embedding fails
            tfidf_sim = _cosine_tfidf(answer, ground_truth)
            semantic_sim = tfidf_sim if tfidf_sim is not None else 0.0
    else:
        # No embedder: use TF-IDF similarity
        tfidf_sim = _cosine_tfidf(answer, ground_truth)
        semantic_sim = tfidf_sim if tfidf_sim is not None else 0.0

    # Weighted combination
    return weight_f1 * f1 + weight_semantic * semantic_sim


# ================= Domain-Specific Metrics =================

def scenario_identification_rate(
    llm_scenarios: List[str],
    ground_truth_scenarios: List[str]
) -> float:
    """
    NEW - Dörnbach's SI metric.

    What fraction of documented scenarios did the LLM identify?

    SI = len(identified_scenarios) / len(ground_truth_scenarios)

    Args:
        llm_scenarios: Scenarios identified by LLM
        ground_truth_scenarios: Expected scenarios from documentation

    Returns:
        SI score between 0.0 and 1.0
    """
    if not ground_truth_scenarios:
        return 1.0 if not llm_scenarios else 0.0

    # Normalize both lists to lowercase tokens
    gt_set = {tuple(_tokens(s)) for s in ground_truth_scenarios}
    llm_set = {tuple(_tokens(s)) for s in llm_scenarios}

    # Count matches
    matches = 0
    for gt_scenario in gt_set:
        for llm_scenario in llm_set:
            # Check if there's significant overlap
            if _jaccard(set(gt_scenario), set(llm_scenario)) >= 0.5:
                matches += 1
                break

    return matches / len(ground_truth_scenarios)


def property_identification_rate(
    llm_properties: List[str],
    ground_truth_properties: List[str]
) -> float:
    """
    NEW - Dörnbach's PI metric.

    What fraction of required properties did the LLM identify?

    PI = len(identified_properties) / len(ground_truth_properties)

    Args:
        llm_properties: Properties identified by LLM
        ground_truth_properties: Expected properties from documentation

    Returns:
        PI score between 0.0 and 1.0
    """
    if not ground_truth_properties:
        return 1.0 if not llm_properties else 0.0

    # Similar to SI, but for properties
    gt_set = {tuple(_tokens(p)) for p in ground_truth_properties}
    llm_set = {tuple(_tokens(p)) for p in llm_properties}

    matches = 0
    for gt_prop in gt_set:
        for llm_prop in llm_set:
            if _jaccard(set(gt_prop), set(llm_prop)) >= 0.5:
                matches += 1
                break

    return matches / len(ground_truth_properties)


def _strip_citations(text: str) -> str:
    """
    Remove all citation markers from text before consistency comparison.

    This ensures consistency is measured on CONTENT, not citation placement.
    Citations can vary across runs even when content is identical.

    Args:
        text: Text with citations like [1], [1,2], [1, 2, 3]

    Returns:
        Text with all citations removed

    Example:
        >>> _strip_citations("Replace pump [1, 2] following procedure [3]")
        "Replace pump  following procedure "
    """
    return re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)


def output_consistency(outputs: List[str]) -> float:
    """
    Measure consistency across multiple outputs.

    IMPORTANT: Citations are stripped before comparison to focus on
    content consistency rather than citation placement variability.

    Example:
        Run 1: "Replace pump [1] following procedure [2]"
        Run 2: "Replace pump [3] following procedure [4]"
        => High consistency (same content, different citation numbers)

    Computes average Jaccard similarity between all pairs of outputs.

    Args:
        outputs: List of output strings from repetitions

    Returns:
        Average Jaccard similarity across all pairs (0.0 to 1.0)
    """
    if len(outputs) < 2:
        return 1.0

    # Strip citations from all outputs FIRST
    clean_outputs = [_strip_citations(out) for out in outputs]

    # Calculate pairwise similarities on cleaned outputs
    similarities = []
    for i in range(len(clean_outputs)):
        for j in range(i + 1, len(clean_outputs)):
            tokens_a = set(_tokens(clean_outputs[i]))
            tokens_b = set(_tokens(clean_outputs[j]))
            sim = _jaccard(tokens_a, tokens_b)
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 1.0


# ================= Hallucination Detection =================

def hallucination_rate(
    answer: str,
    contexts: List[str],
    seed: int = 42,
    embedder: Optional[Any] = None,
    f_thresh: float = 0.5,
    c_thresh: float = 0.6,
) -> int:
    """
    Binary hallucination detection.
    Calculates sub-metrics internally to fit the standard metric signature.

    Returns 1 if likely hallucinated, 0 if grounded.

    The answer is considered hallucinated if either:
    - Faithfulness score is below the faithfulness threshold, OR
    - Citation correctness is below the citation threshold

    Uses semantic similarity when embedder is provided for fairer evaluation.

    Args:
        answer: LLM answer text
        contexts: List of context documents
        seed: Random seed for reproducibility (passed to citation_correctness)
        embedder: Optional sentence transformer model for semantic similarity
        f_thresh: Faithfulness threshold (default: 0.5)
        c_thresh: Citation correctness threshold (default: 0.6)

    Returns:
        1 if hallucinated, 0 if grounded

    Note:
        This metric should only be computed when contexts are available.
        For BASELINE mode (no contexts), return NaN instead.
    """
    # Hallucination check requires contexts
    if not contexts:
        return 0

    # Pass embedder to faithfulness so it uses the SMART Semantic logic
    f_score = faithfulness(answer, contexts, seed=seed, embedder=embedder)
    c_score = _citation_correctness_with_seed(answer, contexts, seed=seed)

    return 1 if (f_score < f_thresh or c_score < c_thresh) else 0


# ================= Readability Metrics =================

_VOWELS = "aeiouy"


def _count_syllables(word: str) -> int:
    """Count syllables in a word"""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0

    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v

    if w.endswith("e") and count > 1:
        count -= 1

    return max(1, count)


def fkgl(text: str) -> float:
    """
    Flesch-Kincaid Grade Level (readability).

    Higher scores = more difficult to read.

    Args:
        text: Text to analyze

    Returns:
        Grade level (e.g., 8.0 = 8th grade reading level)
    """
    sents = _split_sentences(text)
    words = _tokens(text)

    if not sents or not words:
        return 0.0

    syllables = sum(_count_syllables(w) for w in words)
    ASL = len(words) / len(sents)  # Average Sentence Length
    ASW = syllables / len(words)   # Average Syllables per Word

    return 0.39 * ASL + 11.8 * ASW - 15.59


# ================= OCR Quality Metrics =================

def _levenshtein(a: List, b: List) -> int:
    """Calculate Levenshtein distance between two sequences"""
    n, m = len(a), len(b)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur

    return dp[m]


def cer(ocr_text: str, gt_text: str) -> float:
    """
    Character Error Rate.

    CER = edit_distance(chars) / len(ground_truth_chars)

    Args:
        ocr_text: OCR-extracted text
        gt_text: Ground truth text

    Returns:
        CER score (lower is better)
    """
    a_c = list(ocr_text or "")
    b_c = list(gt_text or "")

    if not b_c:
        return 0.0 if not a_c else 1.0

    return _levenshtein(a_c, b_c) / len(b_c)


def wer(ocr_text: str, gt_text: str) -> float:
    """
    Word Error Rate.

    WER = edit_distance(words) / len(ground_truth_words)

    Args:
        ocr_text: OCR-extracted text
        gt_text: Ground truth text

    Returns:
        WER score (lower is better)
    """
    a_w = (ocr_text or "").split()
    b_w = (gt_text or "").split()

    if not b_w:
        return 0.0 if not a_w else 1.0

    return _levenshtein(a_w, b_w) / len(b_w)


# ================= ML Recommendation Metrics =================


def ml_problem_type_accuracy(
    answer: str,
    ground_truth: Dict[str, Any],
    **kwargs
) -> float:
    """
    Calculate accuracy of ML problem type identification.
    Handles multi-label cases like "Classification; Clustering".
    Corresponds to Sonntag et al. (2025) RQ1: Problem understanding.

    Args:
        answer: LLM generated answer
        ground_truth: Dictionary containing 'ml_problem_type'

    Returns:
        Accuracy score between 0 and 1
    """
    gt_types_str = ground_truth.get('ml_problem_type', '')

    # Extract problem type from answer
    type_pattern = r'(?:\*\*)?(?:ML\s+)?(?:Problem\s+)?Type(?:\*\*)?.*?:\s*([^\n]+)'
    match = re.search(type_pattern, answer, re.IGNORECASE)

    if not match:
        return 0.0

    predicted_text = match.group(1).lower()

    # Parse ground truth types (semicolon-separated)
    gt_types = set()
    for t in gt_types_str.split(';'):
        t = t.strip().lower()
        if 'classif' in t:
            gt_types.add('classification')
        elif 'regress' in t:
            gt_types.add('regression')
        elif 'clust' in t:
            gt_types.add('clustering')
        elif 'assoc' in t or 'association' in t:
            gt_types.add('association')

    # Parse predicted types
    pred_types = set()
    if 'classif' in predicted_text:
        pred_types.add('classification')
    if 'regress' in predicted_text:
        pred_types.add('regression')
    if 'clust' in predicted_text:
        pred_types.add('clustering')
    if 'assoc' in predicted_text or 'association' in predicted_text:
        pred_types.add('association')

    if not pred_types:
        return 0.0

    # Calculate accuracy
    correct = len(gt_types & pred_types)
    total_gt = len(gt_types)

    # Penalize for wrong types
    wrong = len(pred_types - gt_types)

    if wrong > 0:
        score = max(0, correct - wrong) / total_gt if total_gt > 0 else 0.0
    else:
        score = correct / total_gt if total_gt > 0 else 0.0

    return score


def algorithm_suitability(
    answer: str,
    ground_truth: Dict[str, Any],
    **kwargs
) -> float:
    """
    Evaluate if suggested algorithm is suitable for the problem.
    Equivalent to Sonntag et al. (2025) Task Fulfillment Rate (TFR).
    Corresponds to RQ2: Candidate recommendation.

    Uses multi-stage extraction to work with both structured (RAG) and
    unstructured (BASELINE) outputs:
      - Strategy A: Look for "Algorithm:" header (structured format)
      - Strategy B: Search entire answer for known algorithm names (fallback)
      - Strategy C: Pattern-based extraction ("I recommend X", "use X")

    Per Sonntag (p.6): "A task is considered fulfilled when the suggested
    ML algorithm is suitable, even if it differs from the source publication."

    Args:
        answer: LLM generated answer
        ground_truth: Dictionary containing 'ml_problem_type'

    Returns:
        1.0 if suitable algorithm suggested, 0.0 otherwise
    """
    ml_problem_type = ground_truth.get('ml_problem_type', '') if isinstance(ground_truth, dict) else ''

    # Define algorithm-to-problem-type mappings (based on Sonntag's dataset)
    # IMPORTANT: Longer names should come first to avoid partial matches
    # e.g., "random forest" before "forest", "support vector machine" before "svm"
    classification_algos = [
        # Full names first (to match before abbreviations)
        'support vector machine', 'support vector classifier', 'random forest',
        'decision tree', 'gradient boosting', 'naive bayes', 'logistic regression',
        'k-nearest neighbor', 'k nearest neighbor', 'neural network', 'deep learning',
        'multinomial naive bayes', 'convolutional neural network',
        # Common variations
        'classifier', 'classification tree', 'boosted trees', 'ensemble',
        # Then abbreviations (only if they are standalone terms)
        'xgboost', 'xgb', 'svm', 'svc', 'knn', 'cart', 'c4.5', 'c5.0',
        'mlp', 'ann', 'lstm', 'bert', 'cnn', 'rnn',
        # Short abbreviations last (risk of false matches)
        'rf', 'dt', 'nb', 'lr', 'mnb'
    ]

    regression_algos = [
        # Full names first
        'linear regression', 'support vector regression', 'polynomial regression',
        'neural network', 'random forest', 'gradient boosting', 'deep learning',
        'k-nearest neighbor', 'k nearest neighbor', 'ridge regression', 'lasso regression',
        'fuzzy neural network', 'least squares',
        # Common variations
        'regression tree', 'boosted trees', 'predictor', 'regressor',
        # Abbreviations
        'xgboost', 'lightgbm', 'svr', 'lasso', 'ridge', 'knn',
        'ann', 'mlp', 'lssvr', 'fuzzy nn', 'lda',
        # Short abbreviations last
        'pr'
    ]

    clustering_algos = [
        # Full names first
        'k-means', 'k means', 'kmeans', 'hierarchical clustering', 'spectral clustering',
        'fuzzy clustering', 'gaussian mixture', 'agglomerative clustering',
        'density-based', 'dbscan', 'hdbscan',
        # Common variations
        'cluster analysis', 'clustering algorithm', 'segmentation',
        # Abbreviations
        'gmm', 'rskc', 'hierarchical', 'spectral', 'agglomerative'
    ]

    association_algos = [
        # Full names first
        'association rule', 'association rules', 'market basket', 'frequent itemset',
        'frequent pattern', 'fuzzy association',
        # Algorithms
        'apriori', 'fp-growth', 'fp growth', 'eclat',
        # Abbreviations
        'cba', 'farm', 'arm'
    ]

    # Combine all known algorithms for fallback search
    # CRITICAL: Sort by length (longest first) to avoid short abbreviations
    # matching inside longer words (e.g., 'pr' matching inside 'Apriori')
    all_known_algos = classification_algos + regression_algos + clustering_algos + association_algos
    all_known_algos = sorted(all_known_algos, key=len, reverse=True)

    suggested_algo = None

    # STRATEGY A: Structured extraction (for RAG/formatted outputs)
    # Try to capture algorithm from "Algorithm:" or "Recommended Algorithm(s):" header
    algo_pattern = r'(?:\*\*)?(?:Recommended\s+)?Algorithm(?:s)?(?:\*\*)?.*?:\s*([^\n]+)'
    match = re.search(algo_pattern, answer, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        # Only use if we actually captured content (not just whitespace)
        if extracted and len(extracted) > 2:
            suggested_algo = extracted.lower()

    # STRATEGY B: Fallback - search entire answer for known algorithm names
    if not suggested_algo:
        answer_lower = answer.lower()
        for algo in all_known_algos:
            if algo in answer_lower:
                suggested_algo = algo
                break  # Found an algorithm, stop searching

    # STRATEGY C: Pattern-based extraction ("I recommend X", "use X", etc.)
    if not suggested_algo:
        # Look for recommendation patterns
        patterns = [
            r'(?:i\s+)?recommend(?:ing)?\s+(?:using\s+)?([a-zA-Z][a-zA-Z0-9\s\-]+?)(?:\s+(?:for|because|since|to|as|,|\.))',
            r'(?:should|could|would)\s+use\s+([a-zA-Z][a-zA-Z0-9\s\-]+?)(?:\s+(?:for|because|since|to|as|,|\.))',
            r'suggest(?:ing)?\s+([a-zA-Z][a-zA-Z0-9\s\-]+?)(?:\s+(?:for|because|since|to|as|,|\.))',
            r'apply(?:ing)?\s+([a-zA-Z][a-zA-Z0-9\s\-]+?)(?:\s+(?:for|because|since|to|as|,|\.))',
        ]
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                potential_algo = match.group(1).lower().strip()
                # Check if it contains a known algorithm
                for algo in all_known_algos:
                    if algo in potential_algo:
                        suggested_algo = algo
                        break
                if suggested_algo:
                    break

    # If still no algorithm found, return 0.0
    if not suggested_algo:
        return 0.0

    # Check if suggested algorithm matches any of the problem types
    problem_types = ml_problem_type.lower()
    suitable = False
    extraction_strategy = kwargs.get('_extraction_strategy', 'unknown')

    if 'classif' in problem_types:
        suitable = suitable or any(algo in suggested_algo for algo in classification_algos)
    if 'regress' in problem_types:
        suitable = suitable or any(algo in suggested_algo for algo in regression_algos)
    if 'clust' in problem_types:
        suitable = suitable or any(algo in suggested_algo for algo in clustering_algos)
    if 'assoc' in problem_types:
        suitable = suitable or any(algo in suggested_algo for algo in association_algos)

    return 1.0 if suitable else 0.0


# ================= Metrics Calculator =================

class MetricCalculator:
    """
    Centralized metric calculator with dynamic metric execution and automatic seed injection.

    Key features:
    - Configurable metric registry (no hardcoded methods)
    - Automatic seed injection for reproducibility
    - Shared embedding model for efficiency
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize metric calculator with embedding model and metric registry.

        Args:
            random_seed: Random seed for reproducibility (automatically injected into all metrics)
        """
        self.random_seed = random_seed

        # Instantiate embedding model once (expensive operation)
        # This ensures:
        # 1. Model is only downloaded/loaded once per experiment
        # 2. Consistent environment across metric calculations
        # 3. Better performance (no repeated instantiation)
        self.embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # Embedding model not available - metrics will fall back to token-based scoring
            pass

        # Dynamic metric registry: Maps metric names to (function, requires_seed, requires_embedder)
        # This allows easy swapping of metrics without changing core code
        self._metric_registry = {
            # Citation & Grounding
            "citation_coverage": (citation_coverage, False, False),
            "citation_correctness": (_citation_correctness_with_seed, True, False),
            "faithfulness": (faithfulness, True, True),

            # Context Quality
            "context_precision": (context_precision, False, False),
            "context_recall": (context_recall, False, False),

            # Answer Quality
            "answer_correctness": (_answer_correctness_with_embedder, False, True),

            # Domain-Specific (Repurposing)
            "scenario_identification_rate": (scenario_identification_rate, False, False),
            "property_identification_rate": (property_identification_rate, False, False),
            "output_consistency": (output_consistency, False, False),

            # Domain-Specific (ML Recommendation - Sonntag et al., 2025)
            "ml_problem_type_accuracy": (ml_problem_type_accuracy, False, False),
            "algorithm_suitability": (algorithm_suitability, False, False),

            # Hallucination
            "hallucination_rate": (hallucination_rate, True, True),

            # Readability
            "fkgl": (fkgl, False, False),

            # OCR Quality
            "cer": (cer, False, False),
            "wer": (wer, False, False),
        }

    def calculate_metric(self, metric_name: str, **kwargs) -> float:
        """
        Dynamically execute a metric with automatic seed and embedder injection.

        This is the core method for calculating metrics. It:
        1. Looks up the metric function from the registry
        2. Automatically injects self.random_seed if the metric accepts it
        3. Automatically injects self.embedder if the metric needs it
        4. Calls the metric function with the provided kwargs

        Args:
            metric_name: Name of the metric to calculate
            **kwargs: Metric-specific arguments (answer, contexts, etc.)

        Returns:
            Metric score

        Raises:
            ValueError: If metric_name is not registered

        Example:
            calculator = MetricCalculator(random_seed=42)
            score = calculator.calculate_metric(
                "faithfulness",
                answer="The pump is blocked.",
                contexts=["Manual section 1", "Manual section 2"]
            )
        """
        if metric_name not in self._metric_registry:
            raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(self._metric_registry.keys())}")

        metric_func, requires_seed, requires_embedder = self._metric_registry[metric_name]

        # Automatically inject seed if metric accepts it
        if requires_seed:
            kwargs["seed"] = self.random_seed

        # Automatically inject embedder if metric needs it
        if requires_embedder:
            kwargs["embedder"] = self.embedder

        return metric_func(**kwargs)

    def register_metric(self, name: str, func: callable, requires_seed: bool = False, requires_embedder: bool = False):
        """
        Register a custom metric function.

        This allows users to add their own metrics without modifying the core class.

        Args:
            name: Metric name
            func: Metric function
            requires_seed: Whether the metric accepts a 'seed' parameter
            requires_embedder: Whether the metric accepts an 'embedder' parameter

        Example:
            def custom_metric(answer: str, seed: int = 42) -> float:
                # Custom logic
                return 0.5

            calculator.register_metric("custom_metric", custom_metric, requires_seed=True)
        """
        self._metric_registry[name] = (func, requires_seed, requires_embedder)

    def get_available_metrics(self) -> List[str]:
        """Get list of all registered metric names."""
        return list(self._metric_registry.keys())


# ================= Metrics Configuration =================

class MetricsConfig:
    """Predefined metric sets for different task types"""

    DIAGNOSIS_METRICS = [
        "answer_correctness",
        "citation_coverage",
        "citation_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
        "hallucination_rate",
        "fkgl",
    ]

    REPURPOSING_METRICS = [
        "scenario_identification_rate",
        "property_identification_rate",
        "answer_correctness",
        "output_consistency",
    ]

    ML_RECOMMENDATION_METRICS = [
        # Problem understanding (Sonntag RQ1)
        "ml_problem_type_accuracy",  # Correct problem type identification
        "algorithm_suitability",      # Suitable algorithm recommended (TFR)

        # RAG-specific (existing metrics - work out of the box)
        "citation_coverage",          # How well citations cover answer
        "citation_correctness",       # Are citations correct?
        "faithfulness",               # Faithful to retrieved context?
        "context_precision",          # Precision of retrieval
        "context_recall",             # Recall of retrieval
        "hallucination_rate",         # Hallucination detection

        # Quality metrics
        "answer_correctness",         # Overall quality
        "output_consistency",         # Consistency (Sonntag's OCR)
    ]

    @staticmethod
    def get_metrics_for_task(task_type: str) -> List[str]:
        """Get recommended metrics for a task type"""
        if task_type == "diagnosis":
            return MetricsConfig.DIAGNOSIS_METRICS
        elif task_type == "repurposing":
            return MetricsConfig.REPURPOSING_METRICS
        elif task_type == "ml_recommendation":
            return MetricsConfig.ML_RECOMMENDATION_METRICS
        else:
            return MetricsConfig.DIAGNOSIS_METRICS  # Default
