"""Corpus preprocessing and reproducible train/test matrix construction."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

WORD_PATTERN = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class CorpusSplit:
    """Normalized matrices and metadata for a reproducible document split."""

    training_matrix: np.ndarray
    testing_matrix: np.ndarray
    training_document_ids: tuple[str, ...]
    testing_document_ids: tuple[str, ...]
    training_bigrams: int
    testing_bigrams: int


def _letter_index(letter: str) -> int:
    return ord(letter) - ord("a")


def count_token_bigrams(
    tokens: Iterable[str],
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """Count adjacent letters inside tokens; token boundaries break transitions."""
    counts = np.zeros((26, 26), dtype=float)
    total = 0

    for token in tokens:
        for word in WORD_PATTERN.findall(str(token).lower()):
            for first, second in zip(word, word[1:]):
                counts[_letter_index(first), _letter_index(second)] += 1.0
                total += 1

    if total == 0:
        raise ValueError("The supplied tokens contain no alphabetic bigrams.")
    if normalize:
        counts /= total
    return counts, total


def count_text_bigrams(
    texts: Iterable[str],
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """Count within-word bigrams from one or more raw text strings."""
    tokens: list[str] = []
    for text in texts:
        tokens.extend(WORD_PATTERN.findall(str(text).lower()))
    return count_token_bigrams(tokens, normalize=normalize)


def _ensure_brown_corpus(download: bool) -> object:
    try:
        import nltk
        from nltk.corpus import brown
    except ImportError as error:
        raise RuntimeError(
            "NLTK is required for the Brown Corpus. Run 'python -m pip install -e .'."
        ) from error

    try:
        brown.fileids()
    except LookupError:
        if not download:
            raise RuntimeError(
                "The Brown Corpus is not installed and download=False was requested."
            )
        if not nltk.download("brown", quiet=False):
            raise RuntimeError("NLTK could not download the Brown Corpus.")
    return brown


def load_brown_train_test(
    *,
    test_fraction: float = 0.20,
    seed: int = 42,
    download: bool = True,
) -> CorpusSplit:
    """Create a reproducible document-level Brown Corpus split."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    brown = _ensure_brown_corpus(download)
    document_ids = list(brown.fileids())
    random.Random(seed).shuffle(document_ids)

    testing_count = max(1, round(len(document_ids) * test_fraction))
    testing_ids = tuple(sorted(document_ids[:testing_count]))
    training_ids = tuple(sorted(document_ids[testing_count:]))
    if not training_ids:
        raise RuntimeError("The training split is empty.")

    training_tokens: list[str] = []
    for document_id in training_ids:
        training_tokens.extend(brown.words(fileids=[document_id]))

    testing_tokens: list[str] = []
    for document_id in testing_ids:
        testing_tokens.extend(brown.words(fileids=[document_id]))

    training_matrix, training_total = count_token_bigrams(training_tokens)
    testing_matrix, testing_total = count_token_bigrams(testing_tokens)

    return CorpusSplit(
        training_matrix=training_matrix,
        testing_matrix=testing_matrix,
        training_document_ids=training_ids,
        testing_document_ids=testing_ids,
        training_bigrams=training_total,
        testing_bigrams=testing_total,
    )

