from __future__ import annotations

import re
from collections import Counter
from itertools import tee
from typing import Sequence

import numpy as np
from numpy import ma


def bleu(actual: str, reference: str, ks: Sequence[int]) -> list[float]:
    """Compute the BLEU-k score(s).

    BLEU, a.k.a. Bilingual Evaluation Understudy, computes the k-gram overlap between
    the actual and the reference sentences to measure similarity.

    Parameters
    ----------
    actual : str
        The actual sentence.
    reference : str
        The reference sentence.
    ks : list of int
        The list degrees k.

    Returns
    -------
    scores : list of float
        The list of scores corresponding to the degrees.
    """
    act_list = _split_sentence(actual)
    ref_list = _split_sentence(reference)
    act_length, ref_length = len(act_list), len(ref_list)

    # Algorithm in https://dl.acm.org/doi/10.3115/1073083.1073135
    # Compute the modified precision for each degree
    maxk = max(ks)
    precisions = np.zeros(maxk, dtype=np.float64)
    for i in range(maxk):
        act_counter = _ngram_counter(act_list, i + 1)
        ref_counter = _ngram_counter(ref_list, i + 1)
        tot_clip, tot = 0, 0
        for gram, count in act_counter.items():
            tot_clip += min(count, ref_counter[gram])
            tot += count
        # https://github.com/nltk/nltk/blob/develop/nltk/translate/bleu_score.py#L364
        precisions[i] = tot_clip if tot == 0 else tot_clip / tot

    # Compute the brevity penalty
    brevity_penalty: float = 1
    if act_length == 0:
        brevity_penalty = 0
    elif act_length <= ref_length:
        brevity_penalty = np.exp(1 - ref_length / act_length)

    # Compute the BLEU scores
    sc = np.exp(np.sum(ma.log(precisions).filled(0)))
    return [brevity_penalty * np.float_power(sc, 1 / k) for k in ks]


def _split_sentence(sentence: str) -> list[str]:
    """Split a sentence.

    This split Chinese characters one by one, and all other characters by spaces.

    Parameters
    ----------
    sentence : str
        The sentence to split.

    Returns
    ------
    splitted_sentence : list of str
        The splitted sentence.
    """
    pattern = re.compile(r"[\u4e00-\u9fff]+")
    matches = re.finditer(pattern, sentence)

    # Iterate through the matches and extend the splits to the results
    splitted_sentence: list[str] = []
    current = 0
    for mat in matches:
        mat_start = mat.start()
        if mat_start != current:
            splitted_sentence.extend(sentence[current:mat_start].split())
        splitted_sentence.extend(mat.group(0))
        current = mat.end()

    # Maybe missing one last part
    if current < len(sentence):
        splitted_sentence.extend(sentence[current:].split())
    print(splitted_sentence)
    return splitted_sentence


def _ngram_counter(seq: list[str], n: int) -> Counter[tuple[str]]:
    """Generates n-grams and wrap in a counter.

    Parameters
    ----------
    seq : list of str
        The list to generate n-grams of.
    n : int
        The degree.

    Returns
    -------
    counter : collections.Counter
        The counter of the n-grams. For instance, the 2-grams of ``[1, 2, 1, 2]`` are
        ``[(1, 2), (2, 1), (1, 2)]``, so that ``(1, 2)`` counts twice and ``(2, 1)``
        counts once. If the length of ``seq`` is greater than the degree, an empty
        counter will be returned.

    Notes
    -----
    The way to generate the n-grams refers to the sample code snippet of
    :func:`itertools.pairwise`. :func:`itertools.tee` returns n independent iterators.
    For the ith iterator, consume its first i elements so that their starting element
    are consecutive. Finally :func:`zip` respects the shortest iterator so that the n-
    grams are done. See also
    https://docs.python.org/3/library/itertools.html#itertools.pairwise.
    """
    if len(seq) < n:
        return Counter()

    # https://docs.python.org/3/library/itertools.html#itertools.pairwise
    its = tee(seq, n)
    for i, it in enumerate(its):
        for _ in range(i):
            next(it, None)
    # mypy does not recognize zip as Iterable
    return Counter(zip(*its))  # type: ignore[arg-type]
