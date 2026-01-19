from __future__ import annotations
from typing import List, Tuple
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


_WORD_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)  # keeps simple contractions like don't


def _to_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return "a"  # adj
    if treebank_tag.startswith("V"):
        return "v"  # verb
    if treebank_tag.startswith("N"):
        return "n"  # noun
    if treebank_tag.startswith("R"):
        return "r"  # adv
    return "n"


class BaselineNLPChecker:
    """
    Traditional-NLP baseline:
        - tokenize + POS-tag + lemmatize sentence
        - lemmatize each rule word
        - compliance: sentence is NON-compliant if any rule word lemma appears
        - score: 1 if contains a forbidden word else 0
    """

    def __init__(self, nltk_data_dir: str | None = None, auto_download: bool = True):
        self._configure_nltk(nltk_data_dir, auto_download)
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("english")
        self._rule_pos = ("v", "n", "a")  # verb, noun, adj

    def _configure_nltk(self, nltk_data_dir: str | None, auto_download: bool):
        if nltk_data_dir:
            os.makedirs(nltk_data_dir, exist_ok=True)
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_dir)

        if auto_download:
            self._ensure_nltk_resources(download_dir=nltk_data_dir)

    def _ensure_nltk_resources(self, download_dir: str | None):
        """
        Robustly ensure required NLTK resources exist.

        NOTE: Newer NLTK may require 'averaged_perceptron_tagger_eng' instead of
        'averaged_perceptron_tagger'. We support both.
        """
        required = [
            # Punkt tokenizer
            ("tokenizers/punkt", "punkt"),
            # POS tagger (new + old names)
            (
                "taggers/averaged_perceptron_tagger_eng",
                "averaged_perceptron_tagger_eng",
            ),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            # WordNet lemmatizer data
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]

        for find_path, download_name in required:
            try:
                nltk.data.find(find_path)
            except LookupError:
                # If download_dir is None, NLTK uses its default download locations
                nltk.download(download_name, download_dir=download_dir, quiet=True)

    def _sentence_norm(self, sentence: str):
        tokens = _WORD_RE.findall(sentence.lower())
        pos_tags = nltk.pos_tag(tokens)

        lemmas = []
        stems = []
        for tok, tag in pos_tags:
            wn_pos = _to_wordnet_pos(tag)
            lemmas.append(self.lemmatizer.lemmatize(tok, pos=wn_pos))
            stems.append(self.stemmer.stem(tok))
        return set(lemmas), set(stems)

    def _rules_norm(self, rules: str):
        def extract_words(rule: str) -> List[str]:
            match = re.search(r"\[(.*?)\]", rule)
            return [item.strip() for item in match.group(1).split(",")] if match else []

        words = extract_words(rules)
        lemmas = []
        stems = []
        for w in words:
            w = (w or "").strip().lower()
            if not w:
                continue

            # keep raw, plus multiple lemma POS
            lemmas.append(w)
            for pos in self._rule_pos:
                lemmas.append(self.lemmatizer.lemmatize(w, pos=pos))

            stems.append(self.stemmer.stem(w))
        return set(lemmas), set(stems)

    def check_compliance(self, sentence: str, rules: str) -> Tuple[bool, int]:
        sent_lemmas, sent_stems = self._sentence_norm(sentence)
        rule_lemmas, rule_stems = self._rules_norm(rules)
        contains = bool(sent_lemmas & rule_lemmas) or bool(sent_stems & rule_stems)
        return contains, int(contains)
