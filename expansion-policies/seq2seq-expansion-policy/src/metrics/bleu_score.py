from typing import List, Sequence

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BleuScore:
    """
    BleuScore

    Provides methods to compute BLEU (Bilingual Evaluation Understudy) scores for evaluating the quality
    of machine-translated text by comparing it to one or more reference translations.

    BLEU scores are widely used in natural language processing tasks to assess the similarity between
    generated text and reference texts, with higher scores indicating better performance.

    Methods
    -------
    smoothed_corpus_bleu(references, hypotheses)
        Computes the smoothed corpus-level BLEU score.
    """
    @staticmethod
    def smoothed_corpus_bleu(
        references: List[Sequence[str]],
        hypotheses: List[Sequence[str]]
    ) -> float:
        """
        Compute the smoothed corpus-level BLEU score.

        Applies smoothing to handle cases where n-gram counts are zero, which can adversely affect the BLEU score,
        especially for shorter hypotheses or references.

        Parameters
        ----------
        references : List[Sequence[str]]
            A list where each element is a list of reference translations for a single hypothesis.
            Each reference translation is itself a list of tokens (strings).

            Example:
                [
                    [["this", "is", "a", "test"], ["this", "is", "test"]],
                    [["another", "test"]]
                ]

        hypotheses : List[Sequence[str]]
            A list of hypothesis translations, where each hypothesis is a list of tokens (strings).

            Example:
                [
                    ["this", "is", "a", "test"],
                    ["another", "test"]
                ]

        Returns
        -------
        float
            The smoothed corpus-level BLEU score ranging from 0.0 to 1.0.

        Raises
        ------
        ValueError
            If the number of hypotheses does not match the number of reference lists.
        """
        smoothing_function = SmoothingFunction().method1
        return corpus_bleu(references, hypotheses, smoothing_function=smoothing_function)
