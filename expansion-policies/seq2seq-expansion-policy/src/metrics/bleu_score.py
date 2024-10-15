from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class BleuScore:
    @staticmethod
    def smoothed_corpus_bleu(references, hypotheses):
        smoothing_function = SmoothingFunction().method1
        bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing_function)

        return bleu_score
