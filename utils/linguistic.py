import os
import re

import numpy as np
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm_notebook as tqdm

try:
    from spacy.lang.de import German
except ModuleNotFoundError:
    spacy.cli.download('de')
    from spacy.lang.de import German


# Follows https://github.com/bheinzerling/bpemb/blob/master/preprocess_text.sh
# (ignoring urls as there shouldn't be any in parliament discussions)
def preprocess_input_europarl(line):
    line = re.sub(r'\d+', '0', line)
    line = re.sub(r'\s+', ' ', line)
    return line.lower().strip()


def read_europarl(language, data_path='data'):

    corpus_fname = os.path.join(data_path, f'europarl-v7.de-en.{language}')
    preprocessed_fname = f'{corpus_fname}.preprocessed'

    if os.path.exists(preprocessed_fname):
        return [line.strip() for line in open(preprocessed_fname)]

    preprocessed = [preprocess_input_europarl(line) for line in tqdm(open(corpus_fname, 'r'))]

    try:  # to cache
        open(preprocessed_fname, 'w').writelines(l + '\n' for l in preprocessed)
    except IOError as e:
        print(f'Could not cache: {e}')

    return preprocessed


def bleu_scores_europarl(
    input_texts, target_texts, predict,
    parser=German()
):
    assert len(input_texts) == len(target_texts)
    N = len(input_texts)

    # to handle short sequences, see also
    # http://www.nltk.org/_modules/nltk/translate/bleu_score.html#SmoothingFunction.method3
    chencherry = SmoothingFunction()
    
    def remove_spaces_and_puncts(tokens):
        return [token.orth_ for token in tokens if not (token.is_space or token.is_punct)]

    bleu_scores = np.zeros(N)

    for i in tqdm(range(N)):
        ref_tokens = remove_spaces_and_puncts(parser(target_texts.iloc[i]))
        pred_tokens = remove_spaces_and_puncts(parser(predict(input_texts.iloc[i])))
        bleu_scores[i] = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=chencherry.method3
        )

    return bleu_scores
