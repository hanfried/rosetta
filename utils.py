import math
import os
import re
import tarfile

import numpy as np
import requests
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm_notebook as tqdm

try:
    from spacy.lang.de import German
except ModuleNotFoundError:
    spacy.cli.download('de')
    from spacy.lang.de import German


# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(fname, url):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    url, id = re.search(r'^(.*)\?id=(.*)', url).groups()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, fname)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file(fname, url):
    if 'drive.google.com' in url:
        print(f'Download {fname} from Google Drive {url}')
        download_file_from_google_drive(fname, url)
        return

    print(f'Downloading {fname} from {url} ...')
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    download = tqdm(
        response.iter_content(block_size),
        total=math.ceil(total_size // block_size),
        unit='KB',
        unit_scale=True
    )
    with open(f"{fname}", "wb") as handle:
        for data in download:
            handle.write(data)


def download_and_extract_resources(fnames_and_urls, dest_path):
    os.makedirs(dest_path, exist_ok=True)

    for name, url in fnames_and_urls.items():
        fname = os.path.join(dest_path, name)
        exists = os.path.exists(fname)
        size = os.path.getsize(fname) if exists else -1
        if exists and size > 0:
            print(f'{name} already downloaded ({size / 2**20:3.1f} MB)')
            continue
        download_file(fname, url)
        if (re.search(r'\.(tgz|tar\.gz)$', fname)):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall(path=dest_path)
            tar.close()
            print(f'Extracted {fname} ...')


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
    else:
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
        ref_tokens = remove_spaces_and_puncts(parser(target_texts[i]))
        pred_tokens = remove_spaces_and_puncts(parser(predict(input_texts[i])))
        bleu_scores[i] = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=chencherry.method3
        )

    return bleu_scores
