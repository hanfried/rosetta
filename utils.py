import math
import os
import re
import tarfile

import pandas as pd
import requests
from tqdm import tqdm_notebook as tqdm


def download_file(fname, url):
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
def preprocess_europarl(line):
    line = re.sub(r'\d+', '0', line)
    line = re.sub(r'\s+', ' ', line)
    return line.lower().strip()


def read_europarl(language, data_path='data'):

    corpus_fname = os.path.join(data_path, f'europarl-v7.de-en.{language}')
    preprocessed_fname = f'{corpus_fname}.preprocessed'

    if os.path.exists(preprocessed_fname):
        return [line.strip() for line in open(preprocessed_fname)]
    else:
        preprocessed = [preprocess_europarl(line) for line in tqdm(open(corpus_fname, 'r'))]

        try:  # to cache
            open(preprocessed_fname, 'w').writelines(l + '\n' for l in preprocessed)
        except IOError as e:
            print(f'Could not cache: {e}')

        return preprocessed
