import re
import math
import os
import tarfile

from tqdm import tqdm_notebook as tqdm
import requests

from utils.google_drive import download_file_from_google_drive


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
