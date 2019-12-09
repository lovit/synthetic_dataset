import os
from ..utils import download_a_file
from ..utils import external_path
from ..utils import unzip


movielens_20m_url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
movielens_20m_readme_url = 'http://files.grouplens.org/datasets/movielens/ml-20m-README.html'
movielens_small_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
movielens_small_readme_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html'

movielens_20m_dir = f'{external_path}/movielens/ml-20m/'
movielens_small_dir = f'{external_path}/movielens/ml-latest-small/'

def download_movielens_20m(force=False):
    data_url = movielens_20m_url
    readme_url = movielens_20m_readme_url
    dirname = movielens_20m_dir
    download(data_url, readme_url, dirname, force)

def download_movielens_small(force=False):
    data_url = movielens_small_url
    readme_url = movielens_small_readme_url
    dirname = movielens_small_dir
    download(data_url, readme_url, dirname, force)

def download(data_url, readme_url, dirname, force=False):
    print("This function downloads MovieLens data from GroupLens\n" \
          f"Please read first {readme_url}\n" \
          "All permissions are in GroupLens, and this function is an external utility" \
          " to conventiently use MovieLens data.\n")

    if os.path.exists(dirname) and (not force):
        answer = input('The data already is downloaded. Re-download it? [yes|no]').lower().strip()
        if answer != 'yes':
            print('Terminated')
            return None

    filename = data_url.split("/")[-1]
    zippath = f'{external_path}/movielens/{filename}'
    if download_a_file(data_url, zippath):
        print('downloaded')
    if unzip(zippath, f'{external_path}/movielens/'):
        print('unzip the downloaded file')
    if os.path.exists(zippath):
        os.remove(zippath)
