import os
import numpy as np
from scipy.sparse import csr_matrix

from ..utils import download_a_file
from ..utils import external_path
from ..utils import unzip


movielens_20m_url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
movielens_20m_readme_url = 'http://files.grouplens.org/datasets/movielens/ml-20m-README.html'
movielens_small_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
movielens_small_readme_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html'

movielens_20m_dir = f'{external_path}/movielens/ml-20m/'
movielens_small_dir = f'{external_path}/movielens/ml-latest-small/'

def download_movielens(size='20m', force=False):
    """
    Arguments
    ---------
    size : str
        Datasize. one of ['20m', 'small']
    force : Boolean
        If True, it overwrites data if the file exists
    """

    data_url, readme_url, dirname = check_size(size)
    download(data_url, readme_url, dirname, force)

def download(data_url, readme_url, dirname, force=False):
    if os.path.exists(dirname) and (not force):
        answer = input('The data already is downloaded. Re-download it? [yes|no]').lower().strip()
        if answer != 'yes':
            print('Terminated')
            return None

    notice(readme_url)

    filename = data_url.split("/")[-1]
    zippath = f'{external_path}/movielens/{filename}'
    if download_a_file(data_url, zippath):
        print('downloaded')
    if unzip(zippath, f'{external_path}/movielens/'):
        print('unzip the downloaded file')
    if os.path.exists(zippath):
        os.remove(zippath)

def notice(readme_url):
    print("This function downloads MovieLens data from GroupLens\n" \
          f"Please read first {readme_url}\n" \
          "All permissions are in GroupLens, and this function is an external utility" \
          " to conventiently use MovieLens data.\n")

def check_size(size):
    if not size in ['20m', 'small']:
        raise ValueError("`size` must be one of ['20m', 'small']")

    if size == '20m':
        data_url = movielens_20m_url
        readme_url = movielens_20m_readme_url
        dirname = movielens_20m_dir
    else:
        data_url = movielens_small_url
        readme_url = movielens_small_readme_url
        dirname = movielens_small_dir
    return data_url, readme_url, dirname

def load_rating(size='20m'):
    """
    Arguments
    ---------
    size : str
        Datasize. one of ['20m', 'small']

    Returns
    -------
    user_item : scipy.sparse.csr_matrix
        (user, item) = rate
    timestamps : numpy.ndarray
        UNIX time format datetime
    """
    data_url, readme_url, dirname = check_size(size)
    if not os.path.exists(dirname):
        print("The data has not been downloaded, Download it first.\n")
        download(data_url, readme_url, dirname)
    else:
        notice(readme_url)

    rating_path = f'{dirname}/ratings.csv'

    users = []
    items = []
    ratings = []
    timestamps = []

    with open(rating_path, encoding='utf-8') as f:
        next(f)
        for line in f:
            u, i, r, t = line.strip().split(',')
            users.append(int(u))
            items.append(int(i))
            ratings.append(float(r))
            timestamps.append(int(t))

    user_item = csr_matrix((ratings, (users, items)))
    timestamps = np.array(timestamps)
    return user_item, timestamps
