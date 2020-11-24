import os
import re
import sys
import tarfile


def open_data():
    """

    :return: List of text objects
    """
    fpath = os.getcwd() + "\\data\\nips12raw_str602.tgz"
    if not os.path.isfile(fpath):
        sys.exit("You need to download the dataset in https://cs.nyu.edu/~roweis/data.html")
    with tarfile.open(fpath, mode='r:gz') as tar:
        # Ignore directory entries, as well as files like README, etc.
        files = [
            m for m in tar.getmembers()
            if m.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', m.name)
        ]
        for member in sorted(files, key=lambda x: x.name):
            member_bytes = tar.extractfile(member).read()
            yield member_bytes.decode('utf-8', errors='replace')

