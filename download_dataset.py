import urllib.request
import tarfile
import os

def download_imdb_dataset():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    
    print("Downloading IMDb dataset...")
    urllib.request.urlretrieve(url, filename)
    
    print("Extracting dataset...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall('data/')
    
    os.remove(filename)
    print("Dataset ready in data/aclImdb/")

if __name__ == "__main__":
    download_imdb_dataset()