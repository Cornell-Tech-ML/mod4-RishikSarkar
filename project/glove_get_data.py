import os
import zipfile
import urllib.request

def download_glove():
    if not os.path.exists('project/data'):
        os.makedirs('project/data')

    if os.path.exists('project/data/glove.6B'):
        print("GloVe directory already present, exiting")
        return

    glove_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    zip_path = 'project/data/glove.6B.zip'

    print("Downloading GloVe embeddings...")
    urllib.request.urlretrieve(glove_url, zip_path)

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('project/data/glove.6B')

    os.remove(zip_path)
    print("Done!")

def load_glove_embeddings(path="project/data/glove.6B/glove.6B.50d.txt"):
    """Load GloVe embeddings from local file."""
    word2emb = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            word2emb[word] = vector
    return word2emb

class GloveEmbedding:
    """Simple wrapper class to mimic the original GloveEmbedding interface."""

    def __init__(self, word2emb):
        self.word2emb = word2emb
        self.d_emb = len(next(iter(word2emb.values())))

    def emb(self, word, default=None):
        """Get embedding for a word."""
        return self.word2emb.get(word, default)

    def __contains__(self, word):
        """Support for 'in' operator."""
        return word in self.word2emb

def get_embeddings():
    """Helper function to download and load embeddings."""
    download_glove()
    word2emb = load_glove_embeddings()
    return GloveEmbedding(word2emb)

if __name__ == '__main__':
    embeddings = get_embeddings()
    print(f"Embedding dimension: {embeddings.d_emb}")
    print(f"Example embedding for 'the': {embeddings.emb('the')[:5]}...")