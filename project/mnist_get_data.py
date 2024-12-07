import os
import gzip
import urllib.request

def download_mnist():
    if os.path.exists('data'):
        print("data directory already present, exiting")
        return
    
    os.makedirs('data')
    os.chdir('data')
    
    base_url = 'https://raw.githubusercontent.com/fgnt/mnist/master'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    for file in files:
        print(f"Downloading {file}...")
        urllib.request.urlretrieve(f"{base_url}/{file}", file)
    
    for file in files:
        with gzip.open(file, 'rb') as f_in:
            with open(file[:-3], 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(file)
    
    os.chdir('..')

if __name__ == '__main__':
    download_mnist()