import torchtext

# URL of the Penn Treebank dataset
url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'

# Destination directory
destination_directory = '../input/'

# Download the dataset
torchtext.utils.download_from_url(url=url, root=destination_directory)
