import sys
from config import *
from dataset import *
from inference import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss

#hail mary collate fn; seems like it doesn't error out-> great! the masking fn is kinda a black box
def custom_collate_fn(batch):
    
    padded_sentences, padded_labels, mask = None,None,None
    return padded_sentences, padded_labels, mask


with open(DATA, 'r') as file:
    text = file.read()

sentences = text.split("\n")
# print(len(max(sentences)))
#98
# import sys
# sys.exit()
dataset = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)
dataset[0]
"""

need a dataloader to work

Label
[(([2436], [1058, 2497]), [13221, 7716], ([3187], [1050, 2078])), (([3187], [1050, 2078]), [2553], ([1
997], [1999])), (([3187], [1050, 2078]), [20010], ([1996], [26718])), (([2436], [1058, 2497]), [13221,
 7716], ([3639], [1050, 2078])), (([3639], [1050, 2078]), [2553], ([1997], [1999]))]

Sentence
OFFICE OF THE SECRETARY OF DEFENSE


Cool one issue was i wasn't tokenized sentence, i have adjusted that

now I need to handle the padding issue for each batch(?) w.r.t token ids of both sentences and
labels

see tracebacks (most recent)
Traceback (most recent call last):
  File "dataloader.py", line 45, in <module>
    for (sentence, label) in enumerate(train_dataloader):
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/dataloader.py",
line 521, in __next__
    data = self._next_data()
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/dataloader.py",
line 1203, in _next_data
    return self._process_data(data)
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/dataloader.py",
line 1229, in _process_data
    data.reraise()
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/_utils.py", line 425, in re
raise
    raise self.exc_type(msg)
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/_utils/worker.py
", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py"
, line 47, in fetch
    return self.collate_fn(data)
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/_utils/collate.p
y", line 84, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/_utils/collate.p
y", line 84, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/home/arelius/anaconda3/envs/tmp/lib/python3.8/site-packages/torch/utils/data/_utils/collate.p
y", line 82, in default_collate
    raise RuntimeError('each element in list of batch should be of equal size')
RuntimeError: each element in list of batch should be of equal size
"""
val_ratio = 0.2
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
ct=0
for (sentence, label) in enumerate(train_dataloader):
    ct+=1
print(ct)
## Split the dataset into training and validation sets
#dataset = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)

## Create DataLoader for training and validation sets
#train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2,
#                              collate_fn=custom_collate_fn)
#val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2,
#                            collate_fn=custom_collate_fn)



#print(len(train_dataloader))
#"""the custom collate fn is causing problems for my dataloader

#   also seems like there may be another issue. uggggghh
#"""
#for i, (sentences, tokenized_labels, mask) in enumerate(val_dataloader):
#    print(sentences[0])
#    print(tokenized_labels[0])
#    print(mask[0])
#    print("________________________")
#    print(tokenized_sentences.numpy())
#    print(labels.numpy())
#    print(mask.numpy)
#    # print(1)

##lets try just hitting it with a dataloader



## sent, token_label = ds[55]
## print(sent)
## print(token_label)

## decoded_label = detokenize_label(token_label)
## print(decoded_label)
