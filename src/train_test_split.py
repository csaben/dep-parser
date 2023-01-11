import sys
from config import *
from dataset import *
from inference import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

#def custom_pad_label(labels):
#    #get the max element lens per label in batch in the form [maxlen tup1, maxlen conn, maxlen tup2]
#    max_lens_batch = [max(max(len(sub_elem) for sub_elem in elem) if isinstance(elem, tuple) else len(elem)
#               for elem in example) for example in batch]
#    #find the greatest len batchwise
#    max_len = max(max_lens_batch)
#    pass



# def collate_fn(batch):
#     sentences, labels = zip(*batch)
#     padded_sentences = pad_sequence(sentences, batch_first=True)
#     print(padded_sequences)
#     padded_labels = custom_pad_label(labels)
#     padded_labels = torch.tensor(padded_labels, dtype=torch.long)
#     return padded_sentences, padded_labels

#hail mary collate fn; seems like it doesn't error out-> great! the masking fn is kinda a black box
def custom_collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences = [torch.tensors(s) for s in sentences]
    padded_sentences = pad_sequence(sentences, batch_first=True)

    #Find the max lengths of each sub-element of the label tuple
    max_lens_batch = [max(max(len(sub_elem) for sub_elem in elem) if isinstance(elem, tuple) else len(elem) for elem in example) for example in batch]                                      

    #Find the greatest len batchwise 
    max_len = max(max_lens_batch)
    
    #Create masks
    mask = [[[1 if j < len(elem[i]) else 0 for j in range(max_len)] if isinstance(elem[i], (tuple, list)) else [1]*len(elem[i])+[0]*(max_len-len(elem[i]))  for i in range(3)] for elem in batch]

    #pad labels
    padded_labels = [[[np.pad(sub_elem, (0, max_len - len(sub_elem)), 'constant') if isinstance(sub_elem, np.ndarray) else sub_elem for sub_elem in elem] if isinstance(elem, tuple) else np.pad(elem, (0, max_len - len(elem)), 'constant') for elem in example] for example in batch]
    
    return padded_sentences, padded_labels, mask



with open(DATA, 'r') as file:
    text = file.read()

sentences = text.split("\n")

# Split the dataset into training and validation sets
dataset = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)
val_ratio = 0.2
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4,
                              collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4,
                            collate_fn=custom_collate_fn)



for i, (sentences, tokenized_labels) in enumerate(train_dataloader):
    print(sentences)
    print(labels)
    break

#lets try just hitting it with a dataloader



# sent, token_label = ds[55]
# print(sent)
# print(token_label)

# decoded_label = detokenize_label(token_label)
# print(decoded_label)
