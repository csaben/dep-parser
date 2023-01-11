import random
import sys
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.parse.corenlp import CoreNLPDependencyParser
import torchtext
from config import *

def main():
    #assume we open file, read contents to array as list of sentences
    sentences = ["I prefer the morning flight to Denver.", "I fight the world."]
    ds = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)
    print(ds[0])
    print(ds.model)


class DependencyParserDataset(Dataset):
    def __init__(self, sentences, tokenizer, model, dep_parser):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.model = model
        self.dep_parser = dep_parser

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.dep_parser.parse(sentence.split())
        label = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for
                 parse in label]
        self.clean_label(label)
        #FINAL TODO S:
        #LITERALLY JUST TRYING TO GET THE glove tokenizer to ignore ones it doesnt know and
        #from there I just need to format the outputted tensors into what I described just
        #after the sys.exit() line. from there I make basic transformer that handles the
        #dicts and could possibly train. I then make article download to dataset pipeline
        #for the beginning of this generator so it is a list of sentences from each article.
        #then I train model. then i eval model. then i reconstruct model guesses using a 
        #decoder glove vocab function (maybe huggingface has something better than glove)
        #then I slap the docker file together and run a test document and look at the dep parse
        sys.exit()

        #im thinking final format will be a dictionary torch.tensor([{'token':'word', pos:[nsub,NN], dep:[0,3]}, ...)]
        #and the pos and dep will be respective so it has an implicit format, kinda like mnist
        #index framing of the problem
        return sentence, label

    def clean_label(self, label):
        #remove exterior list
        label = label[0]
        #stacked tuples as illustrated [ (('word', 'pos'), 'dep', ('word', pos)) , ...]
        
        # Load the GloVe vectors
        glove = torchtext.vocab.GloVe(name='6B', dim=100)

        # Add a special OOV token to the vocabulary
        # oov_token = '<OOV>'
        # glove.stoi[oov_token] = len(glove.stoi)
        # glove.itos.append(oov_token)

        # trouble shooting oov
        #https://github.com/pytorch/text/issues/1350 i am using unk better to actually overwite
        # with rng nums from normal dist.
        # Create a special <unk> token and set its index to 0
        # unk_token = '<pad>'
        unk_token = 'shebeest' #bc no unk token in sight
        unk_index = 0

        # Create a Vocab object from the stoi dictionary of the GloVe object
        glove_vocab = torchtext.vocab.GloVe().stoi

        # Insert the <unk> token into the Vocab object
        # glove_vocab.insert_token(unk_token, unk_index)

        # Set the default index of the Vocab object to the index of the <unk> token
        # glove_vocab.set_default_index(unk_index)

        # Use a list comprehension to apply the vectors attribute of the GloVe object to each string in the tuples
        result = []
        for tup in label:
            try:
                result.append((glove.vectors[glove.stoi[tup[0][0]]], glove.vectors[glove.stoi[tup[0][1]]], glove.vectors[glove.stoi[tup[2][0]]]))
            except KeyError:
                glove_vocab[unk_index] = unk_token
                unk_token = random.choice(list(glove_vocab.keys()))
                # If a string does not have a corresponding entry in the stoi dictionary, use the OOV token
                #just throw a random number in
                # random_idx = random.choice(list(glove_vocab.values()))
                # print(random_idx)
                # # unk_token = self.unk_init(random_idx)
                # unk_token = random_idx
                result.append((glove.vectors[glove.stoi[unk_token]],
                               glove.vectors[glove.stoi[unk_token]],
                               glove.vectors[glove.stoi[unk_token]]))
        print(result)







if __name__ == '__main__':
    main()
