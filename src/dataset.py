from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logging.set_verbosity(40) #only show errors
import sys

import torch
from transformers import BertTokenizer, BertModel
from config import *
from label_mapping import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Type
from typing import List
import pickle

def main():
    #punctuation showing up in label is weird
    sentences = ["I prefer the morning flight to Denver.", "I fight the world."]
    ds = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)
    ds[0]


class DependencyParserDataset(Dataset):
    def __init__(self, sentences, tokenizer, model, dep_parser):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.model = model
        self.dep_parser = dep_parser
        self.original_labels = []
        self.tokenized_labels = []
        # self.label_mapping = LabelMapping()

    def __len__(self):
        return len(self.sentences)

    def tokenize_label(self, label):
        stack = [label]
        while stack:
            curr = stack.pop()
            if isinstance(curr, tuple):
                curr = tuple(self.tokenize_label(elem) for elem in curr)
            elif isinstance(curr, list):
                curr = [self.tokenize_label(elem) for elem in curr]
            elif isinstance(curr, str):
                curr = '[CLS] '+curr+' [SEP]'
                curr = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(curr))
                self.add_tokens([curr])
        return curr

    #@staticmethod
    #def detokenize_label(label):
    #    stack = [label]
    #    while stack:
    #        curr = stack.pop()
    #        if isinstance(curr, tuple):
    #            curr = tuple(DependencyParserDataset.detokenize_label(elem) for elem in curr)
    #        elif isinstance(curr, list):
    #            curr = [DependencyParserDataset.detokenize_label(elem) for elem in curr]
    #        elif isinstance(curr, int):
    #            curr = self.tokenizer.convert_ids_to_tokens([curr])
    #            #this doesn't fully recover it so instead store original label and keep track
    #            curr = self.tokenizer.convert_tokens_to_string(curr)
    #            #apparently there is one
    #            #TODO: i just add this line below, does it break things?
    #            curr = tokenizer.decode(curr)#should i dump the curr[0]?
    #    return curr

    def save_labels(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.original_labels, self.tokenized_labels), f)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        original_label = self.dep_parser.parse(sentence.split())
        original_label = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for
                 parse in original_label]
        #I could be storing original_label instead 
        self.original_labels.append(original_label[0])

        tokenized_label = self.tokenize_label(original_label[0])

        #TODO: is this actually accessible this way or should I also store original?
        #TODO: this prob goes away if I have a mapping generator script
        # self.label_mapping.add_mapping(tokenized_label, )

        
        return sentence, tokenized_label

    def add_tokens(self, tokens: List[str]):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        #apparently to do this it has to not be in place (not sure how to handle it rn)
        # self.model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([self.model.config.hidden_size])



if __name__ == '__main__':
    main()