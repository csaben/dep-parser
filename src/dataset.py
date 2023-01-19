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
                curr = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(curr))
                self.add_tokens([curr])
        return curr

    def pad_label(self, label):
        stack = [label]
        while stack:
            curr = stack.pop()
            if isinstance(curr, tuple):
                curr = tuple(self.pad_label(elem) for elem in curr)
            elif isinstance(curr, list):
                #MAX_LEN / 5 = 30 , feels fine to me, might need to remove instances of super long sentences
                # curr = 
                # does my labels have to be same size??? how do people normally do both??
        return curr

    def save_labels(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.original_labels, self.tokenized_labels), f)

    def __getitem__(self, index):
        """
        for better training increase the number of documents you train on and adjust your
        getitem to handle each item as a list of sentences rather than the step less of that
        that we are currently handling
        source for that idea: https://github.com/abhishekkrthakur/bert-entity-extraction/blob/master/src/dataset.py

        YO actually that loop in his tokenizes each word at a time
        his pos tagging is also weird bc he effectively adds pos labels for each
        additional indice that a word that becomes tokenized gets. this all starts to
        make sense

        Clark
        Tokenized: Cl# a## rk# k##
        Token_id = [2, 3, 6, 7, 9]
        now in your pos tag if for Clark:word [6]:pos_tag then you now need [6 6 6 6 6] because
        Clark:word is now 5 subwords that are token_ids, so yeah I need to make some adjustments
        to this data getitem function . or maybe its fine as it is but I still need to
        pad the sentence and pad the labels to be the same. so like does my complex tuple
        thing have to have each array padded the same then, is that the only criteria?
        """
        sentence = self.sentences[index]
        original_label = self.dep_parser.parse(sentence.split())
        original_label = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for
                 parse in original_label]
        #I could be storing original_label instead 
        self.original_labels.append(original_label[0])

        #would put in a loop if you were processing more than one sentence per batch
        sentence = self.tokenizer.tokenize(sentence)
        sentence = self.tokenizer.convert_tokens_to_ids(sentence)

        # sentence = sentence[:MAX_LEN - 2] #-2 for the CLS SEP tags



        # sentence = [101]+sentence+[102]


        tokenized_label = self.tokenize_label(original_label[0]) #unpacks tuple of tuple
        print(sentence)
        print(tokenized_label)
        
        # tokenized_label=[2023]
        return (sentence, tokenized_label)

    def add_tokens(self, tokens: List[str]):
        self.tokenizer.add_tokens(tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        #NOTE: is training affected if I don't do this, no converge == come back here
        #apparently to do this it has to not be in place (not sure how to handle it rn)
        # self.model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([self.model.config.hidden_size])



if __name__ == '__main__':
    main()
