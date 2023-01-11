from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logging.set_verbosity(40) #only show errors

import os
import sys
import glob
import pickle
from config import *
from dataset import *


class Inference:
    def __init__(self):
        self.label_mapping = {}
        self.directory = "../output/label_mappings/"

    def load_label_mapping(self):
        number_of_label_mappings = len(glob.glob(self.directory))
        for i in range(number_of_label_mappings):
            with open(f'../output/label_mappings/label_mapping_{i}.pkl', 'rb') as f:
                # original_label, tokenized_label = pickle.load(f)
                self.label_mapping.update(pickle.load(f))
                #TODO: did the connector get washed away by mistake double check to_tuple in
                #      label_mapping clas
    
    def get_original_label(self, tokenized_label):
        return self.label_mapping.get(tokenized_label, None)

def detokenize_label(label):
    stack = [label]
    while stack:
        curr = stack.pop()
        if isinstance(curr, tuple):
            curr = tuple(detokenize_label(elem) for elem in curr)
        elif isinstance(curr, list):
            curr = [detokenize_label(elem) for elem in curr]
        elif isinstance(curr, int):
            curr = TOKENIZER.convert_ids_to_tokens([curr])
            curr = TOKENIZER.convert_tokens_to_ids(curr)
            curr = TOKENIZER.decode(curr)#should i dump the curr[0]?
            #looks like the "##" can simply be removed to get back OG english acronyms
            if "##" in curr:
                curr = curr[2:]
    return curr

#TODO: for some reason the list splices arent persistent
#def readable(decoded_string):
#    #first tuple
#    copy = list(decoded_string)
#    copy[0][1][1:-1] = "".join(decoded_string[0][1][1:-1])
#    #connector
#    copy[1][1:-1] = "".join(decoded_string[1][1:-1])

#    #second tuple
#    copy[2][1][1:-1] = "".join(decoded_string[2][1][1:-1])

#    return tuple(copy)


if __name__ == "__main__":
    inference_obj = Inference()
    inference_obj.load_label_mapping()

    #NOTE: technically store label_mappings is unnecessary as I have a decode method now
    #      so the real utility I guess is that I now can create an offline set of mappings
    #      which is kinda redundant because im about to make a train and test file 
    #      (bright side: I got more practice lol)

    sentences = ["I prefer the morning flight to Denver.", "I fight the world."]
    ds = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)

    ### "MODEL OUTPUT"
    sentence, tokenized_label = ds[0]
    tokenized_label = tokenized_label[0]
    valid_tokenized_label = LabelMapping.convert_to_tuple(tokenized_label[0])

    original_label = inference_obj.get_original_label(valid_tokenized_label)

    print(f"Tokenized Label: {tokenized_label}\n")
    print(f"Original Label: {original_label}")

    ### now lets assume that our tokenized label is new

    decoded_string = detokenize_label(tokenized_label)
    #i want to concatenate strings 
    # print(decoded_string[1:-1])

    
    #total output
    print(decoded_string)
    # print(readable(decoded_string))


