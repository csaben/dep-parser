from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logging.set_verbosity(40) #only show errors
import sys

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from config import *
from label_mapping import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Type
from typing import List
import pickle
#!pip install datasets evaluate transformers[sentencepiece]
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import itertools
from transformers import AutoModel

def main():
    sentence = "I prefer the morning flight through denver."
    ds = TestDataset([sentence])
    parsed=DEP_PARSER.parse(sentence)
    collate_fn = Collate(20)
    dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    print(parsed)
    print(ds[0].keys())
    print(ds[0])
    for i, output in enumerate(dataloader):
        print(output)

    #text_dataset = load_dataset("text", data_files=DATA)
    #sentences = text_dataset["train"]['text']
    #sentences = list(filter(None, sentences))
    #ds = TestDataset(sentences)
    ## ids, label = ds[0]
    #collate_fn = Collate(20)
    #dataloader = DataLoader(ds, batch_size=3, collate_fn=collate_fn)
    #model = DepParseModel()
    #optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    #model.train()
    #    # output["ids"] = [sample["ids"] for sample in batch]
    #    # output["mask"] = [sample["mask"] for sample in batch]
    #    # output["targets"] = [sample["targets"] for sample in batch]
    #steps=1e5

    ##REDO THE TRAINING LOOP; The label poses a pretty big problem to meaningful training
    ##TIME FOR WRITEUP
    #for step in range(steps):
    #    for i, output in enumerate(dataloader):
    #        input, mask, label = output["ids"], output["mask"], output["targets"]
    #        optimizer.zero_grad()
    #        outputs = model(input, mask)
    #        loss = model.loss(label, input)
    #        loss.backward()
    #        optimizer.step()
    #        if step % 100 == 0:
    #            print(f"step {step}, Loss: {loss.itm()}")

        # print(i)
        # print(output)
        # print(output['targets'])
        # print(len(output['targets']))

class DepParseModel(nn.Module):
    def __init__(self, num_labels=10, steps_per_epoch=5634//3, model_name=MODEL, num_train_steps=1e5, tokenizer=TOKENIZER, learning_rate=2e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        # = no. of training samples / batch_size
        self.steps_per_epoch = steps_per_epoch
        hidden_dropout_prob: float = 0.0

        self.transformer = AutoModel.from_pretrained(model_name)##, config=config)
        self.dropout = nn.Dropout(0.1)##config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(512, self.num_labels)##config.hidden_size, self.num_labels)

    def loss(self, outputs, targets):
        loss_fct = nn.MSELoss()
        loss = loss_fct(outputs, targets)
        return loss

    def __call__(self, ids, mask, targets=None):
        transformer_out = self.transformer(input_ids=ids, attention_mask=mask)
        sequence_output = transformer_out.last_hidden_state[:, 0, :]
        sequence_output = self.dropout(sequence_output)
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets)
            loss2 = self.loss(logits2, targets)
            loss3 = self.loss(logits3, targets)
            loss4 = self.loss(logits4, targets)
            loss5 = self.loss(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            return logits, loss
        return logits, loss


class Collate:
    def __init__(self, max_len):
        self.max_len = max_len
        self.pad_value = 0

    def __call__(self, batch):
        global output
        output = {}
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["targets"] = [sample["targets"] for sample in batch]

        max_arr = 0
        for tuples in output["targets"]:
            for tupl in tuples:
                # print(tupl)
                # print(tupl[0])#first tuple
                len1 = len(tupl[0][0])
                len2 = len(tupl[0][1])
                # print(tupl[1:-1])#middle arrays
                len3 = len(tupl[1:-1][0])
                # print(tupl[-1])#last tuple
                len4 = len(tupl[-1][0])
                len5 = len(tupl[-1][1])
                # print(len1, len2, len3, len4, len5)
                batch_max = max([len1, len2, len3, len4, len5])
                if max_arr<batch_max:
                    max_arr=batch_max

        #flatten this, when doing inference, to repack into format you must know
        #the batch_max or max_arr in order to iteratively repack

        #pad according to max_arr in batch prior to flattening
        global output_list
        output_list = list(output["targets"])
        for tuples in output_list:
            for i, tupl in enumerate(tuples):
                tupl = list(tupl)
                tupl1= list(tupl[0])
                tupl2= list(tupl[-1])
                tupl1[0] = tupl1[0] + (max_arr-len(tupl1[0])) * [TOKENIZER.pad_token_id]
                tupl1[1] = tupl1[1] + (max_arr-len(tupl1[1])) * [TOKENIZER.pad_token_id]
                mid_arr = tupl[1:-1][0] + (max_arr-len(tupl[1:-1][0])) * [TOKENIZER.pad_token_id]
                tupl2[0] = tupl2[0] + (max_arr-len(tupl2[0])) * [TOKENIZER.pad_token_id]
                tupl2[1] = tupl2[1] + (max_arr-len(tupl2[1])) * [TOKENIZER.pad_token_id]
                test = (tupl1, mid_arr, tupl2)
                flat_tupl1 = [item for sublist in tupl1 for item in sublist]
                flat_tupl2 = [item for sublist in tupl2 for item in sublist]
                test_flat = flat_tupl1+mid_arr+flat_tupl2
                tuples[i] = test_flat
                #to convert back pack your prediction into 

                #try without making into 150 labels in the test flat first
                # output["targets"] = np.reshape(output["targets"], (150,1))
        output["targets"] = tuple(output_list)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.float)

        #might need to set a hard max_arr so I get a max label size for the model (def adjust this
        # framework next time)
        # print(np.shape(output["targets"]))
        # sys.exit()
        #output label is 10(tuple max in dep parse) * max_arr(in batch)

        #hindsight is 20/20 maybe I should have set up my labels to be the specific pos and conn in
        # vocab and somehow had my model output be those 3 per sublabel in a sentence and somehow
        # make meaning out of it (somehow actually would be workable with enough finnagling)

        return output




class TestDataset(Dataset):
   def __init__(self, sentences, tokenizer=TOKENIZER, model=MODEL, dep_parser=DEP_PARSER):
       self.sentences = sentences
       self.tokenizer = tokenizer
       self.model = model
       self.dep_parser = dep_parser
       self.max_subtoken_arr = 5

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
       return curr

   def add_tokens(self, tokens: List[str]):
       self.tokenizer.add_tokens(tokens)
       self.model.resize_token_embeddings(len(self.tokenizer))

   def __getitem__(self, idx):
       """I will be setting an arbitrary lable tuple length of 10"""
       sentence = self.sentences[idx]
       ids = self.tokenizer(sentence, padding=True, truncation=True, max_length=512,
                            return_tensors='pt')
       label = self.dep_parser.parse(sentence.split())
       label = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for
                 parse in label]

       #parse array of super tuples and pad or truncate to get a constant size across all batches
       pad_super_tuple = (([0,0],[0,0]), [0,0],([0,0],[0,0]))
       labels = label[0]
       while len(labels)<MAX_TUPLE:
           labels.append(pad_super_tuple)
       while len(labels)>MAX_TUPLE:
           labels.pop()

       labels[:] = [self.tokenize_label(label) for label in labels]
       input_ids = ids["input_ids"]
       attention_mask = ids["attention_mask"]

       res = {
           "ids":input_ids,
           "mask":attention_mask,
           "targets":labels,
       }
       return res

       #next up, pad contents of every arrary, return it, and then try and smoosh through
       #dataloader

       #potentially a collator function (because we need dynamic collator from HF for each batch)
       #(next comment block)

       #because of the super tuple structure I might need  acustom dynamic collator fn
       #parse each super tuple and tokenize the words with HF use padding but exclude mask info 

       #parse each super tuple and tokenize the special pos1, conn, and pos2 with either same as
       # above or make a custom vocab that has the max_len built in to the integer choices of the
       # labels(i prefer just using HF and adding special tokens using below but we have two ways
       # now)

if __name__ == '__main__':
    main()
