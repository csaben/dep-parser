#!/usr/bin/env python

from label_mapping import *
from dataset import *

def main():
    label_mapping_obj = LabelMapping()
    #TODO:when I use actual files I might just load path and have a fn for parsing files
    #     into list of sentences to then be tokenized
    sentences = ["I prefer the morning flight to Denver.", "I fight the world."]
    ds = DependencyParserDataset(sentences, TOKENIZER, MODEL, DEP_PARSER)

    for i in range(len(ds)):
        sentence, tokenized_label = ds[i]
        original_label = ds.original_labels[i]
        #TODO: each time I run script we should make a new label mapping directory and a 
        #      log of which run belongs to each dir to decrease the recall time on finding
        #      the corresponding original sentence
        output_path = f'../output/label_mappings/label_mapping_{i}.pkl'
        label_mapping_obj.add_mapping(original_label, tokenized_label)
        label_mapping_obj.save(output_path)

        #not sure why i was doing this, save should only be in label_mappings class
        # ds.save_labels(output_path)






if __name__ == '__main__':
    main()
