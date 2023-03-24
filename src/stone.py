#stone simple regex
import re
import fire
import pandas as pd
import config
import sys

def helper_fn(sentences: str) -> list:
    # regex pattern to find sentences
    pattern = r"[A-Z][^.]*\."

    #replace newline characters with spaces and make into a list of sentences
    def sanitize_string(*args):
        # Replace unwanted characters with spaces and remove leading/trailing spaces
        sanitized_strings = [re.sub(r"\n", " ", arg.strip()) for arg in args]
        return sanitized_strings

    # Find all sentences in the string
    matches = re.findall(pattern, sentences)

    # apply the sanitize function to the matches
    samplefn = sanitize_string(*matches)
    return samplefn

def sample_df_packing(sentences):
    #use importlib.reload() when working in the ipython console
    sentences = helper_fn(sentences)

    #packs the sentences into a dataframe
    df = pd.DataFrame(sentences, columns=['sentence'])
    def custom(sentence):
        parses = config.DEP_PARSER.parse(sentence.split())
        s = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
        return s
        # pp.pprint(s)
        # pp =  pprint.PrettyPrinter(indent=4)

    #for each sentence apply the DEP_PARSER
    df['label'] = df['sentence'].apply(lambda x: custom(x))

    #next unpack your tuples into a set padded allotment 

    #make a tokenized list of the sentences

    #make a tokenized list of the labels

    #slap a txt file with this shit and have a dataset to save and use for training

    return df
