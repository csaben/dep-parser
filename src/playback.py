import fire
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from config import *
import pprint

def sample():
    # Neural Dependency Parser
    sentence = 'I prefer the morning flight through Denver'
    parses = DEP_PARSER.parse(sentence.split())
    s = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
    pp =  pprint.PrettyPrinter(indent=4)
    pp.pprint(s)

def custom(sentence):
    parses = DEP_PARSER.parse(sentence.split())
    s = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
    pp =  pprint.PrettyPrinter(indent=4)
    pp.pprint(s)


if __name__ == '__main__':
    fire.Fire({
        'sample': sample,
        'custom': custom
    })

