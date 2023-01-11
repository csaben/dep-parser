from transformers import BertTokenizer, BertModel
from nltk.parse.corenlp import CoreNLPDependencyParser

DATA = ""
MODEL = BertModel.from_pretrained("bert-base-cased")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
DEP_PARSER = CoreNLPDependencyParser(url='http://localhost:9000')
