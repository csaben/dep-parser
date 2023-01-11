from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logging.set_verbosity(40) #only show errors

from transformers import BertTokenizer, BertModel
from nltk.parse.corenlp import CoreNLPDependencyParser

DATA = "../input/pdf_txt/china_military_report.txt"
TRAIN = ""
TEST = ""
MODEL = BertModel.from_pretrained("bert-base-cased")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
DEP_PARSER = CoreNLPDependencyParser(url='http://localhost:9000')
