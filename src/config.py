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
MAX_LEN=512 #was 98 for document but we will use a mask anyway
MAX_TUPLE=10

tester = """OFFICE OF THE SECRETARY OF DEFENSE
Annual Report to Congress:  Military  and Security Developments Involving the People’s Republic of
China. This is what a sentence looks like. This is more than one sentence. this is THIS 2222 HH iS
NONESENCE."""
