import nltk
from nltk.parse import CoreNLPParser

parser = CoreNLPParser(url='http://localhost:9000')

sentence = "What is the airspeed of an unladen swallow ?"
b = list(parser.raw_parse(sentence))
print(b)

