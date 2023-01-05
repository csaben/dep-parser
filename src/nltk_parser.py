import nltk
from nltk.parse import CoreNLPParser

# parser = CoreNLPParser(url='http://localhost:9000')

# sentence = "What is the airspeed of an unladen swallow ?"
# b = list(parser.raw_parse(sentence))
# print(b)


'''i think i want to use a neural parser to generate data, make data into some tuple structure
array or dict and then combine into a dataset class to churn out a dataset to train on'''

# Neural Dependency Parser
from nltk.parse.corenlp import CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
sentence = 'I prefer the morning flight through Denver'
# parses = dep_parser.parse('What is the airspeed of an unladen swallow ?'.split())
parses = dep_parser.parse(sentence.split())
s = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
print(s)

#use this dep parse, finding the root is trivially easy if you can read grammar well, so use this
# parse as  a label!!


""" 
TODO

makes dataset class, make dataset, train model, make inference script that works for documents,
make docker image, update README.md, call it a day
"""
