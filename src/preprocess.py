import pandas as pd
from config import *
"""
[   [   (('prefer', 'VBP'), 'nsubj', ('I', 'PRP')),
        (('prefer', 'VBP'), 'dobj', ('flight', 'NN')),
        (('flight', 'NN'), 'det', ('the', 'DT')),
        (('flight', 'NN'), 'compound', ('morning', 'NN')),
        (('prefer', 'VBP'), 'nmod', ('Denver', 'NNP')),
        (('Denver', 'NNP'), 'case', ('through', 'IN'))
    ]     ]
"""

# Steps:
# 1. lazy load the document into RAM (how to lazy load? save this for a pandas dataframe later)
# 2. parse given section of document into a list of sentences w/ some rule based algorithm
#    rule should include truncating sentences that are too long

#make parser class for just getting all text of a file into a csv/df
def naive_parser(path: str):
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line)

    #make a naive rule, if there isn't a period in th next
    #three lines don't include it in the sentence
    print(len(sentences))
    for idx, sentence in enumerate(sentences):
        if sentences[idx+1:idx+3].count('.') == 0:
            #remove the current sentence
            sentences.pop(idx)

    #combine elements [first after a period, sentence with period] in one sentence
    #use the fact we know every 3 lines is a sentence
    new_sentences = []
    for idx, sentence in enumerate(sentences):
        if idx % 3 == 0:
            new_sentences.append(sentence)
        else:
            new_sentences[-1] += sentence

    #remove blank lines
    new_sentences = [sentence for sentence in new_sentences if sentence != '']

    #remove all words with caps>1
    new_sentences = [sentence for sentence in new_sentences if sum(1 for c in sentence if
                                                                   c.isupper()) < 10]


    #remove new line characters
    new_sentences = [sentence.replace('\n', '') for sentence in new_sentences]

    #generate labels
    labels = []
    for sentence in new_sentences:
        raw_parse = DEP_PARSER.parse(new_sentences)
        #[[  (('prefer', 'VBP'), 'nsubj', ('I', 'PRP')),
        #    (('prefer', 'VBP'), 'dobj', ('flight', 'NN')) ]]
        #unpack raw parse of N relations into Nx5 array of strings
        for line in raw_parse.to_conll().split('\n'):
            if line:
                labels.append(line.split('\t'))
                print(labels)
        """
        try this:

            from stanfordnlp.server immport CoreNLPClient
from stanfordnlp import DependencyGraph

# Start the CoreNLP server and create a client
with CoreNLPClient(annotators=['depparse'], timeout=30000, memory='16G') as client:
    # Process a sentence and get the dependency graph
    ann = client.annotate("John likes Mary.")
    dep_graph = DependencyGraph(ann.sentence[0].parseTree.dependency)

    # Convert the dependency graph to a list
    dep_list = []
    for line in dep_graph.to_conll().split('\n'):
        if line:
            fields = line.split('\t')
            dep_list.append(fields)
            
    print(dep_list)




        """


        # new_parse = []
        # raw_parse = list(next(raw_parse))
        # for relation in raw_parse:
        #     new_parse.append([relation[0][0], relation[0][1], relation[1], relation[2][0], relation[2][1]])
        # labels.append(new_parse)

    #make a dataframe
    result = (new_sentences, labels)
    df = pd.DataFrame(result, columns=['sentence', 'label'])
    
    return df

#make a regex parser for this idea using the following inspiration (for now continue):
    #https://stackoverflow.com/questions/47982949/how-to-parse-complex-text-files-using-python/47984221#47984221:~:text=the%20column%20names).-,The%20code%3A,-import%20pandas%20as
    #https://www.vipinajayakumar.com/parsing-text-with-python/

        
# 3. function for using dependency parser on a sentence and 
#    tokenizing the sentence into a list of tokens.
sentences = naive_parser(DATA)
# 4. determine label format when unpacking prior to tokenizing to be used (research)
# 5. find a reasonable max label length and truncate labels that are too long (height and width
#    wise as a tensor)

# 6. subsequent build step for storing data to be loaded while training (best format?)
# 7. train an attention network with no causal masking on the data
