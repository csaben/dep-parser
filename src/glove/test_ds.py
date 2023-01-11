'''

i might need to just write this myself. chat is having a hard time
'''
import torch
import torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.parse.corenlp import CoreNLPDependencyParser

def main():
        # Example list of sentences
    sentences = ['I prefer the morning flight through Denver', 'What is the airspeed of an unladen swallow ?']

    # Create the dataset
    dataset = DependencyParserDataset(sentences)

    # dataset[0]
    # print(dataset[0][1])
    # Create a DataLoader to iterate over the dataset
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # # # Iterate over the data
    for (sentence, label) in dataloader:
    #     # Clean up the formatting of the label
    #     label = dataset.clean_label(label)
        print(sentence)
        print(label)


class DependencyParserDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]

        # print("sentence\n", sentence)
        label = self.generate_label(sentence)
        # print("\nlabel\n",label)

        # Split the sentence into words
        words = sentence.split()

        # Create a vocabulary that maps words to integers
        vocabulary = {word: i for i, word in enumerate(set(words))}

        # Convert the sentence to a sequence of integers
        sentence_data = [vocabulary[word] for word in words]

        # Convert the sequence to a tensor
        sentence_tensor = torch.tensor(sentence_data, dtype=torch.long)

        return (sentence_tensor, label)

    def from_tuples(self, tuples):
        # Initialize an empty list of tokens
        tokens = []
        counter = 0
        
        # Iterate over the tuples
        for i, tup in enumerate(tuples):
            # Extract the token, part-of-speech, and dependency index from the tuple
            # token, pos, dep = tup[0][0], tup[0][1], i+1
            token, pos, dep = tup[0][0], tup[0][1], tup[2][0]
            # Create a dictionary for the token and add it to the list of tokens
            tokens.append({'token': token, 'pos': pos, 'dep': dep, 'key': counter})
            #we can ignore counter in our training and conversion back to human readable
            counter+=1
        
        return tokens

    def generate_label(self, sentence):
        parses = self.dep_parser.parse(sentence.split())
        prelim = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
        # tokenized = nltk.word_tokenize(prelim[0])
        embeddings = torchtext.vocab.GloVe(name='6B', dim=100)
        tokens = prelim[0]
        #conversion function here
        tokens = self.from_tuples(tokens)
        print(tokens)
        embedding_vectors = [embeddings.vectors[embeddings.stoi[token['token']]] for token in
                             tokens]
        # print(embedding_vectors)

        '''
        [(('word', 'nsub'), 'connection', (('word','nn')), ...] is a terrible label bro. adjust
        your preprocessing. see the tmp file to continue. need to get dinner
        '''
        tokenized_label = embeddings.stoi(prelim)
        return embeddings.vector[tokenized_label]

    # def clean_label(self, label):
    #     return [[(governor[0], dep, dependent[0]) for governor, dep, dependent in parse] for parse in label]

def collate_fn(data):
    # Unpack the data into a list of sentences and a list of labels
    sentences, labels = zip(*data)
    # print(sentences, labels)

    # Convert the sentences to tensors
    tensor_sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
    # tensor_sentences = tensor_sentences[0]

    # Pad the sequences so they are all the same length
    padded_sentences = torch.nn.utils.rnn.pad_sequence(tensor_sentences, batch_first=True)
    print(labels)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)


    return (padded_sentences, padded_labels)




if __name__ == '__main__':
    main()
