'''

i might need to just write this myself. chat is having a hard time
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.parse.corenlp import CoreNLPDependencyParser

def main():
        # Example list of sentences
    sentences = ['I prefer the morning flight through Denver', 'What is the airspeed of an unladen swallow ?']

    # Create the dataset
    dataset = DependencyParserDataset(sentences)

    print(dataset[0][1])
    # Create a DataLoader to iterate over the dataset
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # # Iterate over the data
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
        label = self.generate_label(sentence)

        # Split the sentence into words
        words = sentence.split()

        # Create a vocabulary that maps words to integers
        vocabulary = {word: i for i, word in enumerate(set(words))}

        # Convert the sentence to a sequence of integers
        sentence_data = [vocabulary[word] for word in words]

        # Convert the sequence to a tensor
        sentence_tensor = torch.tensor(sentence_data, dtype=torch.long)

        return (sentence_tensor, label)

    def generate_label(self, sentence):
        parses = self.dep_parser.parse(sentence.split())
        return [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]

    def clean_label(self, label):
        return [[(governor[0], dep, dependent[0]) for governor, dep, dependent in parse] for parse in label]

def collate_fn(data):
    # Unpack the data into a list of sentences and a list of labels
    sentences, labels = zip(*data)
    # print(sentences, labels)

    # Convert the sentences to tensors
    tensor_sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]

    # Pad the sequences so they are all the same length
    padded_sentences = torch.nn.utils.rnn.pad_sequence(tensor_sentences, batch_first=True)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return (padded_sentences, padded_labels)




if __name__ == '__main__':
    main()
