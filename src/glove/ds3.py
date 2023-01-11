import torch
from torch.utils.data import Dataset
import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

#man chat is doing a shite job. but its giving me an idea of how to do it myself

def main():
    # Create dataset
    sentences = ["What is the airspeed of an unladen swallow?", "I prefer the morning flight through Denver"]
    dataset = DependencyParseDataset(sentences)

    # Create data loader
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=lambda x: (pad_sequence([torch.tensor(d[0]) for d in x], padding_value=0), pad_sequence([torch.tensor(d[1]) for d in x], padding_value=0)))

    # Iterate through data
    for input_data, labels in data_loader:
        print(input_data)
        print(labels)



class DependencyParseDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        parse = self.dep_parser.parse(sentence.split())
        label = self._tree_to_list(parse)
        return sentence, label
    
    def _tree_to_list(self, parse):
        """Convert parse tree to list of integers"""
        result = []
        for tree in parse:
            for governor, dep, dependent in tree.triples():
                result.append(governor[0])
                result.append(dep)
                result.append(dependent[0])
        return result


if __name__ == '__main__':
    main()
