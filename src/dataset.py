import torch
from nltk.parse import CoreNLPParser
from nltk.tree import Tree


def main():
    # Create a list of sentences
    sentences = [
        "What is the airspeed of an unladen swallow?",
        "I ran to the store.",
        "He ate a delicious meal."
    ]

    # Create the dataset
    dataset = DependencyParsingDataset(sentences)

    # Create a data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Train a model
    for input_data, labels in data_loader:
        print(input_data)
        print(labels)
        break
        # i get an error here because of the tree see the chat log to fix it. but tldr fix this
        # class and generate data for the transformer model or treeLSTM

class DependencyParsingDataset(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.parser = CoreNLPParser(url='http://localhost:9000')

    def __getitem__(self, index):
        # Parse the raw string and convert the resulting tree to a list of tuples
        sentence = self.sentences[index]
        tree = list(self.parser.raw_parse(sentence))[0]
        label = self._tree_to_list(tree)

        # Return the original sentence as the input data and the label
        return sentence, label

    def __len__(self):
        return len(self.sentences)

    def _tree_to_list(self, tree):
        """Convert a tree to a list of tuples."""
        if isinstance(tree, Tree):
            if tree.label() == 'ROOT':
                return self._tree_to_list(tree[0])
            else:
                return [(label, self._tree_to_list(child)) for label, child in tree]
        else:
            return tree
 

if __name__ == '__main__':
    main()
