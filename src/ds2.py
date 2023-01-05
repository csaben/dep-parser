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
    for i, j in data_loader:
        print(i)

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
        """Recursively convert a tree to a list of tuples, where each tuple
        consists of a label and a depth.
        """
        # Base case: tree is a leaf node
        if len(tree) == 1:
            return [(tree.label(), 0)]

        # Recursive case: tree is a non-leaf node
        label = tree.label()
        children = [self._tree_to_list(child) for _, child in tree]
        max_len = max(len(c) for c in children)
        padded_children = [c + [("PAD", 0)] * (max_len - len(c)) for c in children]
        return [(label, 0)] + [item for child in padded_children for item in child]


if __name__ == '__main__':
    main()
