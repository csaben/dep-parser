import matplotlib.pyplot as plt
import networkx as nx
from nltk.parse import CoreNLPParser
from nltk.tree import Tree

def visualize_dependencies(sentence):
    # Create a parser
    parser = CoreNLPParser(url='http://localhost:9000')

    # Parse the sentence
    tree = list(parser.raw_parse(sentence))[0]

    # Build the graph using the dependencies
    G = nx.Graph()
    for subtree in tree:
        if isinstance(subtree, Tree):
            head = subtree.label()
            for i, child in enumerate(subtree):
                if isinstance(child, Tree):
                    dependent = child.label()
                    G.add_edge(head, dependent)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    # Save the image
    plt.savefig('../output/dependencies6.png')

# Example usage
visualize_dependencies("I prefer the morning flight through Denver")
