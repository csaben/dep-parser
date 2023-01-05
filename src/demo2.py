import matplotlib.pyplot as plt
import networkx as nx

def main():
    sentence = "What is the airspeed of an unladen swallow?"
    dependencies = [("ROOT", (0, 0)), ("cop", (1, 0)), ("nsubj", (2, 0)), ("det", (3, 2)), ("dep", (4, 2)), ("case", (5, 4)), ("det", (6, 4)), ("amod", (7, 4))]
    visualize_dependencies(dependencies, sentence)

def visualize_dependencies(dependencies, sentence):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes and edges to the graph
    for i, word in enumerate(sentence.split()):
        graph.add_node(i, label=word)
    for label, (dependent, head) in dependencies:
        graph.add_edge(head, dependent, label=label)

    # Use the spring layout for the graph
    pos = nx.spring_layout(graph)

    # Draw the graph and save it to an image
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, arrows=True)
    plt.savefig("../output/dependencies3.png")

if __name__ == '__main__':
    main()
