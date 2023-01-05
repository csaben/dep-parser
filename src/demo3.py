import matplotlib.pyplot as plt
import networkx as nx

def main():
    sentence = "What is the airspeed of an unladen swallow?"
    dependencies = [("ROOT", (0, 0)), ("cop", (1, 0)), ("nsubj", (2, 0)), ("det", (3, 2)), ("dep", (4, 2)), ("case", (5, 4)), ("det", (6, 4)), ("amod", (7, 4))]
    visualize_dependencies(sentence, dependencies)

def visualize_dependencies(sentence, dependencies):
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
    nx.draw(graph, pos, with_labels=True, arrows=True, node_size=3000, font_size=20, font_weight='bold', node_color='white')
    labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_size=20)
    plt.savefig("../output/dependencies3.png")

if __name__ == '__main__':
    main()
