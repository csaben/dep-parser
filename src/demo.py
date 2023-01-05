import matplotlib.pyplot as plt
import networkx as nx

def main():
    sentence = "What is the airspeed of an unladen swallow?"
    dependencies = [("ROOT", (0, 0)), ("cop", (1, 0)), ("nsubj", (2, 0)), ("det", (3, 2)), ("dep", (4, 2)), ("case", (5, 4)), ("det", (6, 4)), ("amod", (7, 4))]
    # visualize_dependencies(sentence, dependencies)
    vis2(dependencies)


def vis2(dependencies):
    # Build the graph using the dependencies
    G = nx.Graph()
    for dependency in dependencies:
        G.add_edge(dependency[0][0], dependency[1][0], label=dependency[0][1])

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    # Add labels to the edges
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Save the image
    plt.savefig('../output/dependencies2.png')

def visualize_dependencies(sentence, dependencies):
    # Create a directed graph
    graph = nx.DiGraph()
    
    # Add nodes and edges to the graph
    for i, word in enumerate(sentence):
        graph.add_node(i, label=word)
    for label, (dependent, head) in dependencies:
        graph.add_edge(head, dependent, label=label)
    
    # Use the spring layout for the graph
    pos = nx.spring_layout(graph)
    
    # Draw the graph and save it to an image
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, arrows=True)
    plt.savefig("../output/dependencies.png")
    """
    ideally
    visualize_dependencies(dependencies, sentence, root_word)

    saves an image so we can easily look at a sentence, then scale it to work for a whole document

    after this we need to make the model, make the dataset, and make the docker image to use
    """
    


if __name__ == '__main__':
    main()
