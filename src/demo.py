import matplotlib.pyplot as plt
import networkx as nx

def main():
    sentence = "What is the airspeed of an unladen swallow?"
    dependencies = [("ROOT", (0, 0)), ("cop", (1, 0)), ("nsubj", (2, 0)), ("det", (3, 2)), ("dep", (4, 2)), ("case", (5, 4)), ("det", (6, 4)), ("amod", (7, 4))]
    visualize_dependencies(sentence, dependencies)



# def visualize_dependencies(dependencies, sentence, root_word):
#     graph = nx.DiGraph()
#     graph.add_node(root_word, pos=(0,0))
    
#     for dependency in dependencies:
#         graph.add_node(dependency[2], pos=(1, dependencies.index(dependency)))
#         graph.add_edge(dependency[0], dependency[2])
    
#     pos = nx.get_node_attributes(graph, 'pos')
#     nx.draw(graph, pos)
#     labels = nx.get_edge_attributes(graph, 'weight')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
#     plt.show()


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
