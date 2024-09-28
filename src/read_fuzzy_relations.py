import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt

# Set project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set file paths
fuzzy_relations_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations.pkl')
vocab_path = os.path.join(project_root, 'data', 'processed', 'vocab.pkl')

# Load fuzzy relations
with open(fuzzy_relations_path, 'rb') as f:
    fuzzy_relations = pickle.load(f)

# Load vocabulary mapping
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']

# Build NetworkX graph
G = nx.Graph()

# Iterate through fuzzy relations and add nodes and edges to the graph
for target_idx, related_words in list(fuzzy_relations.items())[:10]:  # Show only first 10 words
    target_word = idx2word[target_idx]
    G.add_node(target_word)  # Add nodes
    for related_idx, degree in related_words.items():
        related_word = idx2word[related_idx]
        G.add_edge(target_word, related_word, weight=degree)  # Add edges with weights

# Set the appearance of nodes and edges
pos = nx.spring_layout(G)  # Use spring layout for graph arrangement
weights = [G[u][v]['weight'] for u, v in G.edges()]  # Get the edge weights

# Create subplots to manage colorbar and plot layout
fig, ax = plt.subplots()

# Draw edges with color and width based on weights
edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color=weights, width=[w * 2 for w in weights], edge_cmap=plt.cm.Blues)

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_family='sans-serif')

# Add colorbar to indicate membership degree (weights)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
sm.set_array(weights)
plt.colorbar(sm, ax=ax, label='Membership Degree')  # Specify colorbar axis

# Save and display the graph
output_image_path = os.path.join(project_root, 'data', 'processed', 'fuzzy_relations_graph.png')
plt.title("Fuzzy Relations Graph")
plt.savefig(output_image_path)
plt.show()

print(f"Fuzzy relations graph has been saved as {output_image_path}")

