import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Set serif fonts globally
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = ["Cambria", "Times New Roman", "DejaVu Serif"]
plt.rcParams['font.size'] = 26
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'DejaVu Serif'
plt.rcParams['mathtext.it'] = 'DejaVu Serif:italic'
plt.rcParams['mathtext.bf'] = 'DejaVu Serif:bold'

# Create the DAG
G = nx.DiGraph()

# Add nodes with custom attributes
G.add_node("Age", color="#bbdefb", label="Age\n(x)")
G.add_node("Income", color="#ffecb3", label="Income\n(ŵ)")
G.add_node("Mental Health", color="#e1bee7", label="Mental\nHealth\n(ŷ)")

# Set up the plot
fig, ax = plt.subplots(figsize=(7, 5))
pos = {'Age': [-0.1, -0.3],
       'Income': [0.25, 0.],
       'Mental Health': [1.1, -0.3]}

# Define node properties
node_size = 4000
node_radius = 0.07

# Draw nodes with colors
node_colors = [G.nodes[n]['color'] for n in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=node_size, alpha=0.8, ax=ax)

# Get the custom labels from node attributes
node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
text_items = nx.draw_networkx_labels(G, pos, labels=node_labels, 
                                     font_weight='bold', 
                                     font_family="serif", ax=ax)

# Fixed function to create shortened arrow
def create_arrow(start_pos, end_pos, radius, connection_rad=0.1):
    # Calculate direction vector
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist == 0:
        return None
    
    # Normalize (FIXED: don't subtract 0.15 from dist)
    dx_norm = dx / dist + 0.2
    dy_norm = dy / dist  + 0.2
    
    # Shorten at both ends
    shortened_start = (start_pos[0] + dx_norm * radius, 
                       start_pos[1] + dy_norm * radius)
    
    shortened_end = (end_pos[0] - dx_norm * radius, 
                     end_pos[1] - dy_norm * radius)
    
    # Create curved arrow
    arrow = FancyArrowPatch(shortened_start, shortened_end,
                            arrowstyle='->', 
                            connectionstyle=f'arc3,rad={connection_rad}',
                            color='black',
                            linewidth=2,
                            mutation_scale=30)
    
    return arrow

# Create and add arrows
arrows = []

# Age -> Income
arrow = create_arrow(pos['Age'], pos['Income'], node_radius, 0.1)
if arrow:
    ax.add_patch(arrow)
    arrows.append(arrow)

# Age -> Mental Health  
arrow = create_arrow(pos['Age'], pos['Mental Health'], node_radius, 0.1)
if arrow:
    ax.add_patch(arrow)
    arrows.append(arrow)

# Income -> Mental Health
arrow = create_arrow(pos['Income'], pos['Mental Health'], node_radius, 0.1)
if arrow:
    ax.add_patch(arrow)
    arrows.append(arrow)

# Draw edge labels with explicit serif font
edge_labels = {
    ('Age', 'Income'): ('a', (0.05, -0.15)),
    ('Age', 'Mental Health'): ("c'", (0.4, -0.35)),
    ('Income', 'Mental Health'): ('b', (0.7, -0.16))
}

for (u, v), (label, label_pos) in edge_labels.items():
    ax.text(label_pos[0], label_pos[1], label,
            fontsize=16, 
            fontweight='bold', 
            fontfamily='serif',
            ha='center', 
            va='center')

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.38, 0.05)
ax.axis('off')
plt.tight_layout()
plt.savefig("DAG.png", dpi=600)
plt.show()