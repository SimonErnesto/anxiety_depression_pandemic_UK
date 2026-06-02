import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import numpy as np

np.random.seed(342)

# 1. Use 'neato' engine with Digraph for ARROWS
dot = graphviz.Digraph('Causal_DAG', engine='neato', format='png')
dot.attr(overlap='false', splines='true', sep='+0.5', pad='0.5')
dot.attr('node', fontname='Helvetica', fontsize='11', margin='0.2')
dot.attr('edge', arrowsize='0.8')

# 2. Define Node Styles
# Exposure (Yellow)
dot.node('Age', 'Age', shape='box', style='filled', fillcolor='#ffffb3', color='#000000')
dot.node('Income', 'Income', shape='box', style='filled', fillcolor='#ffffb3', color='#000000')
# Outcome (Blue)
dot.node('MH', 'Mental Health', shape='box', style='filled', fillcolor='#b3d9ff', color='#000000')
# Confounder + Stratification Covariate (Green) - Sex
dot.node('Sex', 'Sex', shape='box', style='filled', fillcolor='#b3e6b3', color='#000000')
# Other Adjusted Confounders (Gray)
for c in ['Education', 'Ethnicity']:
    dot.node(c, c, shape='box', style='filled', fillcolor='#e6e6e6', color='#000000')

# Latent Variables (Ellipses, white fill)
latents = ['Children', 'Employment', 'Trauma', 'Politics', 'Residence']
for l in latents:
    dot.node(l, l, shape='ellipse', style='solid', fillcolor='white', color='#000000')

# 3. Add Edges
# Main causal path (Thick)
dot.edge('Age', 'Income', color='#000000', penwidth='2')
dot.edge('Income', 'MH', color='#000000', penwidth='2')
dot.edge('Age', 'MH', color='#000000', penwidth='2', style='dashed')

# Confounder edges (Solid black)
for conf in ['Sex', 'Education', 'Ethnicity']:
    for target in ['Age', 'Income', 'MH']:
        dot.edge(conf, target, color='#000000')

# Latent edges (Dotted gray)
latent_edges = [
    ('Age', 'Children'), ('Age', 'Employment'), ('Age', 'Trauma'), ('Age', 'Politics'), ('Age', 'Residence'),
    ('Sex', 'Children'), ('Sex', 'Employment'), ('Sex', 'Trauma'), ('Sex', 'Politics'), ('Sex', 'Residence'),
    ('Education', 'Children'), ('Education', 'Employment'), ('Education', 'Politics'), ('Education', 'Residence'),
    ('Ethnicity', 'Children'), ('Ethnicity', 'Employment'), ('Ethnicity', 'Trauma'), ('Ethnicity', 'Politics'), ('Ethnicity', 'Residence'),
    ('Children', 'MH'), ('Children', 'Residence'),
    ('Employment', 'MH'), ('Employment', 'Children'),
    ('Trauma', 'MH'), ('Trauma', 'Residence'),
    ('Politics', 'MH'),
    ('Residence', 'MH')
]

for u, v in latent_edges:
    dot.edge(u, v, color='#666666', style='dotted', penwidth='1.5')

# 4. Render the graphviz DAG (without legend)
dot.render('DAG_extended', format='png', cleanup=True)

# 5. Load the rendered image and add matplotlib legend
fig, ax = plt.subplots(figsize=(14, 10))
img = mpimg.imread('DAG_extended.png')
ax.imshow(img)
ax.axis('off')  # Hide axes

# Create legend handles
legend_elements = [
    # Node shapes
    mpatches.Patch(facecolor='#ffffb3', edgecolor='#000000', label='Exposure variables of interest\n(mediation)'),
    mpatches.Patch(facecolor='#b3d9ff', edgecolor='#000000', label='Outcome variable\n(PHQ-9 / GAD-7 Scores)'),
    mpatches.Patch(facecolor='#b3e6b3', edgecolor='#000000', label='Confounder and covariate\n(stratified)'),
    mpatches.Patch(facecolor='#e6e6e6', edgecolor='#000000', label='Confounders and nuisance'),
    mpatches.Ellipse((0, 0), 0.3, 0.2, facecolor='white', edgecolor='#000000', 
                     label='Unnecessary for minimal\nadjustment set'),
    # Edge styles
    Line2D([0], [0], color='#000000', linewidth=2, label='Causal relationship'),
    Line2D([0], [0], color='#000000', linewidth=2, linestyle='--', label='Direct effect'),
    Line2D([0], [0], color='#666666', linewidth=1.5, linestyle=':', 
           label='Relationship does not open\nback-door path')
]

# Add legend
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
          frameon=True, fancybox=True, shadow=True, title='Legend', 
          title_fontsize=11, edgecolor='black')

plt.tight_layout()
plt.savefig('DAG_extended.png', dpi=300, bbox_inches='tight')
plt.savefig('DAG_extended.pdf', dpi=300, bbox_inches='tight')
plt.show()


########### Dagitty DAG ####################
'''
dag {
"Mental Health" [outcome,pos="-0.178,0.035"]
Age [exposure,pos="-1.672,1.494"]
Children [latent,pos="1.407,0.431"]
Education [adjusted,pos="-1.666,-1.430"]
Employment [latent,pos="0.853,-1.336"]
Ethnicity [adjusted,pos="-1.053,-1.638"]
Income [exposure,pos="-0.998,0.916"]
Trauma [latent,pos="1.300,-0.241"]
Politics [latent,pos="1.183,-0.841"]
Residence [latent,pos="1.381,1.119"]
Sex [adjusted,pos="-2.071,-0.757"]
Age -> "Mental Health"
Age -> Children
Age -> Employment
Age -> Income
Age -> Trauma
Age -> Politics
Age -> Residence
Children -> "Mental Health"
Children -> Residence
Education -> "Mental Health"
Education -> Age
Education -> Children
Education -> Employment
Education -> Income
Education -> Politics
Education -> Residence
Employment -> "Mental Health"
Employment -> Children
Ethnicity -> "Mental Health"
Ethnicity -> Children
Ethnicity -> Education
Ethnicity -> Employment
Ethnicity -> Income
Ethnicity -> Trauma
Ethnicity -> Politics
Ethnicity -> Residence
Ethnicity -> Sex
Income -> "Mental Health"
Income -> Children
Income -> Trauma
Income -> Politics
Trauma -> "Mental Health"
Politics -> "Mental Health"
Residence -> "Mental Health"
Residence -> Trauma
Residence -> Politics
Sex -> "Mental Health"
Sex -> Age
Sex -> Children
Sex -> Education
Sex -> Employment
Sex -> Income
Sex -> Trauma
Sex -> Politics
}
'''

