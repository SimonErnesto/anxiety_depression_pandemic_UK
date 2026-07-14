
import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import numpy as np

np.random.seed(342)

## Dagitty DAG Mediator (Adjustment (direct effect))
'''
dag {
Age [exposure,pos="-1.672,1.494"]
Children [adjusted,pos="0.531,0.187"]
Education [adjusted,pos="-1.777,-1.351"]
Employment [adjusted,pos="0.809,-1.771"]
Gender [exposure,pos="-2.106,-0.207"]
Housing [pos="0.868,0.788"]
Income [outcome,pos="-1.508,0.572"]
Loneliness [latent,pos="-0.252,-1.761"]
Politics [latent,pos="1.183,-0.841"]
Religion [latent,pos="1.198,-1.642"]
Age -> Children
Age -> Education
Age -> Employment
Age -> Housing
Age -> Income
Age -> Loneliness
Age -> Politics
Children -> Income
Children -> Loneliness
Education -> Children
Education -> Employment
Education -> Income
Education -> Politics
Employment -> Housing
Gender -> Age
Gender -> Children
Gender -> Education
Gender -> Employment
Gender -> Income
Gender -> Loneliness
Gender -> Politics
Income -> Housing
Income -> Politics
Income -> Religion
Religion -> Politics
}

'''


## Dagitty DAG Outcome (Adjustment (direct effect))
'''
dag {
"Mental Health" [outcome,pos="-0.326,-0.104"]
Age [exposure,pos="-1.570,1.323"]
Children [adjusted,pos="0.659,1.250"]
Education [adjusted,pos="-1.777,-1.351"]
Employment [adjusted,pos="0.197,-1.995"]
Gender [exposure,pos="-2.106,-0.207"]
Housing [adjusted,pos="-0.590,-2.026"]
Income [exposure,pos="-0.792,-0.944"]
Loneliness [adjusted,pos="1.206,-0.375"]
Ownership [adjusted,pos="0.579,-1.955"]
Politics [adjusted,pos="1.338,-1.070"]
Religion [adjusted,pos="1.198,-1.642"]
Age -> "Mental Health"
Age -> Children
Age -> Education
Age -> Employment
Age -> Housing
Age -> Income
Age -> Loneliness
Age -> Ownership
Age -> Politics
Children -> "Mental Health"
Children -> Income
Children -> Loneliness
Education -> "Mental Health"
Education -> Children
Education -> Employment
Education -> Income
Education -> Politics
Employment -> "Mental Health"
Employment -> Housing
Employment -> Income
Gender -> "Mental Health"
Gender -> Age
Gender -> Children
Gender -> Education
Gender -> Employment
Gender -> Income
Gender -> Loneliness
Gender -> Politics
Housing -> "Mental Health"
Income -> "Mental Health"
Income -> Housing
Income -> Ownership
Income -> Politics
Income -> Religion
Loneliness -> "Mental Health"
Ownership -> "Mental Health"
Politics -> "Mental Health"
Religion -> "Mental Health"
Religion -> Politics
}


'''


np.random.seed(342)

def build_mediator_dag():
    """Builds the DAG for the Income (Mediator) Model"""
    dot = graphviz.Digraph('Mediator_DAG', engine='neato', format='png')
    dot.attr(overlap='false', splines='true', sep='+0.5', pad='0.5')
    dot.attr('node', fontname='Arial', fontsize='12', margin='0.2')
    dot.attr('edge', arrowsize='0.8', dpi="600")

    # 1. Nodes (Adjusted for Mediator Model)
    # Exposures
    dot.node('Age', 'Age', shape='box', style='filled', fillcolor='#ffffb3', color='#000000')
    dot.node('Gender', 'Gender', shape='box', style='filled', fillcolor='#b3e6b3', color='#000000')
    
    # Outcome of THIS specific model (Income)
    dot.node('Income', 'Income', shape='box', style='filled', fillcolor='#b3d9ff', color='#000000')
    
    # Adjusted Confounders (Gray boxes)
    for c in ['Children', 'Education', 'Employment']:
        dot.node(c, c, shape='box', style='filled', fillcolor='#e6e6e6', color='#000000')
        
    # Latent Variables (White ellipses)
    for l in ['Loneliness', 'Politics', 'Religion', 'Housing']:
        dot.node(l, l, shape='ellipse', style='filled', fillcolor='#ffffff', color='#000000')

    # 2. Edges
    # Main causal paths (Thick solid)
    dot.edge('Age', 'Income', color='#000000', penwidth='2')
    dot.edge('Gender', 'Income', color='#000000', penwidth='1.5')
    dot.edge('Gender', 'Age', color='#000000', penwidth='1.5')
    
    # Adjusted confounder paths (Thin solid)
    for c in ['Children', 'Education', 'Employment']:
        dot.edge(c, 'Income', color='#000000')

    # Latent / Nuisance paths (Dotted gray)
    latent_edges = [
        ('Age', 'Children'), ('Age', 'Education'), ('Age', 'Employment'), 
        ('Age', 'Loneliness'), ('Age', 'Politics'),
        ('Children', 'Loneliness'), ('Education', 'Children'), ('Education', 'Employment'),
        ('Education', 'Politics'), ('Income', 'Politics'), ('Income', 'Religion'),
        ('Religion', 'Politics'), ('Gender', 'Children'), ('Gender', 'Education'), 
        ('Gender', 'Employment'), ('Gender', 'Loneliness'), ('Gender', 'Politics'),
        ('Age', 'Housing'), ('Income', 'Housing'), ('Employment', 'Housing')
    ]
    for u, v in latent_edges:
        dot.edge(u, v, color='#666666', style='dotted', penwidth='1.5')

    dot.render('DAG_Mediator', cleanup=True)
    return dot

def build_outcome_dag():
    """Builds the DAG for the Mental Health (Outcome) Model"""
    dot = graphviz.Digraph('Outcome_DAG', engine='neato', format='png')
    dot.attr(overlap='false', splines='true', sep='+0.5', pad='0.5')
    dot.attr('node', fontname='Arial', fontsize='12', margin='0.2')
    dot.attr('edge', arrowsize='0.8', dpi="600")

    # 1. Nodes (Adjusted for Outcome Model)
    # Exposures
    dot.node('Age', 'Age', shape='box', style='filled', fillcolor='#ffffb3', color='#000000')
    dot.node('Gender', 'Gender', shape='box', style='filled', fillcolor='#b3e6b3', color='#000000')
    dot.node('Income', 'Income', shape='box', style='filled', fillcolor='#ffb347', color='#000000') # Orange for Mediator
    
    # Outcome of THIS specific model (Mental Health)
    dot.node('MH', 'Mental Health', shape='box', style='filled', fillcolor='#b3d9ff', color='#000000')
    
    # Adjusted Confounders (Gray boxes - NOTE: Loneliness, Politics, Religion are now adjusted!)
    for c in ['Children', 'Education', 'Employment', 'Loneliness', 
              'Politics', 'Religion', 'Housing']:
        dot.node(c, c, shape='box', style='filled', fillcolor='#e6e6e6', color='#000000')

    # 2. Edges
    # Main causal paths (Thick solid)
    dot.edge('Age', 'Income', color='#000000', penwidth='2')
    dot.edge('Income', 'MH', color='#000000', penwidth='2')
    dot.edge('Gender', 'MH', color='#000000', penwidth='1.5')
    dot.edge('Gender', 'Income', color='#000000', penwidth='1.5')
    dot.edge('Gender', 'Age', color='#000000', penwidth='1.5')
    dot.edge('Age', 'MH', color='#000000', penwidth='2', style='dashed') # Direct effect

    # Adjusted confounder paths to MH (Thin solid)
    for c in ['Children', 'Education', 'Employment', 'Loneliness', 
              'Politics', 'Religion', 'Housing']:
        dot.edge(c, 'MH', color='#000000')

    # Nuisance / Inter-confounder paths (Dotted gray)
    nuisance_edges = [
        ('Age', 'Children'), ('Age', 'Education'), ('Age', 'Employment'), 
        ('Age', 'Loneliness'), ('Age', 'Politics'),
        ('Children', 'Income'), ('Children', 'Loneliness'), 
        ('Education', 'Children'), ('Education', 'Employment'), ('Education', 'Income'), 
        ('Education', 'Politics'),
        ('Employment', 'Income'),
        ('Income', 'Politics'), ('Income', 'Religion'),
        ('Religion', 'Politics'),
        ('Gender', 'Children'), ('Gender', 'Education'), ('Gender', 'Employment'), 
        ('Gender', 'Loneliness'), ('Gender', 'Politics'),
        ('Income', 'Housing'), ('Age', 'Housing'), ('Employment', 'Housing') 
    ]
    for u, v in nuisance_edges:
        dot.edge(u, v, color='#666666', style='dotted', penwidth='1.5')

    dot.render('DAG_Outcome', cleanup=True)
    return dot

# --- Generate and Plot ---
build_mediator_dag()
build_outcome_dag()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot Mediator DAG
img1 = mpimg.imread('DAG_Mediator.png')
axes[0].imshow(img1)
axes[0].set_title("Model 1: Mediator (Income)", fontsize=10, fontweight='bold', pad=20)
axes[0].axis('off')

# Plot Outcome DAG
img2 = mpimg.imread('DAG_Outcome.png')
axes[1].imshow(img2)
axes[1].set_title("Model 2: Outcome (Mental Health)", fontsize=10, fontweight='bold', pad=20)
axes[1].axis('off')

# --- Unified Legend ---
legend_elements = [
    mpatches.Patch(facecolor='#ffffb3', edgecolor='#000000', label='Primary Exposure (Age)'),
    mpatches.Patch(facecolor='#b3e6b3', edgecolor='#000000', label='Stratification Covariate (Gender)'),
    mpatches.Patch(facecolor='#ffb347', edgecolor='#000000', label='Mediator (Income)'),
    mpatches.Patch(facecolor='#b3d9ff', edgecolor='#000000', label='Model Outcome'),
    mpatches.Patch(facecolor='#e6e6e6', edgecolor='#000000', label='Adjusted Confounder'),
    mpatches.Ellipse((0, 0), 0.3, 0.2, facecolor='white', edgecolor='#000000', label='Latent / Unmeasured'),
    
    Line2D([0], [0], color='#000000', linewidth=2, label='Main causal path'),
    Line2D([0], [0], color='#000000', linewidth=2, linestyle='--', label='Direct effect (Age -> MH)'),
    Line2D([0], [0], color='#666666', linewidth=1.5, linestyle=':', label='Nuisance / Latent path')
]

fig.legend(handles=legend_elements, loc='lower center', fontsize=8, 
           ncol=4, frameon=True, fancybox=True, shadow=True, 
           edgecolor='black',
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout(rect=[0, 0.08, 1, 1]) # Make room for the bottom legend
plt.savefig('DAG_Two_Models_Final.png', dpi=600, bbox_inches='tight')
plt.savefig('DAG_Two_Models_Final.pdf', dpi=300, bbox_inches='tight')
plt.savefig('DAG_Two_Models_Final.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.show()