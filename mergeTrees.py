import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

"""
Helper functions
"""

def get_critical_values(G,f):

    critical_values = []
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if sum(np.array([f[nb] for nb in neighbors]) >= f[node]) > 1 or sum(np.array([f[nb] for nb in neighbors]) <= f[node]) > 1:
            critical_values.append(f[node])

    return sorted(list(set(critical_values)))

def get_equivalence_classes(G,f,height):

    nodes_in_sublevel_set = [node for node in G.nodes() if f[node] <= height]
    subgraph = G.subgraph(nodes_in_sublevel_set)

    return [conn for conn in nx.connected_components(subgraph)]

"""
Main merge tree function 
"""

def get_merge_tree(G,f):

    critical_values = get_critical_values(G,f)

    # Initialize the graph at the first level
    T = nx.Graph()

    critical_value = critical_values[0]

    equiv_classes = get_equivalence_classes(G,f,critical_value)

    nodes = list(range(len(equiv_classes)))

    for k in range(len(equiv_classes)):
        T.add_node(k,height = critical_value,subset = equiv_classes[k])

    num_nodes = len(T)

    # Fill in subsequent layers recursively
    for j in range(1,len(critical_values)):
        
        critical_value = critical_values[j]

        equiv_classes = get_equivalence_classes(G,f,critical_value)
        nodesNew = list(range(num_nodes,num_nodes+len(equiv_classes)))

        for k in range(len(equiv_classes)):
            
            T.add_node(k + num_nodes,height = critical_value,subset = equiv_classes[k])
        
            # for node in nodes:
            #     if T.nodes[node]['subset'] <= equiv_classes[k]:
            #         T.add_edge(node,k+num_nodes)

        for node in nodes:
            for k in range(len(equiv_classes)):
                if T.nodes[node]['subset'] <= equiv_classes[k]:
                    T.add_edge(node,k+num_nodes)
                    break

        nodes = nodesNew
        num_nodes = len(T)
#         if len(equiv_classes) == 1:
#             break

    return T

def simplify_merge_tree(T):
    
    heights = {node:T.nodes[node]['height'] for node in T.nodes()}
    root = get_key(heights,max(list(heights.values())))[0]
    
    TNew = T.copy()

    for node in T.nodes():
        if TNew.degree(node) == 2 and node != root:
            neighbors = [n for n in TNew.neighbors(node)]
            TNew.remove_node(node)
            TNew.add_edge(neighbors[0],neighbors[1])     
            
    return TNew
        
"""
Visualization
"""

def get_key(dictionary, val):

    # Given a dictionary and a value, returns list of keys with that value
    res = []
    for key, value in dictionary.items():
         if val == value:
            res.append(key)

    return res
    
def mergeTree_pos(G, height, root=None, width=1.0, xcenter = 0.5):

    '''
    Adapted from Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    height: dictionary {node:height} of heights for the vertices of G.
            Must satisfy merge tree conditions, but this is not checked in this version of the function.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    xcenter: horizontal location of root
    '''
    
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    height_vals = list(height.values())
    max_height = max(height_vals)

    root = get_key(height,max_height)[0]
    # The root for the tree is the vertex with maximum height value

    vert_loc = max_height

    def _hierarchy_pos(G, root, vert_loc, width=1., xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)

        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                vert_loc = height[child]
                pos = _hierarchy_pos(G, child, vert_loc, width = dx, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, vert_loc, width, xcenter)

def draw_merge_tree_V1(G,height,axes=False,style = 'no_nodes'):
    # Input: merge tree as G, height
    # Output: draws the merge tree with correct node heights

    pos = mergeTree_pos(G,height)
    fig, ax = plt.subplots()

    if style=='no_nodes':
        nx.draw_networkx(G, pos=pos, with_labels=False,node_size = 1,width = 5)
    elif style=='nodes':
        nx.draw_networkx(G, pos=pos, with_labels=True)
    if axes:
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    return

def draw_merge_tree(T,axes = True, style = 'no_nodes', simplify = True):
    
    if simplify:
        TNew = simplify_merge_tree(T)

        height = {node:TNew.nodes[node]['height'] for node in TNew.nodes()}

        draw_merge_tree_V1(TNew,height,axes = axes,style = style)
        
    else:
        
        height = {node:T.nodes[node]['height'] for node in T.nodes()}

        draw_merge_tree_V1(T,height,axes = axes,style = style)

    return