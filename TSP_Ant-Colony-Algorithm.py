import matplotlib.pyplot as plt
from math import *
import random
import numpy as np
import os
import matplotlib.patheffects as pe

# Define the number of nodes and their coordinates
num_nodes = 5  # Number of nodes
nodes_x = [1, 3, 2, 5, 4]  # Array of size 'num_nodes'
nodes_y = [4, 1.5, 3, 2, 3.5]

# Obtain the distance of each edge
edges_distances = np.matrix([[0.0 for i in range(num_nodes)] for j in range(num_nodes)])
for i in range(num_nodes):
    for j in range(i, num_nodes):
        edges_distances[i, j] = sqrt((nodes_x[j] - nodes_x[i])**2 + (nodes_y[j] - nodes_y[i])**2)
        edges_distances[j, i] = edges_distances[i, j]

# Define the parameters of the algorithm
alpha = 1
beta = 1

# Define the initial pheromone level of each edge
edges_pheromones = np.matrix([[1 for i in range(num_nodes)] for j in range(num_nodes)])

# Define the number of ants (number of iterations of the algorithm)
num_ants = 6

nodes = set(range(num_nodes))
optimal_length = np.matrix.sum(edges_distances)  # Initialized as a very high number
optimal_voyage = []

images = []
img_index = 0
filename = "TSP_AntColony_Animation_WithOptimum_BetterCode"


def capture(nframes=1):
    """
    Generates 'nframes' images of the current PyPlot figure
    """
    for i in range(nframes):
        global img_index  # Necessary to change its value
        image = filename + "_" + str(img_index) + ".png"
        img_index += 1
        plt.savefig(image)
        images.append(image)


def lighten_color(color, amount=0.5):
    """
    Function from StackOverflow (https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib)
    Returns a lighter (amount<1) or darker (amount>1) version of the color
    Examples:
    >> lighten_color('green', 0.3)
    # Returns a color that is like the 'green' color, but lighter
    >> lighten_color('green', 1.3)
    # Returns a color that is like the 'green' color, but darker
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


ax_main = plt.gca()
ax_optimum = plt.gcf().add_axes((0.05, 0.05, 0.25, 0.25))
ax_optimum.set_xticks([])
ax_optimum.set_yticks([])
ax_optimum.set_title("Current optimum",
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])

for ant in range(num_ants):

    # Clear main axes
    ax_main.clear()
    ax_main.scatter(nodes_x, nodes_y, c='black')
    ax_main.set_title("Iteration " + str(ant) + " of " + str(num_ants - 1))

    current_node = 0
    length = 0  # Sum of the distances of the voyage
    voyage = [0]  # Sequence of nodes

    while len(nodes.difference(set(voyage))) > 0:  # While there are unvisited nodes

        unvisited_nodes = []
        probabilities = []  # Likelihood of going to each unvisited node
        summation = 0

        # Calculate the probabilities to go to each node
        for node in nodes.difference(set(voyage)):
            unvisited_nodes.append(node)
            probability = (edges_pheromones[current_node, node]**alpha) *\
                          (1/edges_distances[current_node, node]**beta)
            probabilities.append(probability)
            summation += probability
        probabilities = np.array(probabilities)
        probabilities = probabilities / summation  # Normalized probabilities
        # Draw each possible path, indicating its probability with transparency
        lines = []
        for i in range(len(nodes.difference(set(voyage)))):
            lines.append(ax_main.plot([nodes_x[current_node], nodes_x[unvisited_nodes[i]]],
                                      [nodes_y[current_node], nodes_y[unvisited_nodes[i]]],
                                      c='blue', alpha=probabilities[i])[0])
        capture()

        # Choose a node among the unvisited nodes
        chosen_node = random.choices(unvisited_nodes, weights=probabilities, k=1)[0]
        # Mark the chosen path as green, and delete the others
        for i in range(len(nodes.difference(set(voyage)))):
            if unvisited_nodes[i] == chosen_node:
                lines[i].set_color('green')
                lines[i].set_alpha(1)
            else:
                lines[i].remove()
        capture()

        # Update edges, length and current node
        edges_pheromones[current_node, chosen_node] += 1  # Update the chosen path
        edges_pheromones[chosen_node, current_node] += 1  # We want it to be a symmetrical matrix
        length += edges_distances[current_node, chosen_node]  # Update the length of the voyage
        current_node = chosen_node
        voyage.append(current_node)

    # Include the path of return from the last node to 0
    edges_pheromones[current_node, 0] += 1
    edges_pheromones[0, current_node] += 1
    length += edges_distances[current_node, 0]
    voyage.append(0)

    # Update optimal length and voyage
    if length < optimal_length:
        optimal_length = length
        optimal_voyage = voyage
        ax_optimum.clear()
        ax_optimum.scatter(nodes_x, nodes_y, c='black')
        ax_optimum.set_xticks([])
        ax_optimum.set_yticks([])
        ax_optimum.set_title("Current optimum - " + str(round(optimal_length, 2)) + "m",
                             path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        for i in range(len(optimal_voyage) - 1):
            ax_optimum.plot([nodes_x[optimal_voyage[i]], nodes_x[optimal_voyage[i + 1]]],
                            [nodes_y[optimal_voyage[i]], nodes_y[optimal_voyage[i + 1]]],
                            c=lighten_color('green'))

    # Draw the last path, resulting in a full view of the voyage
    ax_main.plot([nodes_x[current_node], nodes_x[0]],
                 [nodes_y[current_node], nodes_y[0]],
                 c='green')
    ax_main.set_title("Iteration " + str(ant) + " of " + str(num_ants - 1)
                      + " - " + str(round(length, 2)) + "m")
    capture(3)

    # Clear main axes
    ax_main.clear()
    ax_main.scatter(nodes_x, nodes_y, c='black')
    ax_main.set_title("Iteration " + str(ant) + " of " + str(num_ants - 1) + " - resulting pheromone levels")
    # Show pheromone level of each path
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            ax_main.plot([nodes_x[i], nodes_x[j]],
                         [nodes_y[i], nodes_y[j]],
                         c=lighten_color('red', edges_pheromones[i, j] / num_ants))
    capture(3)

# Create a final image
plt.gcf().clf()
ax = plt.gca()
ax.scatter(nodes_x, nodes_y, c='black')
ax.set_title("PROGRAM FINISHED - RESULTING OPTIMUM - " + str(round(optimal_length, 2)) + "m")
for i in range(len(optimal_voyage) - 1):
    ax.plot([nodes_x[optimal_voyage[i]], nodes_x[optimal_voyage[i + 1]]],
            [nodes_y[optimal_voyage[i]], nodes_y[optimal_voyage[i + 1]]],
            c=lighten_color('green'))
capture(4)

# Create the animation using ImageMagick
fps = 2
os.system('convert -delay {} +dither +remap -layers Optimize {} "{}"'.
          format(100//fps, ' '.join(['"' + img + '"' for img in images]), filename + '.gif'))

# Delete the images
for img in images:
    if os.path.exists(img):
        os.remove(img)
