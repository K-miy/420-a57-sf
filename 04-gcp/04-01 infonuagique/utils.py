import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def draw_objects_bounding_boxes(axes, width, height, results, threshold = 0.80):
    """
    Draw detected objects bounding boxes and labels for category and score
    """
    # Iterate through detected objects
    for r in results:
        # Dont show detections with score < threshold
        if r.score < threshold:
            continue
        # Bounding boxes are defined as Top-Left-Width-Height
        L = int(r.bounding_poly.normalized_vertices[0].x * width)
        T = int(r.bounding_poly.normalized_vertices[0].y * height)
        W = int(r.bounding_poly.normalized_vertices[1].x * width) - L
        H = int(r.bounding_poly.normalized_vertices[2].y * height) - T
        # Generate a random red-ish color
        color = np.random.rand(3,)
        color[0] = 1
        # Create a rectangle patch and add it to the axes
        rect = patches.Rectangle((L,T), W, H, linewidth=1, edgecolor=color, facecolor='none')
        axes.add_patch(rect)
        # Create a label for category and score
        label = r.name + '-' + str(r.score)
        plt.text(L+6, T+18, label , bbox=dict(facecolor=color, alpha=0.5), fontsize=16)


def display_image(image):
    """
    Display an image
    """
    # Open image as a NumPy array
    im = np.array(Image.open(image))
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    # Remove axis
    plt.axis('off')

    return im, ax


def display_objects_results(image, results, threshold = 0.50):
    """
    Display Object detection results
    """
    # Open and display image
    im, ax = display_image(image)
    # Get image dimensions
    height, width, _ = im.shape
    # Draw boundling boxes and labels
    draw_objects_bounding_boxes(ax, width, height, results, threshold)