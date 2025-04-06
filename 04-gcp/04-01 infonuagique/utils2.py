import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def draw_objects_bounding_boxes(axes, width, height, results, threshold = 0.80):
    """
    Draw detected objects bounding boxes and labels for category and score
    """
    # Iterate through detected objects
    for k, _ in results.items():
        # Dont show detections with score < threshold
        if results[k]['score'] < threshold:
            continue
        # Bounding boxes are defined as Top-Left-Width-Height
        L = int(results[k]['vertices'][0][0] * width)
        T = int(results[k]['vertices'][0][1] * height)
        W = int(results[k]['vertices'][1][0] * width) - L
        H = int(results[k]['vertices'][2][1] * height) - T
        # Generate a random red-ish color
        color = np.random.rand(3,)
        color[0] = 1
        # Create a rectangle patch and add it to the axes
        rect = patches.Rectangle((L,T), W, H, linewidth=1, edgecolor=color, facecolor='none')
        axes.add_patch(rect)
        # Create a label for category and score
        label = results[k]['name'] + '-' + str(results[k]['score'])
        plt.text(L+6, T+18, label , bbox=dict(facecolor=color, alpha=0.5), fontsize=6)


def create_emotion_label(results):
    """
    Create emotion label
    """
    JOY = 'joy'
    ANGER = 'anger'
    SURPRISE = 'surprise'
    LIKELY = 'LIKELY'
    VERY_LIKELY = 'VERY_LIKELY'
    emotion = str()
    if results[JOY] == LIKELY or results[JOY] == VERY_LIKELY:
        emotion += "JOY"
    if results[ANGER] == LIKELY or results[ANGER] == VERY_LIKELY:
        emotion += " - ANGER"
    if results[SURPRISE] == LIKELY or results[SURPRISE] == VERY_LIKELY:
        emotion += " - SURPRISE"
    
    return emotion


def draw_faces_bounding_boxes(axes, results):
    """
    Draw detected faces bounding boxes and labels for emotion
    """
    # Iterate through detected faces
    for k, _ in results.items():
        # Bounding boxes are defined as top-left-width-height
        L = int(results[k]['vertices'][0][0])
        T = int(results[k]['vertices'][0][1])
        W = int(results[k]['vertices'][1][0]) - L
        H = int(results[k]['vertices'][2][1]) - T
        # Generate a random red-ish color
        color = np.random.rand(3,)
        color[0] = 1
        # Create a rectangle patch and add it to the axes
        rect = patches.Rectangle((L,T), W, H, linewidth=1, edgecolor=color, facecolor='none')
        axes.add_patch(rect)
        # Create a label for emotion
        label = create_emotion_label(results[k])
        plt.text(L+6, T+18, label , bbox=dict(facecolor=color, alpha=0.5), fontsize=6)


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


def display_objects_results(image, results, threshold = 0.80):
    """
    Display Object detection results
    """
    # Open and display image
    im, ax = display_image(image)
    # Get image dimensions
    height, width, _ = im.shape
    # Draw boundling boxes and labels
    draw_objects_bounding_boxes(ax, width, height, results, threshold)
    


def display_faces_results(image, results):
    """
    Display Face detection results
    """
    # Open and display image
    _, ax = display_image(image)
    # Draw boundling boxes and labels
    draw_faces_bounding_boxes(ax, results)
