# STEPS:
# 1. Preprocess image
# 2. Find contours
# 3. Find sudoku grid
# 4. Extract and classify digits
# 5. Solve sudoku
# 6. Overlay solution on original image

# Import necessary libraries
import tensorflow as tf
import cv2
import numpy as np

# Import functions from imageprocessing_helpers.py
from imageProcessing.imageprocessing_helpers import preprocess, findContours, getSudokuGridCorners, drawCorners, cropToGridOnly, getGridBoxes


# Constants
testImagePath = 'images/sudokuTestEasy.png'
# testImagePath = 'images/sudokuTestHard.jpg'
height = 450
width = 450


# Main function
def main():
    # Load image and resize
    img = cv2.imread(testImagePath)
    img = cv2.resize(img, (width, height))

    # Preprocess image to make it easier to find contours
    imgProcessed = preprocess(img)

    # Find all external contours
    contours = findContours(imgProcessed)
    imgContours = img.copy()
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    # Get corners of sudoku grid by finding corners of largest square contour
    corners = getSudokuGridCorners(contours)
    imgCorners = img.copy()
    for corner in corners:
        drawCorners(corner, imgCorners)

    # Crop img to grid only
    imgGridDisplay = cropToGridOnly(img, corners)
    imgGrid = imgGridDisplay.copy()
    imgGrid = cv2.cvtColor(imgGrid, cv2.COLOR_BGR2GRAY) # Convert to grayscale for easier processing

    # Get individual boxes
    boxes = getGridBoxes(imgGrid)
    randBox = np.random.randint(0, len(boxes)-1)

    # Load in trained digit classification model
    classification_model = tf.keras.models.load_model('classificationModel/digit_classification_model.keras')

    # Display img of each step for debugging
    imgArray = [img, imgProcessed, imgContours, imgCorners, imgGridDisplay, boxes[randBox]]
    for img in imgArray:
        cv2.imshow('Image', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
