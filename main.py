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
from imageProcessing.imageprocessing_helpers import preprocess, findContours, getSudokuGridCorners, drawCorners, cropToGridOnly, getGridBoxes, getPredictions


# Constants
# testImagePath = 'testImages/sudokuTestEasy.png' # 72/81 correct -> 88.89% accuracy
# correctGrid = [ 7, 0, 0, 0, 0, 0, 2, 0, 0,
#                 4, 0, 2, 0, 0, 0, 0, 0, 3,
#                 0, 0, 0, 2, 0, 1, 0, 0, 0,
#                 3, 0, 0, 1, 8, 0, 0, 9, 7,
#                 0, 0, 9, 0, 7, 0, 6, 0, 0,
#                 6, 5, 0, 0, 3, 2, 0, 0, 1,
#                 0, 0, 0, 4, 0, 9, 0, 0, 0,
#                 5, 0, 0, 0, 0, 0, 1, 0, 6,
#                 0, 0, 6, 0, 0, 0, 0, 0, 8 ]
# testImagePath = 'testImages/sudokuTestEasy2.png' # 80/81 correct -> 98.77% accuracy, 7 misclassified as 1
# testImagePath = 'testImages/sudokuTestEasy3.jpg' # 80/81 correct -> 98.77% accuracy, 7 misclassified as 1
# testImagePath = 'testImages/1.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/2.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/3.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/4.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/5.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/screenCropped.png' # Can't find sudoku grid
testImagePath = 'testImages/sudoku.jpg' # 81/81 correct -> 100% accuracy

height = 450
width = 450


def main():
    # Load image and resize
    img = cv2.imread(testImagePath)
    img = cv2.resize(img, (width, height))

    cv2.imshow('Image', img)
    cv2.waitKey(0)

    # Preprocess image to make it easier to find contours
    imgProcessed = preprocess(img)

    cv2.imshow('Image', imgProcessed)
    cv2.waitKey(0)

    # Find all external contours
    contours = findContours(imgProcessed)
    imgContours = img.copy()
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Image', imgContours)
    cv2.waitKey(0)

    # Get corners of sudoku grid by finding corners of largest square contour
    corners = getSudokuGridCorners(contours)
    if corners == []:
        print("No sudoku grid found")
        return
    imgCorners = img.copy()
    for corner in corners:
        drawCorners(corner, imgCorners)

    cv2.imshow('Image', imgCorners)
    cv2.waitKey(0)

    # Crop img to grid only
    imgGridDisplay = cropToGridOnly(imgProcessed, corners)
    imgGrid = imgGridDisplay.copy()

    cv2.imshow('Image', imgGrid)
    cv2.waitKey(0)

    # Get individual boxes
    boxes = getGridBoxes(imgGrid)

    # Load in trained digit classification model
    classification_model = tf.keras.models.load_model('classificationModel/generated_digit_classification_model.keras')
    # Classify digits to get grid
    grid = getPredictions(boxes, classification_model)

    # # Check if grid matches correctGrid
    # wrong = 0
    for i in range(81):
        # if grid[i] != correctGrid[i]:
        #     wrong += 1
            print(f'Grid at ({i}), got {grid[i]}')
            cv2.imshow('Image', boxes[i])
            cv2.waitKey(0)
    # print( f'Wrong: {wrong} / 81, Accuracy: {100 - (wrong / 81) * 100}%')

    # # Display img of each step for debugging
    # imgArray = [img, imgProcessed, imgContours, imgCorners, imgGridDisplay,]
    # for img in imgArray:
    #     cv2.imshow('Image', img)
    #     cv2.waitKey(0)


if __name__ == '__main__':
    main()
