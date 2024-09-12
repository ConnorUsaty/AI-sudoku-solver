import cv2
from typing import Final
import os

# Import functions from imageprocessing_helpers.py
from imageProcessing.imageprocessing_helpers import preprocess, findContours, getSudokuGridCorners, cropToGridOnly, getGridBoxes

# Constants
HEIGHT: Final[int] = 450
WIDTH: Final[int] = 450


def processAndLabel(testImagePath):
    img = cv2.imread(testImagePath)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    imgProcessed = preprocess(img) # Preprocess image to make it easier to find contours
    contours = findContours(imgProcessed)     # Find all external contours

    # Get corners of sudoku grid by finding corners of largest square contour
    corners = getSudokuGridCorners(contours)
    if corners == []:
        print("No sudoku grid found")
        return

    imgGridOnly = cropToGridOnly(imgProcessed, corners)  # Crop img to grid only
    boxes = getGridBoxes(imgGridOnly) # Get individual boxes of grid

    # Label processed boxes
    for i in range(81):
        cv2.imshow('Image', boxes[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        label = input("Enter label: ")
        cv2.imwrite(f"labelledImages/{testImagePath.split('/')[1]}_{label}_{i}.png", boxes[i])


def main():
    images = [i for i in os.listdir(os.path.abspath('testImages/'))]
    
    for image in images:
        testImagePath = f'testImages/{image}'
        processAndLabel(testImagePath)



if __name__ == '__main__':
    main()
