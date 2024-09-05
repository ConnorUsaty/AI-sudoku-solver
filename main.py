import tensorflow as tf
import cv2
from typing import Final

# Import functions from imageprocessing_helpers.py
from imageProcessing.imageprocessing_helpers import preprocess, findContours, getSudokuGridCorners, drawCorners, cropToGridOnly, getGridBoxes, getPredictions, drawSolution
# Import functions from sudoku_solver.py
from sudokuSolver.sudoku_solver import solverWrapper


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
testImagePath = 'testImages/5.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/screenCropped.png' # Can't find sudoku grid
# testImagePath = 'testImages/sudoku.jpg' # Can't find sudoku grid
HEIGHT: Final[int] = 450
WIDTH: Final[int] = 450


def main():
    # Load image and resize
    img = cv2.imread(testImagePath)
    img = cv2.resize(img, (WIDTH, HEIGHT))

    imgProcessed = preprocess(img) # Preprocess image to make it easier to find contours
    contours = findContours(imgProcessed)     # Find all external contours
    # imgContours = img.copy()
    # cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    # Get corners of sudoku grid by finding corners of largest square contour
    corners = getSudokuGridCorners(contours)
    if corners == []:
        print("No sudoku grid found")
        return
    # imgCorners = img.copy()
    # for corner in corners:
    #     drawCorners(corner, imgCorners)

    imgGridOnly = cropToGridOnly(imgProcessed, corners)  # Crop img to grid only
    boxes = getGridBoxes(imgGridOnly) # Get individual boxes of grid

    # Load in trained digit classification model
    classification_model = tf.keras.models.load_model('classificationModel/generated_digit_classification_model.keras')
    grid = getPredictions(boxes, classification_model) # Classify digits to get grid

    # Solve sudoku; Returns error message in solution and time = None if no solution found
    solution, solve_time = solverWrapper(grid)

    if solve_time is not None:
        solvedGrid = drawSolution(imgGridOnly, solution, grid)
        print(f"Time taken: {solve_time}")
        cv2.imshow('Image', solvedGrid)
        cv2.waitKey(0)
    else:
        print(f"{solution}") # Contains error message if no solution found


    # # Check if grid matches correctGrid
    # wrong = 0
    # for i in range(81):
    #     # if grid[i] != correctGrid[i]:
    #     #     wrong += 1
    #         print(f'Grid at ({i}), got {grid[i]}')
    #         cv2.imshow('Image', boxes[i])
    #         cv2.waitKey(0)
    # print( f'Wrong: {wrong} / 81, Accuracy: {100 - (wrong / 81) * 100}%')


if __name__ == '__main__':
    main()
