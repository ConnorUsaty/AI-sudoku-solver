import tensorflow as tf
import cv2
import time
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
testImagePath = 'testImages/2.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/3.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/4.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/5.jpg' # 81/81 correct -> 100% accuracy
# testImagePath = 'testImages/screenCropped.png' # Can't find sudoku grid
# testImagePath = 'testImages/sudoku.jpg' # Can't find sudoku grid
IMG_HEIGHT: Final[int] = 450
IMG_WIDTH: Final[int] = 450
WINDOW_WIDTH: Final[int] = 960
WINDOW_HEIGHT: Final[int] = 720
WINDOW_BRIGHTNESS: Final[int] = 150
FRAME_RATE: Final[int] = 30

elapsed_time = time.time()

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH) # Width is id 3
cap.set(4, WINDOW_HEIGHT) # Height is id 4
cap.set(10, WINDOW_BRIGHTNESS) # Brightness is id 10

# Load in trained digit classification model
classification_model = tf.keras.models.load_model('classificationModel/generated_digit_classification_model.keras')


def main():

    prev = time.time()
    seen = {}

    while True:
        success, img = cap.read()
        elapsed_time = time.time() - prev

        if elapsed_time > 1.0/FRAME_RATE:
            prev = time.time()

            img_result = img.copy()
            img_corners = img.copy()

            img_processed = preprocess(img)
            contours = findContours(img_processed)
            corners = getSudokuGridCorners(contours)

            # If sudoku grid on screen
            if corners != []:
                img_grid_only = cropToGridOnly(img_processed, corners)
                boxes = getGridBoxes(img_grid_only) # 1D array of 81 boxes
                grid = getPredictions(boxes, classification_model) # 1D array of 81 digits / predictions
                solution, solve_time = solverWrapper(grid) # Returns error message in solution and time = None if no solution found

                if solve_time is not None:
                    solved_grid = drawSolution(img_grid_only, solution, grid)
                    cv2.imshow('Image', solved_grid)
                else:
                    print(f"{solution}")

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

        # # Get corners of sudoku grid by finding corners of largest square contour
        # corners = getSudokuGridCorners(contours)
        # if corners == []:
        #     print("No sudoku grid found")
        #     return
        # # imgCorners = img.copy()
        # # for corner in corners:
        # #     drawCorners(corner, imgCorners)

        # imgGridOnly = cropToGridOnly(imgProcessed, corners)  # Crop img to grid only
        # boxes = getGridBoxes(imgGridOnly) # Get individual boxes of grid

        # grid = getPredictions(boxes, classification_model) # Classify digits to get grid

        # # Solve sudoku; Returns error message in solution and time = None if no solution found
        # solution, time = solverWrapper(grid)

        # if time is not None:
        #     solvedGrid = drawSolution(imgGridOnly, solution, grid)
        #     cv2.imshow('Image', solvedGrid)
        #     cv2.waitKey(0)
        # else:
        #     print(f"{solution}") # Contains error message if no solution found


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
