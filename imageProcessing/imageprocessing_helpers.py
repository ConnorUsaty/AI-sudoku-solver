import cv2
import numpy as np
import operator
import math
from typing import Final


# Constants
HEIGHT: Final[int] = 450
WIDTH: Final[int] = 450


# STEP 1: Preprocess image
def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    img_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # Apply adaptive threshold -> Make everything black or white based on above or below threshold
    img_inv = cv2.bitwise_not(img_threshold, 0) # Invert image -> Black background, white gridlines and numbers

    return img_inv


# STEP 2: Find contours
def findContours(img):
    # Find external contours (i.e. grid outline but not individual boxes), no hierarchy, simple approximation
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def getCorners(polygon, limit_fn, compare_fn):
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function

    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    # Return the x, y coordinates of the corner
    return polygon[section][0][0], polygon[section][0][1]

def drawCorners(pts, original):
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# STEP 3: Find sudoku grid corners -> Find largest square contour and get its corners
def getSudokuGridCorners(contours):
    grid_corners = []

    # Sort contours by area in descending order -> Grid contour is likely to be one of the largest contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
        num_sides = len(approx)

        # If contour has 4 sides and area > 1000, it's likely the sudoku grid -> Check if it's a square to confirm
        if num_sides == 4 and area > 1000:
            top_left = getCorners(contour, min, np.add)  # has smallest (x + y) value
            top_right = getCorners(contour, max, np.subtract)  # has largest (x - y) value
            bot_left = getCorners(contour, min, np.subtract)  # has smallest (x - y) value
            bot_right = getCorners(contour, max, np.add)  # has largest (x + y) value

            top_side = distance(top_left, top_right)
            left_side = distance(top_left, bot_left)
            right_side = distance(top_right, bot_right)
            bottom_side = distance(bot_left, bot_right)

            # Check if opposite sides are approximately equal -> Some sudoku images may not be perfect squares (Some may even be rectangles)
            if ((abs(top_side - bottom_side) < 10) and (abs(left_side - right_side) < 10)
                 and (top_side / left_side) > 0.6 and (top_side / left_side) < 1.4): # Check if it's a square with large tolerance but not large enough to get non-grid rectangles
                    grid_corners = [top_left, top_right, bot_right, bot_left]
                    break

    return grid_corners


# STEP 4: Crop to grid only
def cropToGridOnly(img, corners):
    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0], [WIDTH,0], [WIDTH,HEIGHT], [0,HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (WIDTH, HEIGHT))

    return img_warp


# STEP 5: Get grid boxes -> Extract individual boxes / digits
def getGridBoxes(img):
    # Since img is cropped to grid only, each box should be approximately 1/9th of the image
    boxes = []

    rows = np.vsplit(img, 9)
    for row in rows:
        cols = np.hsplit(row, 9)
        for col in cols:
            boxes.append(col)

    return boxes


# STEP 6: Classify digits -> Use trained model to classify digits
def getPredictions(boxes, model):
    # Prep boxes for classification -> Resize to 28x28, normalize, reshape to input layer shape
    boxes = np.array([cv2.resize(box, (28, 28)) for box in boxes])
    boxes = boxes / 255.0
    boxes = boxes.reshape(-1, 28, 28, 1)

    batch_size = 81 # Set batch size to 81 to classify all 81 sudoku boxes at once
    predictions = model.predict(boxes, batch_size=batch_size) # Get predictions for all boxes at once

    return [np.argmax(prediction) for prediction in predictions] # Return the digit with the highest probability for each box


# STEP 7: Draw solution on image
def drawSolution(warped_img, solved_puzzle, squares_processed):
    warped_img = cv2.bitwise_not(warped_img, 0) # Turn image back to black background, white text
    width = warped_img.shape[0] // 9
    img_solution = warped_img

    index = 0
    for j in range(9):
        for i in range(9):
            if squares_processed[index] == 0: # If square was empty
                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_solution
