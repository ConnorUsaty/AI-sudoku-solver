import cv2
import numpy as np
import operator


# Constants
height = 450
width = 450


# STEP 1: Preprocess image
def preprocess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (9,9), 0) # Apply Gaussian blur -> Remove noise
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # Apply adaptive threshold -> Make everything black or white based on above or below threshold
    imgInv = cv2.bitwise_not(imgThreshold, 0) # Invert image -> Black background, white gridlines
    imgKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) # Get a rectangular kernel
    imgMorph = cv2.morphologyEx(imgInv, cv2.MORPH_OPEN, imgKernel) # Apply morphological transformation
    imgFinal = cv2.dilate(imgMorph, imgKernel, iterations=1) # Dilate image -> Fill in gaps and make lines bolder

    return imgFinal

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

# STEP 3: Find sudoku grid corners -> Find largest square contour and get its corners
def getSudokuGridCorners(contours):
    gridCorners = None

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
            
            # Check if the contour is a square -> 10% tolerance
            if (0.90 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.10):
                gridCorners = [top_left, top_right, bot_right, bot_left]
                break
        
    if gridCorners is None:
        return []
    
    return gridCorners

# STEP 4: Crop to grid only
def cropToGridOnly(img, corners):
    pts1 = np.float32(corners)
    pts2 = np.float32([[0,0], [width,0], [width,height], [0,height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))

    return imgWarp

# STEP 5: Get grid boxes
def getGridBoxes(img):
    # Since img is cropped to grid only, each box should be approximately 1/9th of the image
    boxes = []

    rows = np.vsplit(img, 9)
    for row in rows:
        cols = np.hsplit(row, 9)
        for col in cols:
            boxes.append(col)

    return boxes
