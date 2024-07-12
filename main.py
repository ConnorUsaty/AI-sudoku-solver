# STEPS:
# 1. Preprocess image
# 2. Find contours
# 3. Find sudoku grid
# 4. Extract and classify digits
# 5. Solve sudoku
# 6. Overlay solution on original image

import cv2
import numpy as np
import operator

testImagePath = 'images/sudokuTestEasy.png'
# testImagePath = 'images/sudokuTestHard.jpg'
height = 450
width = 450

# STEP 1: Preprocess image
def preProcess(img):
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
    # Find all contours, no hierarchy, simple approximation
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_extreme_corners(polygon, limit_fn, compare_fn):
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function

    # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]

def draw_extreme_corners(pts, original):
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

        # If contour has 4 sides and area > 10000, it's likely the sudoku grid -> Check if it's a square to confirm
        if num_sides == 4 and area > 100:
            top_left = find_extreme_corners(contour, min, np.add)  # has smallest (x + y) value
            top_right = find_extreme_corners(contour, max, np.subtract)  # has largest (x - y) value
            bot_left = find_extreme_corners(contour, min, np.subtract)  # has smallest (x - y) value
            bot_right = find_extreme_corners(contour, max, np.add)  # has largest (x + y) value
            
            # Check if the contour is a square -> 10% tolerance
            if not (0.90 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.10):
                continue

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

# Stack images for step by step visualization
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor

    return ver


# Main function
def main():
    img = cv2.imread(testImagePath)
    img = cv2.resize(img, (width, height))

    imgBlank = np.zeros((height, width, 3), np.uint8)

    imgProcessed = preProcess(img)

    contours = findContours(imgProcessed)
    imgContours = img.copy()
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    corners = getSudokuGridCorners(contours)
    # print(biggest)
    imgCorners = img.copy()

    for corner in corners:
        draw_extreme_corners(corner, imgCorners)
    # cv2.drawContours(imgBiggest, biggest, -1, (0, 255, 0), 10)

    imgGridDisplay = cropToGridOnly(img, corners)

    imgGrid = imgGridDisplay.copy()
    imgGrid = cv2.cvtColor(imgGrid, cv2.COLOR_BGR2GRAY)

    boxes = getGridBoxes(imgGrid)

    imgArray = [img, imgProcessed, imgContours, imgCorners, imgGridDisplay]
    # imgStacked = stackImages(imgArray, 1)

    for img in imgArray:
        cv2.imshow('Stacked Images', img)
        cv2.waitKey(0)

    # cv2.imshow('Stacked Images', imgStacked)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
