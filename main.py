# STEPS:
# 1. Preprocess image
# 2. Find contours
# 3. Find sudoku grid
# 4. Extract and classify digits
# 5. Solve sudoku
# 6. Overlay solution on original image

import cv2
import numpy as np

testImagePath = 'images/sudokuTest.jpg'
height = 450
width = 450

# STEP 1: Preprocess image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Apply Gaussian blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2) # Apply adaptive threshold
    return imgThreshold

# STEP 2: Find contours
def findContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

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
    imgThreshold = preProcess(img)
    contours = findContours(imgThreshold)
    imgContours = img.copy()
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

    imgArray = [img, imgThreshold, imgContours, imgBlank]
    imgStacked = stackImages(imgArray, 1)

    cv2.imshow('Stacked Images', imgStacked)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
