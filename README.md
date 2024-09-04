# AI-sudoku-solver
## Overview
An application that:
- Takes in .jpg or .png files
- Preprocesses the image using OpenCV
- Locates the sudoku grid through contours on the image
- Utilizes a CNN to classify the digit in each box
- Solves the sudoku
- Displays solved sudoku for user
## Current Features:
- CNN model acheived **99.96% Validation Accuracy** and **0.13% Validation Loss**
- CNN model is trained on 213,000 unique generated images from data generation script
- Data generation script to easily generate more unique and realistic training data for the CNN
- Extremely fast McGill sudoku solver algorithm
- OpenCV image processing to extract soduko grid from the image and each individual box from the grid
## To-Do List:
- Implement OpenCV's video capture abilities to scan for pictures from a real-time video stream
- Acheive a validation accuracy of 100% (Most common error is 7s misclassified as 1s)
- Test more model architectures and custom datasets to acheive above
- Develop Batch script to easily run locally
- Develop React.JS front end to easily host online
