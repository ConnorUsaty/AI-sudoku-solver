import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_generator import applyRotation, applyNoise, generateRandomBorders


def randomizeImage(index, img_name) -> None:

    orig_image = Image.open(f'labelledImages/{img_name}')

    rotations = [-10, -5, -3, 0, 3, 5, 10]
    noises = [0, 5, 10, 15, 20]

    # Apply randomization
    for rotation in rotations:
        for noise in noises:
            curr_image = orig_image.copy()
            curr_image = applyRotation(curr_image, rotation)
            curr_image = applyNoise(curr_image, noise)

            # Save image with unique name
            curr_image = np.array(curr_image)
            cv2.imwrite(f"{testImagePath.split('.')[0]}_{index}_{rotation}_{noise}", curr_image)



def main():
    images = [i for i in os.listdir(os.path.abspath('labelledImages/'))]
    for i, image_name in tqdm(enumerate(images), desc='Randomizing images'):
        randomizeImage(i, image_name)
    
    print('Finished randomizing images')


if __name__ == '__main__':
    main()
