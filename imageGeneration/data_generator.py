import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
from tqdm import tqdm
import time
import random


def create_text_image(text, font_path, font_size=24, rotation=0, noise_level=0):
    # Generate an image with the specified text and font
    if text != '0':
        font = ImageFont.truetype(font_path, font_size)
        image = Image.new('L', (28, 28), color=0)  # 'L' mode for grayscale, black background
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        dw = random.randint(-5, 5)  # Random horizontal offset
        dh = random.randint(-5, 5)  # Random vertical offset
        draw.text(((28 - width) / 2 - dw, (28 - height) / 2 - 6 - dh), text, font=font, fill=255)  # White text
    else:
        image = Image.new('L', (28, 28), color=0)

    # Apply randomization
    image = applyRotation(image, rotation)
    image = applyNoise(image, noise_level)
    image = generateRandomBorders(image)
    return image


def applyRotation(image, rotation):
    # Apply rotation
    if rotation != 0:
        image = image.rotate(rotation, expand=1, fillcolor=0)
        image = ImageOps.fit(image, (28, 28), method=0, bleed=0.0, centering=(0.5, 0.5))
    return image


def applyNoise(image, noise_level):
    # Apply noise
    if noise_level != 0:
        noise = np.random.randint(0, noise_level, (28, 28), dtype='uint8')
        noise_image = Image.fromarray(noise, mode='L')
        image = ImageChops.add(image, noise_image)
    return image


def generateRandomBorders(image):
    image = np.array(image)

    # Randomly add white borders to simulate sudoku grid lines on edge of image
    border_sides = random.sample(['top', 'bottom', 'left', 'right'], random.randint(1, 4))
    border_width = random.randint(1, 5)  # Random border width between 1 and 4 pixels
    if 'top' in border_sides:
        image[:border_width, :] = 255
    if 'bottom' in border_sides:
        image[-border_width:, :] = 255
    if 'left' in border_sides:
        image[:, :border_width] = 255
    if 'right' in border_sides:
        image[:, -border_width:] = 255

    image = Image.fromarray(image)
    return image


def main():
    # Use 5 popular fonts for suduko puzzles
    popular_fonts = [
        'arial.ttf',
        'times.ttf',
        'verdana.ttf',
        'georgia.ttf',
        'comic.ttf',
    ]
    # Use a large variety of font sizes, different angles, and noise levels for training data
    font_sizes = [i for i in range(18, 33, 2)]
    angles = [i for i in range(-18, 19, 2)]
    noise_levels = [i for i in range(0, 25)]

    images = 0
    start_time = time.time()
    # Generate and save images for numbers 1-9 in each font with specified angles and noise levels
    # Will generate blank images for 0
    for font in popular_fonts:
        for i in tqdm(range(0, 10), desc=f'Generating images for {font.split(".")[0]}'):
            for font_size in font_sizes:
                for angle in angles:
                    for noise_level in noise_levels:
                        img = create_text_image(str(i), font, font_size=font_size, rotation=angle, noise_level=noise_level)

                        # Save image with appropriate name (No dashes allowed in file names)
                        if angle < 0:
                            angle_str = "neg" + str(abs(angle))
                        else:
                            angle_str = str(angle)

                        img.save(f'classificationModel/trainingImages/{font.split(".")[0]}_text_{i}_size_{font_size}_angle_{angle_str}_noise_{noise_level}.png')
                        images += 1

    print(f'\nGenerated {images} images')
    print(f'Elapsed time: {time.time() - start_time}')


if __name__ == '__main__':
    main()
