import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
from tqdm import tqdm


def create_text_image(text, font_path, font_size=32, rotation=0, noise_level=0):
    # Generate an image with the specified text and font
    font = ImageFont.truetype(font_path, font_size)
    image = Image.new('L', (28, 28), color=0)  # 'L' mode for grayscale, black background
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((28 - width) / 2, (28 - height) / 2 - 6), text, font=font, fill=255)  # White text

    # Apply rotation
    if rotation != 0:
        image = image.rotate(rotation, expand=1, fillcolor=0)
        image = ImageOps.fit(image, (28, 28), method=0, bleed=0.0, centering=(0.5, 0.5))
    
    # Apply noise
    if noise_level != 0:
        noise = np.random.randint(0, noise_level, (28, 28), dtype='uint8')
        noise_image = Image.fromarray(noise, mode='L')
        image = ImageChops.add(image, noise_image)
    
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
    # Use a large variety of different angles and noise levels for training data
    angles = [i for i in range(-15, 16)]
    noise_levels = [i for i in range(0, 16)]

    # Generate and save images for numbers 1-9 in each font with specified angles and noise levels
    # Don't need to generate images for 0 since it will be used to represent empty cells
    for font in popular_fonts:
        for i in tqdm(range(1, 10), desc=f'Generating images for {font}'):
            for angle in angles:
                for noise_level in noise_levels:
                    img = create_text_image(str(i), font, rotation=angle, noise_level=noise_level)

                    # Save image with appropriate name (No dashes allowed in file names)
                    if angle < 0:
                        angle_str = "neg" + str(abs(angle))
                    else:
                        angle_str = str(angle)
                    if noise_level < 0:
                        noise_level_str = "neg" + str(abs(noise_level))
                    else:
                        noise_level_str = str(noise_level)

                    img.save(f'classificationModel/trainingImages/{font.split(".")[0]}_text_{i}_angle_{angle_str}_noise_{noise_level_str}.png')


if __name__ == '__main__':
    main()
