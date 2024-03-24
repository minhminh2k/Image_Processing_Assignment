import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os

# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    image = cv2.imread(image_path)
    return image

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    plt.imshow(image)
    plt.title(title)
    plt.show()

# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    '''
    height, width, channels = image.shape
    
    img_gray = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            img_gray[y, x] = int(gray_value)
    
    return img_gray
    '''
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return img_gray

# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)


# flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    flipped_image = cv2.flip(image, 1) # 1 to flip horizontally 
    return flipped_image


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    height, width = image.shape[:2]
    
    center = (width / 2, height / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image


if __name__ == "__main__":
    
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")
    
    # Flipped the image
    img_flipped = flip_image(img)
    
    # Display the grayscale image
    save_image(img_flipped, "images/uet_flipped.jpg")
    
    # Flipped the image
    img_rotated = rotate_image(img, 45)
    
    # Display the grayscale image
    save_image(img_rotated, "images/uet_rotated.jpg")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/uet_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")
    
    # Save the flipped grayscale image
    save_image(img_gray_flipped, "images/uet_gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/uet_gray_rotated.jpg")

    # Show the images
    plt.show() 
