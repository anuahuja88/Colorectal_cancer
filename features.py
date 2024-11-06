import cv2
import os
import numpy as np

def clahe_features(input_dir, output_dir):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on grayscale images from the specified input directory
    and save the processed images to the specified output directory. Also, compute and store the histogram of each CLAHE 
    image for feature analysis.

    Parameters:
    - input_dir (str): Directory containing input images.
    - output_dir (str): Directory where CLAHE-processed grayscale images will be saved.

    Returns:
    - clahe_features_list (list): A list containing the histograms of the CLAHE-processed images.
    """
    clahe_features_list = []  # List to store histograms of CLAHE-processed images

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    def process_directory(directory, feature_list, output_dir):
        """
        Process each image in the specified directory by applying CLAHE, saving the resulting image,
        and storing its histogram for analysis.

        Parameters:
        - directory (str): Path to the directory containing images to process.
        - feature_list (list): List to append histograms of processed images.
        - output_dir (str): Path where processed images will be saved.
        """
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            tile = cv2.imread(filepath)

            # Continue if image loading failed
            if tile is None:
                continue

            # Convert image to grayscale
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE to enhance contrast in grayscale image
            clahe = cv2.createCLAHE(clipLimit=5)
            clahe_gray = clahe.apply(gray)

            cv2.imshow("clahe_gray",  clahe_gray)
            cv2.waitKey(0)

            # Save the processed CLAHE grayscale image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, clahe_gray)

            # Calculate histogram for the CLAHE grayscale image
            hist_gray = cv2.calcHist([clahe_gray], [0], None, [256], [0, 256])
            feature_list.append(hist_gray)

    # Process the input directory and save CLAHE-processed images
    process_directory(input_dir, clahe_features_list, output_dir)

    return clahe_features_list

def main():
    test_input = "Data/SPLIT_IMAGES"
    output_dir = "Data/CLAHE_grey_test"
     # New common directory for all CLAHE images
    clahe_features(test_input, output_dir)
    

if __name__ == "__main__":
    main()