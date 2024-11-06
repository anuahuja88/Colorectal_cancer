import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.feature as feature

def get_HSV_histogram(tile):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    return hist_hue, hist_saturation, hist_value

def get_BGR_histogram(tile):
    B = cv2.calcHist([tile], [0], None, [256], [0, 256])
    G = cv2.calcHist([tile], [1], None, [256], [0, 256])
    R = cv2.calcHist([tile], [2], None, [256], [0, 256])
    
    return B, G, R

def extract_colour_histograms(true_dir, false_dir):
    mss_channel1_hist = []
    mss_channel2_hist = []
    mss_channel3_hist = []
    msi_channel1_hist = []
    msi_channel2_hist = []
    msi_channel3_hist = []

    # true tiles
    for filename in os.listdir(true_dir):
        filepath = os.path.join(true_dir, filename)
        tile = cv2.imread(filepath)
        if tile is None:
            continue
        true_channel1, true_channel2, true_channel3 = get_HSV_histogram(tile)
        msi_channel1_hist.append(true_channel1)
        msi_channel2_hist.append(true_channel2)
        msi_channel3_hist.append(true_channel3)

    # false tiles
    for filename in os.listdir(false_dir):
        filepath = os.path.join(false_dir, filename)
        tile = cv2.imread(filepath)
        if tile is None:
            continue
        false_channel1, false_channel2, false_channel3 = get_HSV_histogram(tile)
        mss_channel1_hist.append(false_channel1)
        mss_channel2_hist.append(false_channel2)
        mss_channel3_hist.append(false_channel3)

    # Combine histograms
    true_hist1 = np.mean(msi_channel1_hist, axis=0)
    true_hist2 = np.mean(msi_channel2_hist, axis=0)
    true_hist3 = np.mean(msi_channel3_hist, axis=0)

    false_hist1 = np.mean(mss_channel1_hist, axis=0)
    false_hist2 = np.mean(mss_channel2_hist, axis=0)
    false_hist3 = np.mean(mss_channel3_hist, axis=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(true_hist1, color='r')
    plt.title('H Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.plot(true_hist2, color='g')
    plt.title('S Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 3)
    plt.plot(true_hist3, color='b')
    plt.title('V Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(false_hist1, color='r')
    plt.title('H Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.plot(false_hist2, color='g')
    plt.title('S Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 3)
    plt.plot(false_hist3, color='b')
    plt.title('V Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def extract_texture(true_dir, false_dir):
    true_contrast_avg = []
    true_dissimilarity_avg = []
    true_homogeneity_avg = []
    true_energy_avg = []
    true_correlation_avg = []
    true_ASM_avg = []

    false_contrast_avg = []
    false_dissimilarity_avg = []
    false_homogeneity_avg = []
    false_energy_avg = []
    false_correlation_avg = []
    false_ASM_avg = []

    def process_directory(directory, contrast_avg, dissimilarity_avg, homogeneity_avg, energy_avg, correlation_avg, ASM_avg):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            tile = cv2.imread(filepath)
            if tile is None:
                continue
            tile_grey = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            # Calculate GLCM
            graycom = feature.graycomatrix(tile_grey, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

            # Find GLCM properties
            contrast = feature.graycoprops(graycom, 'contrast')
            dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
            homogeneity = feature.graycoprops(graycom, 'homogeneity')
            energy = feature.graycoprops(graycom, 'energy')
            correlation = feature.graycoprops(graycom, 'correlation')
            ASM = feature.graycoprops(graycom, 'ASM')

            # Append GLCM properties to lists
            contrast_avg.append(contrast)
            dissimilarity_avg.append(dissimilarity)
            homogeneity_avg.append(homogeneity)
            energy_avg.append(energy)
            correlation_avg.append(correlation)
            ASM_avg.append(ASM)

    # Process 'true' and 'false' directories
    process_directory(true_dir, true_contrast_avg, true_dissimilarity_avg, true_homogeneity_avg, true_energy_avg, true_correlation_avg, true_ASM_avg)
    process_directory(false_dir, false_contrast_avg, false_dissimilarity_avg, false_homogeneity_avg, false_energy_avg, false_correlation_avg, false_ASM_avg)

    # Calculate average GLCM properties for true and false tiles
    true_averages = [np.mean(prop) for prop in [true_contrast_avg, true_dissimilarity_avg, true_homogeneity_avg, true_energy_avg, true_correlation_avg, true_ASM_avg]]
    false_averages = [np.mean(prop) for prop in [false_contrast_avg, false_dissimilarity_avg, false_homogeneity_avg, false_energy_avg, false_correlation_avg, false_ASM_avg]]

    # Plot individual graphs for each GLCM property
    properties = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM']

    for prop, true_avg, false_avg in zip(properties, true_averages, false_averages):
        plt.figure()
        plt.bar(['True', 'False'], [true_avg, false_avg], color=['blue', 'red'])
        plt.title(prop)
        plt.ylabel('Average Value')
        plt.show()

def local_binary_pattern(true_dir, false_dir):
    lbp_features_true = []
    lbp_features_false = []

    def process_directory(directory, lbp_features):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            tile = cv2.imread(filepath)
            if tile is None:
                continue
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)  # Avoid divide by zero

            # Append LBP features to the list
            lbp_features.append(hist)
            
            # Display the LBP image
            lbp_display = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
            lbp_display = cv2.convertScaleAbs(lbp_display)

            # Display the LBP image using OpenCV
            cv2.imshow(f'LBP Image: {filename}', lbp_display)
            cv2.waitKey(0) 

    # Process 'true' and 'false' directories
    process_directory(true_dir, lbp_features_true)
    process_directory(false_dir, lbp_features_false)

    # Plot mean LBP histograms for 'true' and 'false' images
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lbp_features_true[0])), np.mean(lbp_features_true, axis=0), color='b')
    plt.title("True LBP Features")
    plt.xlabel('LBP Pattern')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(lbp_features_false[0])), np.mean(lbp_features_false, axis=0), color='r')
    plt.title("False LBP Features")
    plt.xlabel('LBP Pattern')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return lbp_features_true, lbp_features_false

def clahe_features(input_dir, output_dir):
    clahe_features_list = []

    # Create output directory for all CLAHE grayscale images
    os.makedirs(output_dir, exist_ok=True)

    def process_directory(directory, feature_list, output_dir):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            tile = cv2.imread(filepath)
            if tile is None:
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE to the grayscale image
            clahe = cv2.createCLAHE(clipLimit=5)
            clahe_gray = clahe.apply(gray)

            # Save the CLAHE processed grayscale image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, clahe_gray)

            # For demonstration, storing the histogram of the CLAHE grayscale image
            hist_gray = cv2.calcHist([clahe_gray], [0], None, [256], [0, 256])
            feature_list.append(hist_gray)

    # Process the input directory and save all CLAHE grayscale images to the output directory
    process_directory(input_dir, clahe_features_list, output_dir)

    return clahe_features_list
def main():
    test_input = "Data/SPLIT_IMAGES"
    output_dir = "Data/CLAHE_grey_test"
     # New common directory for all CLAHE images
    clahe_features(test_input, output_dir)
    

if __name__ == "__main__":
    main()