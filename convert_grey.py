import os
import cv2

def convert_to_grayscale(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Modify the extensions as needed
            filepath = os.path.join(input_dir, filename)
            # Read the image
            image = cv2.imread(filepath)
            if image is None:
                print(f"Warning: {filepath} could not be read.")
                continue
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, gray_image)
            print(f"Saved grayscale image to {output_path}")

def main():
    # Define input and output directories
    input_dirs = ['train_crc/msi_0', 'train_crc/msi_1']
    output_dirs = ['RGBGREY/msi_0', 'RGBGREY/msi_1']

    # Process each directory
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        convert_to_grayscale(input_dir, output_dir)

if __name__ == "__main__":
    main()
