from PIL import Image
import os

def split_image(image_path, output_folder):
    img = Image.open(image_path)
    width, height = img.size

    if width != 1024 or height != 1024:
        print(f"Skipping {image_path}: not a 1024x1024 image")
        return

    base_name = os.path.basename(image_path).split('.')[0]
    img_512x512 = [
        img.crop((0, 0, 512, 512)),
        img.crop((512, 0, 1024, 512)),
        img.crop((0, 512, 512, 1024)),
        img.crop((512, 512, 1024, 1024))
    ]

    for i, cropped_img in enumerate(img_512x512):
        cropped_img.save(os.path.join(output_folder, f"{base_name}_part{i+1}.png"))

def main():
    input_folder = 'TEST_IMAGES'
    output_folder = 'SPLIT_IMAGES'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            split_image(image_path, output_folder)

if __name__ == "__main__":
    main()
