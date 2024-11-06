import torch
import torchvision.transforms as T
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from torchvision.datasets.folder import default_loader
import torch.nn as nn
import os
import re

import pandas as pd


# Load the model
model = resnet18()
# model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Sequential(
#     nn.Dropout(p=0.5),  # Dropout layer to reduce overfitting
#     nn.Linear(model.fc.in_features, 2)  # Assuming binary classification
# )\
for params in model.parameters():
    params.requires_grad = True

model.fc = nn.Linear(in_features=512, out_features=1)


model.load_state_dict(torch.load('resnet18CLAHEgrey.pth'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

test_transform = T.Compose([ 
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor()   #Required!
]) 

# Path to the directory containing test images
test_dir = 'Data/CLAHE_grey_test'
image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]

# Function to load and preprocess images
def load_image(image_path):
    image = default_loader(image_path)
    image = test_transform(image)
    return image

def make_predictions(image_paths, model):
    model.eval()
    predictions = []

    with torch.no_grad():
        for img_path in image_paths:
            image = load_image(img_path).unsqueeze(0).to(device)
            output = model(image)
            pred = torch.sigmoid(output[:, 0]).round().cpu().numpy()[0]
            predictions.append((os.path.basename(img_path), int(pred)))

    return predictions

# Make predictions on the test dataset
test_predictions = make_predictions(image_paths, model)


# Save the predictions to an Excel file
def save_predictions_to_excel(predictions, output_path='Tables/resnet18CLAHEgrey.csv'):
    sorted_predictions = sorted(predictions, key=lambda x: x[0])
    print(sorted_predictions)
    # Create a DataFrame and save it to an Excel file
    df = pd.DataFrame(sorted_predictions, columns=['Filename', 'Prediction'])
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path} in ascending order.")

# Call the function to save the predictions
save_predictions_to_excel(test_predictions)

# Print the filenames and their predictions
for img_path, prediction in test_predictions:
    print(f"Filename: {img_path}, Prediction: {prediction}")
