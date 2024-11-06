import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
import torch.nn as nn
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from torchmetrics.functional import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import numpy as np


def apply_clahe(image):
    # Convert the image to a NumPy array and to grayscale
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    
    # Convert the CLAHE image back to PIL format (necessary for PyTorch transforms)
    return Image.fromarray(clahe_image)
  

# Now define a custom transform that first applies CLAHE, and then converts to tensor
class CLAHETransform:
    def __call__(self, img):
        # Apply the CLAHE transformation
        img = apply_clahe(img)
        # Convert to 3-channel grayscale (to keep the ResNet architecture consistent)
        img = img.convert("RGB")
        return img





# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Learning and training parameters.
epochs = 20
batch_size = 32
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
model = resnet50()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

for params in model.parameters():
    params.requires_grad = True

model.fc = nn.Linear(in_features=2048, out_features=1)

# model.to(device)
# Optimizer.     
optimizer = Adam(model.parameters(), lr=learning_rate)

# Loss function.
loss_fn = nn.BCEWithLogitsLoss()  #pos_weight=torch.ones([batch_size]))

# Metrics
precision_metric = BinaryPrecision(threshold=0.5).to(device)
recall_metric = BinaryRecall(threshold=0.5).to(device)
f1_metric = BinaryF1Score(threshold=0.5).to(device)
auroc_metric = BinaryAUROC().to(device)

transform = T.Compose([ 
    CLAHETransform(),
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor()   #Required!
]) 

dataset = ImageFolder('train_crc', transform=transform)
size = len(dataset)     #408  (204 mss + 204 msi)
train_data_len = int(size*0.8)
valid_data_len = int(size - train_data_len)
train_data, val_data = random_split(dataset, [train_data_len, valid_data_len])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)          # 11 batches (326/32) each of size 32

    model.train()
    running_loss = 0
    avg_loss = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model(inputs)    #torch.Size[32, 1]
        labels = labels.float()    #torch.Size[32]
        preds = outputs[:, 0]      #torch.Size[32]
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Compute the loss and its gradients
        loss = loss_fn(preds, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        #print('Loss for batch ', i, ' = ', loss.item())

    avg_loss = running_loss/size
    print("Average loss in this epoch =", avg_loss)


def test_loop(dataloader, model):

    model.eval()
    num_batches = len(dataloader)
    accuracy = 0
    total_acc = 0

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_auroc = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = model(inputs)
            preds = outputs[:, 0]      #torch.Size[32]
            preds = torch.sigmoid(preds)
            accuracy = (preds.round() == labels).float().mean()  #Mean accuracy value per batch
            precision = precision_metric(preds, labels)
            recall = recall_metric(preds, labels)
            f1_score = f1_metric(preds, labels)
            auroc = auroc_metric(preds, labels)

            print('Average accuracy for batch ', i, ' = ', accuracy.item())
            total_acc += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score
            total_auroc += auroc

            all_preds.append(preds.round().cpu())
            all_labels.append(labels.cpu())

    total_acc /= num_batches
    print('Total Average Accuracy = ', total_acc.item())

    total_precision /= num_batches
    total_recall /= num_batches
    total_f1 /= num_batches
    total_auroc /= num_batches

    print(f'Total Average Precision: {total_precision.item():.4f}')
    print(f'Total Average Recall: {total_recall.item():.4f}')
    print(f'Total Average F1 Score: {total_f1.item():.4f}')
    print(f'Total Average AUROC: {total_auroc.item():.4f}')

# Confusion Matrix
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    cm = confusion_matrix(all_preds, all_labels, task='binary')
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_loop(train_loader, model, loss_fn, optimizer)

test_loop(valid_loader, model)


# Save the model
torch.save(model.state_dict(), 'resnet50CLAHE.pth')