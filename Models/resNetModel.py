import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# ===============================================
# Preprocessing
# ===============================================

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize(256),                            # Resize the image to 256x256
    transforms.RandomCrop(224),                        # Randomly crop the image to 224x224
    transforms.RandomHorizontalFlip(),                 # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),                   # Randomly flip the image vertically
    transforms.ToTensor()                              # Convert the image to tensor
])

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(root='./Data/train', transform=train_transforms)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# ===============================================
# Model Creation
# ===============================================

# Define the model
class CancerClassifier(pl.LightningModule):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.model = torchvision.models.resnet18()  # Use a pre-trained ResNet-18 model
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),   # Add dropout for regularization
            nn.Linear(num_ftrs, 1)  # Output layer with a single neuron for binary classification
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        trainer = pl
        images, labels = batch
        outputs = self(images).squeeze()
        loss = nn.BCEWithLogitsLoss()(outputs, labels.float())  # Compute binary cross-entropy loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)  # Adam optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Learning rate scheduler
        return [optimizer], [scheduler]

# ===============================================
# Model Training
# ===============================================

# Create a checkpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitor validation loss
    mode='min',          # Save the model with minimum validation loss
    save_top_k=1,        # Save the top 1 model
    save_last=True,      # Also save the last model
)

# Create a logger for TensorBoard
logger = TensorBoardLogger("tb_logs", name="cancer_classifier")

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Initialize the trainer
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1, callbacks=[checkpoint_callback, lr_monitor], logger=logger)

model = CancerClassifier()

# Load or train the model
if os.path.exists('best_model.pth'):
    print("Loading model from best_model.pth")
    model.load_state_dict(torch.load('best_model.pth'))
else:
    print("Training the model")
    # Train the model
    trainer.fit(model, train_loader)
    # Save the model
    torch.save(model.state_dict(), 'best_model.pth')

# ===============================================
# Model Validation/Analysis
# ===============================================

def test_model(model, test_data_path):
    # Load test dataset
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_path, transform=train_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Put the model in evaluation mode
    model.eval()
    
    # Perform evaluation on the test set
    test_accuracy = torchmetrics.Accuracy(task="binary").to('cuda:0')
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    all_preds = []
    all_labels = []
    all_images = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images = images.to("cuda:0")  # Move images to the GPU
            labels = labels.to("cuda:0")  # Move labels to the GPU
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold at 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            test_accuracy.update(preds.float(), labels.float())
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss*100:.2f}%')
    test_accuracy = test_accuracy.compute() * 100
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    return all_preds, all_labels, all_images

def visualize_predictions(images, preds, labels, num_samples=8, save_to_file=False, file_prefix="prediction"):
    num_batches = len(images) // num_samples

    for batch_idx in range(num_batches):
        batch_images = images[batch_idx * num_samples:(batch_idx + 1) * num_samples]
        batch_preds = preds[batch_idx * num_samples:(batch_idx + 1) * num_samples]
        batch_labels = labels[batch_idx * num_samples:(batch_idx + 1) * num_samples]

        if not save_to_file:
            # Display images and predictions
            fig = plt.figure(figsize=(20, 7))
            for idx in range(min(num_samples, len(batch_images))):
                ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
                imshow(batch_images[idx])
                actual_label = "True" if batch_labels[idx] == 1 else "False"
                ax.set_title(f'Predicted: {batch_preds[idx]}\nActual: {actual_label}')
            plt.show()
        else:
            # Save images and predictions to file
            for idx in range(len(batch_images)):
                plt.figure()
                imshow(batch_images[idx])
                actual_label = "True" if batch_labels[idx] == 1 else "False"
                plt.title(f'Predicted: {batch_preds[idx]}\nActual: {actual_label}')
                plt.savefig(f'./predicted/{file_prefix}_batch{batch_idx}_img{idx}.png')
                plt.close()

# Function to display images
def imshow(inp, title=None):
    inp = inp.transpose((1, 2, 0))  # Transpose the image to (H, W, C) format
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model = model.to('cuda:0')

# Test the model
all_preds, all_labels, all_images = test_model(model, './Data/test')

# Visualize predictions
visualize_predictions(all_images, all_preds, all_labels)

# Print some predictions for review
print("Predictions:", all_preds)

# Print actual labels
all_labels_str = [False if label == 0 else True for label in all_labels]
print("Actual Labels:", all_labels_str)

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix: ")
print(conf_matrix)

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
