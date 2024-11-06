import torch
import torchvision.transforms as T
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from torchmetrics.functional import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Learning and training parameters.
epochs = 20
batch_size = 16
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model
model = vgg16(pretrained=False)

# Modify the classifier to match your output
model.classifier[6] = nn.Linear(in_features=4096, out_features=1)

# Move the model to the device
model = model.to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

# Metrics
precision_metric = BinaryPrecision(threshold=0.5)
recall_metric = BinaryRecall(threshold=0.5)
f1_metric = BinaryF1Score(threshold=0.5)
auroc_metric = BinaryAUROC()

# Data transformation
transform = T.Compose([ 
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
]) 

dataset = ImageFolder('train_crc', transform=transform)
size = len(dataset)     # 408  (204 mss + 204 msi)
train_data_len = int(size*0.8)
valid_data_len = int(size - train_data_len)
train_data, val_data = random_split(dataset, [train_data_len, valid_data_len])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    running_loss = 0
    avg_loss = 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        labels = labels.float()
        preds = outputs[:, 0]
        
        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / size
    print("Average loss in this epoch =", avg_loss)

def test_loop(dataloader, model):
    model.eval()
    num_batches = len(dataloader)
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs[:, 0]
            preds = torch.sigmoid(preds)
            accuracy = (preds.round() == labels).float().mean()
            total_acc += accuracy

            precision = precision_metric(preds, labels)
            recall = recall_metric(preds, labels)
            f1_score = f1_metric(preds, labels)
            auroc = auroc_metric(preds, labels)

            total_precision += precision
            total_recall += recall
            total_f1 += f1_score
            total_auroc += auroc

            all_preds.append(preds.round().cpu())
            all_labels.append(labels.cpu())

    total_acc /= num_batches
    total_precision /= num_batches
    total_recall /= num_batches
    total_f1 /= num_batches
    total_auroc /= num_batches

    print('Total Average Accuracy = ', total_acc.item())
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

    # Save the plot as an image file
    plt.savefig('VGG-16_CM.png', dpi=300, bbox_inches='tight')

    # Optionally, clear the current figure to avoid overlap in future plots
    plt.clf()


# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_loop(train_loader, model, loss_fn, optimizer)

test_loop(valid_loader, model)

# Save the model
torch.save(model.state_dict(), 'vgg16.pth')
