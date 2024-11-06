import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load the Excel file
file_path = 'Tables/resnet18CLAHEgrey.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Check if 'Prediction' matches 'Original Label'
df['Is_Correct'] = df['Prediction'] == df['Original Label']

# Calculate the number of correct and incorrect predictions
correct = df['Is_Correct'].sum()  # Count of True values
total = len(df)  # Total number of rows

# Calculate accuracy
accuracy = correct / total * 100

# Print the accuracy
print(f"Accuracy: {accuracy:.2f}%")

# Optionally, print the number of correct/incorrect predictions
incorrect = total - correct
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")

# Get confusion matrix, precision, recall, F1-score
y_true = df['Original Label']  # Ground truth labels
y_pred = df['Prediction']  # Predicted labels

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Precision
precision = precision_score(y_true, y_pred, average='weighted')
print(f"\nPrecision: {precision:.2f}")

# Recall
recall = recall_score(y_true, y_pred, average='weighted')
print(f"Recall: {recall:.2f}")

# F1-score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1-Score: {f1:.2f}")
