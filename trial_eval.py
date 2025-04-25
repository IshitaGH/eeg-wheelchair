import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- User-configurable paths & settings ---
DATA_PKL   = 'preprocessed/processed_data.pkl'  # path to your pickle
MODEL_PATH = 'models/deep_binary_e200_b32_lr1e-05_20250424_160859.pt'  # trained checkpoint
ARCH_NAME  = 'deep'    # 'baseline' or 'deep'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Model architectures ---
class BaselineP300CNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, 3)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.fc1   = nn.Linear(64 * 23, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return torch.sigmoid(out).squeeze(1)

class DeepP300CNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1   = nn.Linear(128 * 10, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        out = self.fc2(x)
        return torch.sigmoid(out).squeeze(1)

ARCHS = {'baseline': BaselineP300CNN, 'deep': DeepP300CNN}
DIRECTIONS = ['Forward', 'Backward', 'Left', 'Right']
DIR2IDX = {d: i for i, d in enumerate(DIRECTIONS)}

# --- Load data ---
with open(DATA_PKL, 'rb') as f:
    trials = pickle.load(f)

# --- Load model ---
model = ARCHS[ARCH_NAME](num_classes=1).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# --- Evaluate at trial-level ---
y_true, y_pred = [], []

for tr in trials:
    X = tr['X']             # (n_epochs, time, ch)
    dirs = tr['flash_dirs']  # list of length n_epochs
    true_dir = tr['focus']   # single string

    # stack into tensor and run through model
    tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        probs = model(tensor).cpu().numpy()  # (n_epochs,)

    # average the probabilities per direction
    mean_scores = {}
    for d in DIRECTIONS:
        mask = np.array([fd == d for fd in dirs])
        mean_scores[d] = probs[mask].mean() if mask.sum() > 0 else 0.0

    # pick the direction with highest mean score
    pred_dir = max(mean_scores, key=mean_scores.get)

    y_true.append(DIR2IDX[true_dir])
    y_pred.append(DIR2IDX[pred_dir])

# compute and print metrics
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
rp = classification_report(y_true, y_pred, target_names=DIRECTIONS)

print(f"Trial-level Accuracy: {acc:.4f}\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(rp)
