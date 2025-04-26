import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
from datetime import datetime
import csv
from eeg_models import EEGNet


# ----------------------------
# Configuration
# ----------------------------
DATA_PKL        = 'preprocessed/processed_data_v4.pkl'
ARCH_NAME       = 'enhanced'        # 'baseline', 'deep', or 'band'
MODE            = 'binary'          # 'binary' or 'multiclass'
TEST_SIZE       = 0.2
BATCH_SIZE      = 32
EPOCHS          = 100
PATIENCE        = 100
LR              = 1e-4
OUTPUT_MODEL_DIR= 'models'
LOG_CSV         = os.path.join(OUTPUT_MODEL_DIR, 'training_log.csv')

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# ----------------------------
# Checkpoint settings
# ----------------------------
# model_filename = 'deep_binary_e100_b32_lr0.001_20250424_172540.pt'

try:
    model_filename
except NameError:
    date_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{ARCH_NAME}_{MODE}_e{EPOCHS}_b{BATCH_SIZE}_lr{LR}_{date_stamp}.pt"
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, model_filename)
run_id = os.path.splitext(model_filename)[0].split('_')[-1]

prev_epochs = 0
if os.path.isfile(LOG_CSV):
    with open(LOG_CSV, 'r', newline='') as f:
        for log in csv.DictReader(f):
            if log.get('run_id') == run_id:
                prev_epochs = int(log.get('epochs', 0)); break
start_epoch = prev_epochs
total_epochs = start_epoch + EPOCHS

# ----------------------------
# Dataset Definitions
# ----------------------------
class FlashEEGDataset(Dataset):
    def __init__(self, data_pkl, mode='binary'):
        with open(data_pkl, 'rb') as f:
            trials = pickle.load(f)
        X_all, y_all = [], []
        for trial in trials:
            X_all.extend(trial['X'])
            if mode == 'binary': y_all.extend(trial['y'])
            else:
                mapping = {'Forward':0,'Backward':1,'Left':2,'Right':3}
                idx = mapping[trial['focus']]
                y_all.extend([idx]*len(trial['X']))
        self.X = np.array(X_all, dtype=np.float32)
        self.y = np.array(y_all, dtype=np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class BandpowerDataset(Dataset):
    def __init__(self, data_pkl, mode='binary'):
        with open(data_pkl, 'rb') as f:
            trials = pickle.load(f)
        X_all, y_all = [], []
        for trial in trials:
            X_all.extend(trial['bandpower'])
            if mode == 'binary': y_all.extend(trial['y'])
            else:
                mapping = {'Forward':0,'Backward':1,'Left':2,'Right':3}
                idx = mapping[trial['focus']]
                y_all.extend([idx]*len(trial['bandpower']))
        self.X = np.array(X_all, dtype=np.float32)
        self.y = np.array(y_all, dtype=np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ----------------------------
# Model Architectures
# ----------------------------
class BaselineP300CNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, 3)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.fc1   = nn.Linear(64 * 23, 64)
        self.fc2   = nn.Linear(64, num_classes)
        self.num_classes = num_classes
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1) if self.num_classes==1 else x

class DeepP300CNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64,128,3)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1   = nn.Linear(128*10,256)
        self.fc2   = nn.Linear(256,num_classes)
        self.num_classes = num_classes
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1) if self.num_classes==1 else x

class BandpowerCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,3)
        self.pool  = nn.MaxPool1d(2)
        self.fc1   = nn.Linear(16*15,64)
        self.fc2   = nn.Linear(64,num_classes)
        self.num_classes = num_classes
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1) if self.num_classes==1 else x

# --- New enhanced CNN: more conv layers, dropout, batchnorm
class EnhancedP300CNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(128*12,128)
        self.fc2   = nn.Linear(128,num_classes)
        self.num_classes = num_classes
    def forward(self, x):
        x = x.permute(0,2,1)                  # (B, C, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1) if self.num_classes==1 else x
    
# from braindecode or pytorch‐eeg libraries
model = EEGNet(
    in_chans=8,            # your channels
    n_classes=1,           # binary
    input_window_samples=100  # 0.4s you’re using
)


ARCHITECTURES = {
    'baseline': BaselineP300CNN,
    'deep':     DeepP300CNN,
    'band':     BandpowerCNN,
    'enhanced': EnhancedP300CNN
}

# ----------------------------
# Focal loss
# ----------------------------
def focal_loss(logits, targets, alpha=0.5, gamma=1.0):
    """Binary focal loss."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p*targets + (1-p)*(1-targets)
    loss = ce * (1 - p_t)**gamma
    alpha_t = alpha*targets + (1-alpha)*(1-targets)
    return (alpha_t * loss).mean()

# ----------------------------
# Training & Validation
# ----------------------------
def train_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Select dataset
    if ARCH_NAME == 'band':
        ds = BandpowerDataset(DATA_PKL, mode=MODE)
    else:
        ds = FlashEEGDataset(DATA_PKL, mode=MODE)
    X, y = ds.X, ds.y

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42)

    # --- Balanced sampler for training ---
    # weight each class inversely to its frequency
    neg, pos = (y_train==0).sum(), (y_train==1).sum()
    weights = np.where(y_train==1, neg/pos, 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False)

    # Model
    num_classes = 1 if MODE=='binary' else 4
    model = ARCHITECTURES[ARCH_NAME](num_classes).to(device)

    # Criterion: focal loss for binary, cross-entropy for multiclass
    if MODE == 'binary':
        criterion = focal_loss
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=LR)

    # Early stopping on AUC
    best_val_auc, wait = 0.0, 0

    for epoch_offset in range(1, EPOCHS+1):
        epoch_num = start_epoch + epoch_offset

        # Train
        model.train()
        train_sum = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_sum += loss.item() * xb.size(0)
        train_loss = train_sum / len(tr_loader.dataset)

        # Validate
        model.eval()
        val_sum, preds, labs, probs = 0.0, [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                val_sum += criterion(logits, yb).item() * xb.size(0)
                p = torch.sigmoid(logits).cpu().numpy()
                pr = (p >= 0.5).astype(int)
                preds.extend(pr.tolist()); labs.extend(yb.cpu().numpy()); probs.extend(p.tolist())
        val_loss = val_sum / len(val_loader.dataset)
        val_acc  = accuracy_score(labs, preds)
        prec     = precision_score(labs, preds, zero_division=0)
        rec      = recall_score(labs, preds, zero_division=0)
        val_auc  = roc_auc_score(labs, probs) if MODE=='binary' else None

        print(f"Epoch {epoch_num}/{total_epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"acc={val_acc:.4f} prec={prec:.4f} rec={rec:.4f} "
              + (f"auc={val_auc:.4f}" if val_auc is not None else ""))

        # Early stop on AUC
        if MODE=='binary':
            if val_auc > best_val_auc:
                best_val_auc, wait = val_auc, 0
                torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            else:
                wait += 1
        else:
            # fall back to loss-based for multiclass
            if val_loss < best_val_auc or epoch_offset==1:
                best_val_auc, wait = val_loss, 0
                torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            else:
                wait += 1

        if wait >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Best checkpoint saved to {OUTPUT_MODEL_PATH}")

# ----------------------------
# Training & Validation
# ----------------------------
def train_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Select dataset based on architecture
    if ARCH_NAME == 'band':
        ds = BandpowerDataset(DATA_PKL, mode=MODE)
    else:
        ds = FlashEEGDataset(DATA_PKL, mode=MODE)
    X, y = ds.X, ds.y
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=42)
    
    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE)

    num_classes = 1 if MODE=='binary' else 4
    model = ARCHITECTURES[ARCH_NAME](num_classes).to(device)

    if MODE == 'binary':
        num_pos = int((y_train == 1).sum())
        num_neg = int((y_train == 0).sum())
        pos_weight = torch.tensor(num_neg/num_pos, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_loss, wait = float('inf'), 0
    for epoch_offset in range(1, EPOCHS+1):
        epoch_num = start_epoch + epoch_offset
        # Training loop
        model.train(); train_sum = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_sum += loss.item() * xb.size(0)
        train_loss = train_sum / len(tr_loader.dataset)

        # Validation loop
        model.eval(); val_sum, preds, labs, probs = 0.0, [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                logits = model(xb)
                val_sum += criterion(logits, yb).item() * xb.size(0)
                p = torch.sigmoid(logits).cpu().numpy()
                pr = (p >= 0.5).astype(int)
                preds.extend(pr.tolist()); labs.extend(yb.cpu().numpy()); probs.extend(p.tolist())
        val_loss = val_sum / len(val_loader.dataset)
        val_acc  = accuracy_score(labs, preds)
        prec     = precision_score(labs, preds, zero_division=0)
        rec      = recall_score(labs, preds, zero_division=0)
        auc      = roc_auc_score(labs, probs) if MODE=='binary' else None
        print(f"Epoch {epoch_num}/{total_epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} prec={prec:.4f} "
              f"rec={rec:.4f} " + (f"auc={auc:.4f}" if auc else ""))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best checkpoint saved to {OUTPUT_MODEL_PATH}")

        # Final evaluation
    model.eval(); val_sum, all_preds, all_labels, all_probs = 0.0, [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            val_sum += criterion(logits, yb).item() * xb.size(0)
            p = torch.sigmoid(logits).cpu().numpy()
            pr = (p >= 0.5).astype(int)
            all_preds.extend(pr.tolist()); all_labels.extend(yb.cpu().numpy()); all_probs.extend(p.tolist())
    val_loss = val_sum / len(val_loader.dataset)
    val_acc  = accuracy_score(all_labels, all_preds)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

    # Log run (update or append)
    header = ['run_id','timestamp','arch','mode','test_size','batch_size','epochs','lr','val_loss','val_acc','device','model_path']
    row = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'arch': ARCH_NAME,
        'mode': MODE,
        'test_size': TEST_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': total_epochs,
        'lr': LR,
        'val_loss': f"{val_loss:.4f}",
        'val_acc': f"{val_acc:.4f}",
        'device': str(device),
        'model_path': OUTPUT_MODEL_PATH
    }
    logs = []
    if os.path.isfile(LOG_CSV):
        with open(LOG_CSV, 'r', newline='') as f:
            logs = list(csv.DictReader(f))
    updated = False
    for log in logs:
        if log.get('run_id') == run_id:
            log.update(row)
            updated = True
            break
    if not updated:
        logs.append(row)
    cleaned = [{k: log.get(k, '') for k in header} for log in logs]
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(cleaned)
    print(f"Logged metrics to {LOG_CSV}")

if __name__=='__main__':
    train_eval()
