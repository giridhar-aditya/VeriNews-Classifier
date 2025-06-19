import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import joblib

# Suppress warnings if you want
os.environ['PYTHONWARNINGS'] = 'ignore'

# Load your local dataset CSV (update this path)
DATA_PATH = "news.csv"  
data_df = pd.read_csv(DATA_PATH)

# Keep only text and label columns
data_df = data_df[['text', 'label']].copy()

# Shuffle the data
data_df = shuffle(data_df, random_state=987).reset_index(drop=True)

print("Label counts:\n", data_df['label'].value_counts())

# Prepare texts and labels
texts = data_df['text'].values
labels = data_df['label'].values

# Encode labels to integers
label_enc = LabelEncoder()
labels_enc = label_enc.fit_transform(labels)

# Split 90% train+val, 10% test
texts_trainval, texts_test, labels_trainval, labels_test = train_test_split(
    texts, labels_enc, test_size=0.1, stratify=labels_enc, random_state=987
)

# Further split train+val to 70% train, 20% val (of total data)
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts_trainval, labels_trainval, test_size=0.2222, stratify=labels_trainval, random_state=987
)

# Initialize TF-IDF vectorizer with parameters similar to your original
tfidf_vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 3),
    min_df=4,
    max_df=0.75,
    stop_words='english',
    sublinear_tf=True
)

# Fit on train texts and transform all sets
X_train = tfidf_vectorizer.fit_transform(texts_train).toarray()
X_val = tfidf_vectorizer.transform(texts_val).toarray()
X_test = tfidf_vectorizer.transform(texts_test).toarray()

# Define PyTorch Dataset
class NewsDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create Dataset and DataLoader for train, val, test
train_dataset = NewsDataset(X_train, labels_train)
val_dataset = NewsDataset(X_val, labels_val)
test_dataset = NewsDataset(X_test, labels_test)

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define simple Feedforward NN like your TF model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 80),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

model = SimpleClassifier(input_dim=X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Early stopping parameters
patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0
num_epochs = 120

def evaluate_model(dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    return all_preds, all_labels

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    avg_train_loss = sum(train_losses) / len(train_losses)
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_news_classifier.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Load best model for testing
model.load_state_dict(torch.load("best_news_classifier.pth"))
model.eval()

# Predict on test set
test_preds, test_labels = evaluate_model(test_loader)
test_preds_binary = [1 if p >= 0.5 else 0 for p in test_preds]

print("\nTest Set Classification Report:")
print(classification_report(test_labels, test_preds_binary, target_names=label_enc.classes_))

# Save vectorizer and label encoder for inference
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_enc, "label_encoder.pkl")

print("Model, vectorizer, and label encoder saved.")
