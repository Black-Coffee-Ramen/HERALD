"""
scripts/train_char_transformer.py

Trains a character-level transformer model on domain names.
Ensembles it with HERALD v7 (Hybrid model).
"""

import pandas as pd
import json
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import joblib

# STEP 1 — Data prep
print("Loading dataset...")
df = pd.read_csv('data/processed/full_features_v7.csv')
domains = df['domain'].astype(str).str.lower().tolist()
labels = (df['label'] == 'Phishing').astype(int).tolist()

# Build character vocabulary
print("Building vocabulary...")
all_chars = set(''.join(domains))
vocab = {'[PAD]': 0, '[CLS]': 1, '[UNK]': 2}
for i, c in enumerate(sorted(all_chars), start=3):
    vocab[c] = i

os.makedirs('models', exist_ok=True)
with open('models/char_vocab.json', 'w') as f:
    json.dump(vocab, f)

MAX_LEN = 64

def encode(domain):
    tokens = [vocab['[CLS]']] + [vocab.get(c, vocab['[UNK]']) 
              for c in domain[:MAX_LEN-1]]
    tokens += [vocab['[PAD]']] * (MAX_LEN - len(tokens))
    return tokens[:MAX_LEN]

class DomainDataset(Dataset):
    def __init__(self, domains, labels):
        self.X = [encode(d) for d in domains]
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))

print("Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    domains, labels, test_size=0.2, random_state=42, 
    stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42,
    stratify=y_temp)

train_loader = DataLoader(DomainDataset(X_train, y_train), 
                          batch_size=512, shuffle=True)
val_loader = DataLoader(DomainDataset(X_val, y_val), 
                        batch_size=512)
test_loader = DataLoader(DomainDataset(X_test, y_test), 
                         batch_size=512)

# STEP 2 — Model architecture
class DomainTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64,
                 num_heads=4, num_layers=2, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, 
                                      padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=256, dropout=0.1,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        positions = torch.arange(
            x.size(1), device=x.device).unsqueeze(0)
        padding_mask = (x == 0)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x, 
            src_key_padding_mask=padding_mask)
        # Use [CLS] token representation for classification
        x = x[:, 0, :]
        return self.classifier(x)

vocab_size = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() 
                       else 'cpu')
print(f'Training on: {device}')
model = DomainTransformer(vocab_size).to(device)

# STEP 3 — Training
n_legit = labels.count(0)
n_phish = labels.count(1)
weight = torch.tensor([1.0, n_legit/n_phish]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, factor=0.5)

best_f1 = 0
patience_counter = 0
PATIENCE = 4

print("Starting training...")
for epoch in range(25):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            X_batch = X_batch.to(device)
            out = model(X_batch)
            probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_p = precision_score(all_labels, all_preds, zero_division=0)
    val_r = recall_score(all_labels, all_preds, zero_division=0)
    scheduler.step(1 - val_f1)
    
    print(f'Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} '
          f'| Val P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f}')
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'models/domain_transformer.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch+1}')
            break

# STEP 4 — Test evaluation
print("\nFinal Test Evaluation...")
model.load_state_dict(torch.load('models/domain_transformer.pt',
                                  map_location=device))
model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(device)
        out = model(X_batch)
        probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())

print()
print('=== Transformer Test Results ===')
results_rows = []
for thr in [0.35, 0.45, 0.50, 0.55, 0.65]:
    preds = (np.array(all_probs) >= thr).astype(int)
    p = precision_score(all_labels, preds, zero_division=0)
    r = recall_score(all_labels, preds, zero_division=0)
    f = f1_score(all_labels, preds, zero_division=0)
    print(f'Threshold {thr}: P={p:.4f} R={r:.4f} F1={f:.4f}')
    results_rows.append({'Model': 'Transformer', 'Threshold': thr, 'Precision': p, 'Recall': r, 'F1': f})

# STEP 5 — Hybrid ensemble with v7
print()
print('=== Hybrid: Transformer + v7 XGBoost ===')
v7 = joblib.load('models/ensemble_v7.joblib')
feats = v7['features']

# Get v7 scores on same test domains
test_domains_list = X_test
df_test = pd.DataFrame([{'domain': d} for d in test_domains_list])

sys.path.insert(0, '.')
from herald.features.lexical_features import extract_url_features
print("Extracting features for v7 comparison...")
feat_df = extract_url_features(df_test, domain_col='domain')
# Align features
X_lex = feat_df.reindex(columns=feats, fill_value=0)
print("Running v7 predictions...")
v7_probs = (v7['rf'].predict_proba(X_lex)[:,1] * 0.4 +
            v7['xgb'].predict_proba(X_lex)[:,1] * 0.6)

transformer_probs = np.array(all_probs)
for w_trans in [0.3, 0.5, 0.7]:
    w_v7 = 1 - w_trans
    hybrid = w_trans * transformer_probs + w_v7 * v7_probs
    for thr in [0.45, 0.55]:
        preds = (hybrid >= thr).astype(int)
        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        f = f1_score(all_labels, preds, zero_division=0)
        print(f'Trans={w_trans} V7={w_v7} Thr={thr}: '
              f'P={p:.4f} R={r:.4f} F1={f:.4f}')
        results_rows.append({
            'Model': f'Hybrid (w_trans={w_trans})', 
            'Threshold': thr, 
            'Precision': p, 
            'Recall': r, 
            'F1': f
        })

# Save results
pd.DataFrame(results_rows).to_csv('outputs/transformer_results.csv', index=False)
print("\nResults saved to outputs/transformer_results.csv")
