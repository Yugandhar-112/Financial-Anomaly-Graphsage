import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from data_preprocessing import get_financial_dataset
from model import FinancialGraphSAGE

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def optimize_threshold(logits, targets, mask):
    """Finds the optimal threshold to maximize the true F1-Score with an upper search bound."""
    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    actual = targets[mask].cpu().numpy()
    
    precisions, recalls, thresholds = precision_recall_curve(actual, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Restrict search space to thresholds below 0.5 to prioritize higher recall capture
    valid_idx = (thresholds <= 0.5) & (precisions[:-1] > 0) & (recalls[:-1] > 0)
    if not any(valid_idx):
        return 0.35  # Empirical balanced default for imbalanced graphs
        
    best_idx = np.where(valid_idx, f1_scores[:-1], 0).argmax()
    return thresholds[best_idx]

def calculate_metrics(logits, targets, mask, threshold=0.5):
    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    preds = (probs >= threshold).astype(int)
    actual = targets[mask].cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(actual, preds, average='binary', zero_division=0)
    auc = roc_auc_score(actual, probs) if len(set(actual)) > 1 else 0.5
    return precision, recall, f1, auc

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_financial_dataset().to(device)
    
    model = FinancialGraphSAGE(in_channels=data.num_features, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    print("Starting Optimized Training Loop...")
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 15 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                best_thresh = optimize_threshold(val_out, data.y, data.val_mask)
                _, _, val_f1, val_auc = calculate_metrics(val_out, data.y, data.val_mask, best_thresh)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Threshold: {best_thresh:.4f} | Val F1: {val_f1:.4f} | Val AUC-ROC: {val_auc:.4f}")
                
    model.eval()
    with torch.no_grad():
        final_out = model(data.x, data.edge_index)
        optimal_thresh = optimize_threshold(final_out, data.y, data.val_mask)
        prec, rec, f1, auc = calculate_metrics(final_out, data.y, data.test_mask, optimal_thresh)
        print("\n=== Optimized Test Performance ===")
        print(f"Decision Threshold: {optimal_thresh:.4f}")
        print(f"Precision:          {prec:.4f}\nRecall:             {rec:.4f}\nF1-Score:           {f1:.4f}\nAUC-ROC:            {auc:.4f}")

if __name__ == '__main__':
    train_pipeline()