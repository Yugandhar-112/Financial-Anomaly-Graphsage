import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from data_preprocessing import get_financial_dataset
from model import FinancialGraphSAGE

def calculate_metrics(logits, targets, mask):
    """Computes specific optimization KPIs handling high class imbalance."""
    probs = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    preds = logits[mask].argmax(dim=1).detach().cpu().numpy()
    actual = targets[mask].cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(actual, preds, average='binary', zero_division=0)
    auc = roc_auc_score(actual, probs) if len(set(actual)) > 1 else 0.5
    return precision, recall, f1, auc

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_financial_dataset().to(device)
    
    # Calculate negative-to-positive ratio to mitigate extreme class imbalance
    num_neg = sorted(torch.bincount(data.y).tolist())[0]
    num_pos = sorted(torch.bincount(data.y).tolist())[1]
    pos_weight = torch.tensor([num_pos / num_neg]).to(device)
    
    model = FinancialGraphSAGE(in_channels=data.num_features, hidden_channels=32, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(pos_weight.cpu())]).to(device))
    
    print("Starting Training Loop...")
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_prec, val_rec, val_f1, val_auc = calculate_metrics(val_out, data.y, data.val_mask)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val AUC-ROC: {val_auc:.4f}")
                
    # Final Evaluation Check
    model.eval()
    with torch.no_grad():
        final_out = model(data.x, data.edge_index)
        prec, rec, f1, auc = calculate_metrics(final_out, data.y, data.test_mask)
        print("\n=== Final Test Performance ===")
        print(f"Precision: {prec:.4f}\nRecall:    {rec:.4f}\nF1-Score:  {f1:.4f}\nAUC-ROC:   {auc:.4f}")

if __name__ == '__main__':
    train_pipeline()