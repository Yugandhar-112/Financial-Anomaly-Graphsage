# Financial Anomaly Detection via GraphSAGE

An end-to-end Graph Neural Network (GNN) framework implemented in PyTorch Geometric designed to handle transaction structural routing pipelines and accurately classify highly imbalanced fraudulent transaction nodes.

<p align="center">
  <img src="https://raw.githubusercontent.com/williamleif/GraphSAGE/master/bipartite.png" alt="GraphSAGE Aggregation Architecture" width="70%">
  <br>
  <em>Figure 1: GraphSAGE Structural Embedding Aggregation Pipeline</em>
</p>

---

## Architecture Design Matrix

The engine creates geometric multi-hop localized representations without keeping complete graphs resident in RAM memory space, utilizing inductive step learning.

| Pipeline Phase | Operations Matrix | Key Objective |
| :--- | :--- | :--- |
| **Data Engg** | Standard Scaling + Synthetic Graph Construction | Feature normalization & graph structuring |
| **Aggregation** | Mean SAGEConv Operator | Neighbor representation pooling |
| **Regularization**| Node feature dropout ($p=0.3$) | Prevents structural over-smoothing |
| **Loss Handler** | Weighted Cross-Entropy Cost Strategy | Mitigates severe target fraud skewness |

---

## Performance Targets & Optimization Metrics

Traditional precision evaluation drops reliability indicators on heavy imbalanced distributions. The baseline evaluation strategy checks these strict matrices across the testing phase:

* **ROC-AUC:** Assesses continuous probability distributions regardless of absolute classification thresholding boundary configurations.
* **F1-Score:** Harmonic target tracking precision versus recall coverage.