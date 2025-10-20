
# A Multi-Method Approach to Molecular Activity Prediction  

This repository contains the code and analysis for our KTH semester project **“A Multi-Method Approach to Molecular Activity Prediction.”** The goal of this study is to benchmark classical and graph-based machine learning methods for predicting molecular bioactivity across diverse biochemical datasets.

---

##  Overview  
Predicting molecular activity is central to drug discovery and toxicity assessment. Molecules can be represented as graphs, allowing both **handcrafted descriptors** and **learned embeddings** to be compared.  
In this project, we empirically evaluate three major paradigms for molecular representation:

1. **Graph-theoretic descriptors** – global structural metrics (e.g., graph density, clustering coefficient, SPID).  
2. **Molecular fingerprints (ECFP4)** – substructure-based bit vectors generated via RDKit.  
3. **Graph Neural Networks (GNNs)** – learned embeddings using GCNConv and GraphConv architectures (PyTorch Geometric).

We use four benchmark datasets from **TUDataset**: DHFR, AIDS, PTC MR, and Mutagenicity.

---

## Methods  
- **Classical ML Models:** Logistic Regression, SVM, Random Forest, AdaBoost, Gradient Boosting, MLP, and others (implemented in scikit-learn).  
- **Graph Features:** Extracted with NetworkX and RDKit; includes node/edge counts, density, clustering, and shortest-path dispersion (SPID).  
- **GNN Models:**  
  - *GCNConv* (spectral-based convolution)  
  - *GraphConv* (spatial message passing)  
- **Evaluation Metrics:** Accuracy, ROC-AUC, Cohen’s d, and runtime.  
- **Cross-validation:** 5- and 10-fold stratified splits with early stopping and dropout for deep models.

---

## Key Findings  
- **Fingerprints consistently outperform graph-theoretic features** across all datasets.  
- **GraphConv networks outperform GCNConv**, particularly on medium-sized datasets (e.g., DHFR, Mutagenicity).  
- Combining fingerprints and structural features offers **limited performance gain**, suggesting redundancy.  
- Dataset characteristics (imbalance, structure, size) strongly influence which models perform best.  
- **Classical models (e.g., Logistic Regression, SVM)** provide strong baselines at a fraction of the computational cost of GNNs.  

---

## Datasets  
| Dataset | #Graphs | Class Balance | Best Model | Accuracy |
|----------|----------|---------------|-------------|-----------|
| **AIDS** | 2000 | 20/80 | AdaBoost (FP+GT) | 0.999 |
| **DHFR** | 756 | 60/40 | LR (FP) | 0.798 |
| **Mutagenicity** | 4337 | 45/55 | GraphConv | 0.806 |
| **PTC MR** | 344 | 41/59 | SVM (FP) | 0.619 |

---

## Environment  
- Python 3.10  
- RDKit  
- scikit-learn  
- NetworkX  
- PyTorch + PyTorch Geometric  

---
## Citation  
Hölzle, C., Karim, K., Cahyaningrum, K.  
**A Multi-Method Approach to Molecular Activity Prediction**  
KTH Royal Institute of Technology, Technical University of Munich (2025).

---

## Summary  
This repository demonstrates that simple, interpretable molecular fingerprints paired with classical ML models remain **robust and scalable baselines** for molecular activity prediction — rivaling or outperforming early GNNs on standard benchmarks.
