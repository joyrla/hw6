# Serverless Cold-Start Prediction Pipeline

Predicts serverless function invocation traffic using a naive baseline (Lag-1) vs an AI model (Amazon Chronos).

**Results**: AI achieves 6.9% lower MAE than baseline (4403.89 vs 4728.18).

---

## Setup

**Requires**: Python 3.9+

```bash
# Install dependencies
pip install pandas numpy pyarrow matplotlib scikit-learn torch chronos-forecasting

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset
# Extract to: data/azure-functions-dataset-2019/
```

---

## Run

```bash
python3 run_pipeline.py
```

Or use the notebook: `notebooks/pipeline_demo.ipynb`

---

## Expected Output

```
Method               | MAE          | Cold Starts 
------------------------------------------------------------
Baseline (Lag-1)     | 4728.18      | 146         
AI (Chronos)         | 4403.89      | 151         

MAE Improvement: 6.9%
```

---

## Reproducibility

- Seeds: `np.random.seed(42)`, `torch.manual_seed(42)`
- Tested on: macOS 26.1, Python 3.9.6, Apple M1 Pro
- Package versions: pandas 2.2.3, numpy 1.24.4, torch 2.8.0, scikit-learn 1.6.1

---

## Files

```
submission/
├── README.md
├── run_pipeline.py
├── report.tex
├── notebooks/pipeline_demo.ipynb
└── figures/
```
