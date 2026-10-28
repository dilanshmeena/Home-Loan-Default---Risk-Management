# Home Loan Default - Risk Management (PRCP-1006)

A machine learning project that predicts which home loan customers are likely to default, built on the Home Credit dataset.

## Problem

Banks lose money when customers default on loans. This project finds the factors and customer segments linked to default and builds a model to predict it.

- **Target:** `1` = Defaulter, `0` = Not Defaulter
- **Domain:** Banking

## Dataset

[Download PRCP-1006-HomeLoanDef.zip](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip)

7 tables with 307,511 customers: `application_train`, `bureau`, `bureau_balance`, `POS_CASH_balance`, `credit_card_balance`, `previous_application`, `installments_payments`.

The data is not included in this repo because of its size. Unzip the CSV files into a `data/` folder next to the notebook.

## Approach

1. Exploratory data analysis on the main table
2. Summarized each side table to one row per customer and merged all tables
3. Cleaned the data: fixed the wrong `DAYS_EMPLOYED` value, handled missing values, capped extreme incomes, encoded text columns
4. Feature engineering: `EXT_MEAN`, `EXT_MIN`, `CREDIT_TERM`, loan-to-income ratios and more
5. Handled class imbalance (only 8% defaulters) with class weights
6. Trained and compared three models
7. Analyzed feature importance and built customer risk groups

## Results

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.755 | 0.165 | 0.680 | 0.266 |
| Random Forest | 0.752 | 0.169 | 0.659 | 0.270 |
| **LightGBM** | **0.780** | **0.184** | **0.693** | **0.291** |

**Best model: LightGBM.** It catches about 69% of defaulters on the test data.

**Risk groups:** the real default rate rises from 1.3% (Very Low) to 22.5% (Very High), which a bank can use for approval and monitoring decisions.

**Top factors:** external scores (`EXT_MEAN`), loan term, installment underpayment, and past credit history.

## Project Structure

```
├── Home_Loan_Default_Prediction.ipynb
├── data/          (not included, add the CSV files here)
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
jupyter notebook Home_Loan_Default_Prediction.ipynb
```

Run all cells from top to bottom.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, LightGBM
