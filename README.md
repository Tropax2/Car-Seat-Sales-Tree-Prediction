# Sales Prediction using Trees

In this project we use several tree regressor models to predict the number of car seat sales using data from 400 stores. The dataset is included in the files since it comes from the book Introduction to Statistical Learning by James, Witten, Hastie, Tibshirani and Taylor.

## Project Structure 
    - `src/` - source code for model and evaluations;
    - `reports` - detailed report and results obtained;

## Dataset 
The dataset includes predictors capturing competitive and company pricing, advertising spend, and community demographics, as well as store placement and location indicators.
The response variable is the target variable representing the number of sales.

## Methods 
Categorical predictors are one-hot-encoded and numerical predictors are standerdised. The following models are evaluated:

    - Decision Tree Regressor (DTR);
    - Bagging of DTRs;
    - Random Forest of DTR.

The models are evaluated by a validation set approach by computing the test MSE of each one, where the optimal hyperparameters are found by cross-validation.

## Results Summary

**Test MSE (lower is better):**
- **Regression Tree (pruned):** 5.28  
- **Bagging (500 trees):** 3.44 *(best)*  
- **Random Forest (500 trees, all features per split):** 3.45   
- **Random Forest (max_features ≈ √p = 4):** 3.93 

Ensemble methods (bagging/random forest) substantially improved performance over a single regression tree. The most important features were **Price**, **ShelveLoc_Good**, and **Age**.

## How to run 

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

### 2) Install Dependencies
```bash 
pip install -r requirements.txt
```

### 3) Run the Project 
```bash
python src/main.py
```

