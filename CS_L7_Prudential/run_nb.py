import sys
import os

# Automatically download data if running in Google Colab
if 'google.colab' in sys.modules:
    print("Running in Google Colab: Fetching Data-Science-Portfolio from GitHub...")
    #!git clone https://github.com/Llander22/Data-Science-Portfolio.git
    
    # Move the Colab virtual machine into this specific project folder
    os.chdir('Data-Science-Portfolio/CS_L7_Prudential')
    print("\nData download complete! You are ready to execute the notebook.")
else:
    print("Running Locally: Data already available.")
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import  make_column_transformer

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
y_train = y_train - 1  # XGBoost strictly requires labels 0-7, not 1-8
X_test = pd.read_csv("data/X_test.csv")

categories = ["Product_Info_1", "Product_Info_2", "Product_Info_3",
              "Product_Info_5", "Product_Info_6", "Product_Info_7"]

preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categories))
    
model = make_pipeline(preprocessor, DecisionTreeClassifier())

model.fit(X_train, y_train)

y_pred = model.predict(X_test) + 1  # Add 1 back to satisfy KATE grader
RUN_BASELINE_CV = True

if RUN_BASELINE_CV:
    # Execute 5-fold cross-validation to rigorously score the baseline model
    # Note: y_train is a dataframe, so we use .values.ravel() to flatten it into the 1D array scikit-learn expects
    baseline_scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5, scoring='accuracy')

    print(f"Baseline Accuracy (5-fold CV): {np.mean(baseline_scores):.4f} (+/- {np.std(baseline_scores) * 2:.4f})")

# 1. Explicitly map out ALL categorical features defined in the dataset instructions
categorical_features = [
    "Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", 
    "Product_Info_6", "Product_Info_7", "Employment_Info_2", "Employment_Info_3", 
    "Employment_Info_5", "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", 
    "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", 
    "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", 
    "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", 
    "Insurance_History_9", "Family_Hist_1", "Medical_History_2", "Medical_History_3", 
    "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", 
    "Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12", 
    "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", 
    "Medical_History_18", "Medical_History_19", "Medical_History_20", "Medical_History_21", 
    "Medical_History_22", "Medical_History_23", "Medical_History_25", "Medical_History_26", 
    "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30", 
    "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", 
    "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", 
    "Medical_History_40", "Medical_History_41"
]

# 2. Build the fault-tolerant Preprocessor
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    remainder="passthrough"
)

print(f"Preprocessor successfully built. It will process {len(categorical_features)} categorical columns.")

# 1. Instantiate the ensemble model with default parameters for maximum speed
# Setting random_state ensures the same result every time
fast_model = XGBClassifier(tree_method='hist', device='cuda', random_state=42)

# 2. Build the final pipeline
final_pipeline = make_pipeline(preprocessor, fast_model)
RUN_NEW_MODEL_CV = True

if RUN_NEW_MODEL_CV:
    print("Running 5-Fold Cross Validation to check our new model's accuracy...")

    # Score our new pipeline
    new_model_scores = cross_val_score(final_pipeline, X_train, y_train.values.ravel(), cv=5, scoring='accuracy')

    print(f"New Model Accuracy: {np.mean(new_model_scores):.4f} (+/- {np.std(new_model_scores) * 2:.4f})")

# Create explicit interaction features on both training and test sets
X_train_eng = X_train.copy()
X_test_eng = X_test.copy()

X_train_eng['Age_BMI_Risk'] = X_train_eng['Ins_Age'] * X_train_eng['BMI']
X_test_eng['Age_BMI_Risk'] = X_test_eng['Ins_Age'] * X_test_eng['BMI']

print(f"Original feature count: {X_train.shape[1]}")
print(f"Engineered feature count: {X_train_eng.shape[1]}")

#!pip install -q optuna catboost xgboost

RUN_OPTUNA_STUDY = True

if RUN_OPTUNA_STUDY:
    import optuna
    from optuna.samplers import TPESampler
    import warnings
    from sklearn.model_selection import train_test_split
    from catboost import CatBoostClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("--- ATTEMPT 1: UNSTRATIFIED OPTUNA SEARCH ---")
    print("Extracting a 30% random subsample (Unstratified)...")
    X_tune_unstrat, _, y_tune_unstrat, _ = train_test_split(X_train_eng, y_train, train_size=0.3, stratify=None, random_state=42)
    
    def objective_unstrat(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'iterations': trial.suggest_int('iterations', 50, 150),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_state': 42,
            'task_type': 'GPU',
            'verbose': False
        }
        trial_model = CatBoostClassifier(**param)
        trial_pipeline = make_pipeline(preprocessor, trial_model)
        score = cross_val_score(trial_pipeline, X_tune_unstrat, y_tune_unstrat.values.ravel(), cv=3, scoring='accuracy')
        return score.mean()
        
    study_unstrat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_unstrat.optimize(objective_unstrat, n_trials=15)
    
    print("\n--- ATTEMPT 2: CORRECTED STRATIFIED OPTUNA SEARCH ---")
    print("Extracting a 30% random subsample (Stratified)...")
    X_tune_strat, _, y_tune_strat, _ = train_test_split(X_train_eng, y_train, train_size=0.3, stratify=y_train, random_state=42)
    
    def objective_strat(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'iterations': trial.suggest_int('iterations', 50, 150),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_state': 42,
            'task_type': 'GPU',
            'verbose': False
        }
        trial_model = CatBoostClassifier(**param)
        trial_pipeline = make_pipeline(preprocessor, trial_model)
        score = cross_val_score(trial_pipeline, X_tune_strat, y_tune_strat.values.ravel(), cv=3, scoring='accuracy')
        return score.mean()
        
    study_strat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_strat.optimize(objective_strat, n_trials=15)
    
    print(f"\n--- OPTIMIZATION COMPLETE ---")
    print(f"Attempt 1 (Unstratified) Optimal Parameters: {study_unstrat.best_params}")
    print(f"Attempt 2 (Stratified) Optimal Parameters: {study_strat.best_params}")

# 1. Hardcode the absolute optimal parameters found by Stratified Optuna
best_params = {
    'learning_rate': 0.272492924214504,
    'depth': 4,
    'iterations': 100,
    'l2_leaf_reg': 0.6146439247862419,
    'random_state': 42,
    'task_type': 'GPU',
    'verbose': False
}

print("Instantiating Optimized Cycle 2 Model...")
from catboost import CatBoostClassifier
# 2. Instantiate the model with the hardcoded optimal parameters
optimized_model = CatBoostClassifier(**best_params)

# 3. Build the pipeline
final_optimized_pipeline = make_pipeline(preprocessor, optimized_model)

RUN_FINAL_CV = True

if RUN_FINAL_CV:
    from sklearn.model_selection import cross_val_score
    import numpy as np

    print("Running 5-Fold Cross-Validation on Final Optimized Architecture...")
    # Score our optimized pipeline on the full engineered dataset
    final_scores = cross_val_score(final_optimized_pipeline, X_train_eng, y_train.values.ravel(), cv=5, scoring='accuracy')

    print(f"Final Cycle 2 Accuracy: {np.mean(final_scores):.4f} (+/- {np.std(final_scores) * 2:.4f})")

RUN_ENSEMBLE = True

if RUN_ENSEMBLE:
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    print("Instantiating Cycle 3 Hybrid GPU Ensemble A/B Testing...")
    
    # 1. Our Optuna-Tuned CatBoost Model (Locked!)
    model_1 = CatBoostClassifier(
        task_type='GPU',
        learning_rate=0.272492924214504,
        depth=4,
        iterations=100,
        l2_leaf_reg=0.6146439247862419,
        random_state=42,
        verbose=False
    )
    
    # --- TEST 1: Baseline XGBoost (depth=5) ---
    print("\n--- TEST 1: XGBoost Depth=5 ---")
    xgb_test1 = XGBClassifier(
        device='cuda',
        learning_rate=0.1,  # Standard robust baseline LR
        max_depth=5,
        n_estimators=100,
        random_state=42,
        tree_method='hist'
    )
    
    ensemble_1 = VotingClassifier(
        estimators=[('catboost', model_1), ('xgb', xgb_test1)],
        voting='soft'
    )
    
    ensemble_pipeline_1 = make_pipeline(preprocessor, ensemble_1)
    print("Running 5-Fold Cross-Validation...")
    scores_1 = cross_val_score(ensemble_pipeline_1, X_train_eng, y_train.values.ravel(), cv=5, scoring='accuracy')
    print(f"Test 1 Accuracy: {np.mean(scores_1):.4f} (+/- {np.std(scores_1) * 2:.4f})")

    # --- TEST 2: Deep XGBoost (depth=6) ---
    print("\n--- TEST 2: XGBoost Depth=6 ---")
    xgb_test2 = XGBClassifier(
        device='cuda',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=42,
        tree_method='hist'
    )
    
    ensemble_2 = VotingClassifier(
        estimators=[('catboost', model_1), ('xgb', xgb_test2)],
        voting='soft'
    )
    
    ensemble_pipeline_2 = make_pipeline(preprocessor, ensemble_2)
    print("Running 5-Fold Cross-Validation...")
    scores_2 = cross_val_score(ensemble_pipeline_2, X_train_eng, y_train.values.ravel(), cv=5, scoring='accuracy')
    print(f"Test 2 Accuracy: {np.mean(scores_2):.4f} (+/- {np.std(scores_2) * 2:.4f})")

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Executing Confusion Matrix Test to Prove Ordinality...")

# We use the fast_model (HistGradientBoosting) from Cycle 1 to quickly generate 
# 3-Fold Cross-Validated predictions across the entire training set.
y_cv_pred = cross_val_predict(final_pipeline, X_train, y_train.values.ravel(), cv=3)

fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay.from_predictions(
    y_train.values.ravel(), 
    y_cv_pred, 
    cmap='Blues', 
    ax=ax,
    colorbar=False
)
plt.title("Confusion Matrix: Proving Ordinality via Clustered Errors")
plt.show()
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score
import numpy as np

print("Engineering 'Missingness' Feature...")
# 1. Add NaN_Count (How many medical fields did the patient refuse to answer?)
X_train_eng['NaN_Count'] = X_train_eng.isnull().sum(axis=1)
X_test_eng['NaN_Count'] = X_test_eng.isnull().sum(axis=1)

print("Instantiating Deterministic Ordinal Regressor Architecture...")
# 2. Swap to CatBoost REGRESSOR! (We use CPU here to guarantee exact repeatability for the markdown)
regressor_model = CatBoostRegressor(task_type='CPU', random_state=42, verbose=False)

# 3. Wrap it in the preprocessor
regression_pipeline = make_pipeline(preprocessor, regressor_model)

print("Training Regressor...")
# 4. Train the Regressor
regression_pipeline.fit(X_train_eng, y_train.values.ravel())

print("Generating Continuous Predictions and Thresholding...")
# 5. Generate CONTINUOUS predictions (e.g. 5.8, 3.2, 7.9)
raw_continuous_predictions = regression_pipeline.predict(X_train_eng)

# 6. Snap them back to integers! (Round to nearest whole number, clip between 0 and 7)
final_integer_predictions = np.round(raw_continuous_predictions).clip(0, 7)

# 7. Calculate our new Accuracy!
regression_accuracy = accuracy_score(y_train.values.ravel(), final_integer_predictions)
print(f"\nFinal Cycle 4 Regression Accuracy: {regression_accuracy:.4f}")

RUN_100_PERCENT_OPTUNA = True

if RUN_100_PERCENT_OPTUNA:
    print("Unleashing Optuna on 100% of the Dataset...")
    
    def objective_100_percent(trial):
        # We allow the trees to grow deeper this time!
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 15), # Increased upper limit
            'max_iter': trial.suggest_int('max_iter', 50, 150),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
            'random_state': 42
        }
        
        # Instantiate the champion model
        model = XGBClassifier(tree_method='hist', device='cuda', **params)
        
        # Wrap in preprocessor
        optuna_pipeline = make_pipeline(preprocessor, model)
        
        # Run 3-fold CV on the FULL 100% training set
        # n_jobs=-1 is REMOVED so OpenMP can natively use all 16 cores without Python interfering!
        score = cross_val_score(optuna_pipeline, X_train_eng, y_train.values.ravel(), cv=3, scoring='accuracy')
        
        return score.mean()
    
    print("Initiating 10-Trial Bayesian Search. Spooling up hardware...")
    
    # Create the study and optimize
    study_100 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_100.optimize(objective_100_percent, n_trials=10)
    
    print(f"\nOptimization Complete!")
    print(f"Absolute Mathematical Limit Parameters: \n{study_100.best_params}")
    print(f"Peak CV Accuracy Achieved: {study_100.best_value:.4f}")

from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

print("Instantiating KATE-Optimized XGBoost Champion...")
final_model = XGBClassifier(
    device='cuda', # Forces XGBoost to use the Cloud GPU!
    tree_method='hist',
    n_estimators=117,
    depth=7,
    learning_rate=0.11982423635529445,
    subsample=0.8680425751541432,
    colsample_bytree=0.8119101189073585,
    random_state=42
)

champion_pipeline = make_pipeline(preprocessor, final_model)

print("Training Champion Model on Engineered Dataset...")
# XGBoost rigidly expects classes to start at 0 instead of 1
y_train_adj = y_train.values.ravel() - 1 
champion_pipeline.fit(X_train_eng, y_train_adj)

print("Generating predictions...")
y_pred_adj = champion_pipeline.predict(X_test_eng)

# Re-adjust predictions back to original 1-8 scale for KATE
y_pred = y_pred_adj + 1 
print("SUCCESS! Final predictions are ready for grading.")
