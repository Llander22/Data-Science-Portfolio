import nbformat

nb_path = 'prudential-cloud-gpu.ipynb'
nb = nbformat.read(nb_path, as_version=4)

new_source = """RUN_ENSEMBLE = True

if RUN_ENSEMBLE:
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    print("Instantiating Cycle 3 Hybrid GPU Ensemble...")
    
    # 1. Our Optuna-Tuned CatBoost Model
    model_1 = CatBoostClassifier(
        task_type='GPU',
        learning_rate=0.272492924214504,
        depth=4,
        iterations=100,
        l2_leaf_reg=0.6146439247862419,
        random_state=42,
        verbose=False
    )
    
    # 2. The XGBoost Powerhouse
    # (Using highly robust default parameters for tabular data)
    model_2 = XGBClassifier(
        device='cuda',
        learning_rate=0.272492924214504,
        depth=6,
        n_estimators=100,
        random_state=42,
        tree_method='hist'
    )
    
    # 3. Combine them into the Swarm!
    # 'soft' voting averages the probabilities for maximum accuracy
    ensemble = VotingClassifier(
        estimators=[('catboost', model_1), ('xgb', model_2)],
        voting='soft'
    )
    
    # 4. Wrap the Ensemble in our Preprocessor Pipeline
    ensemble_pipeline = make_pipeline(preprocessor, ensemble)
    
    print("Running 5-Fold Cross-Validation on the Hybrid Classifier...")
    ensemble_scores = cross_val_score(ensemble_pipeline, X_train_eng, y_train.values.ravel(), cv=5, scoring='accuracy')
    
    print(f"\\nFinal Cycle 3 Ensemble Accuracy: {np.mean(ensemble_scores):.4f} (+/- {np.std(ensemble_scores) * 2:.4f})")"""

modified = False
for c in nb.cells:
    if c.cell_type == 'code' and 'model_1 = CatBoostClassifier' in c.source and 'VotingClassifier' in c.source:
        c.source = new_source
        modified = True
        break

if modified:
    nbformat.write(nb, nb_path)
    print("Successfully replaced the ensemble cell in private notebook!")
else:
    print("Could not find the ensemble cell with CatBoost.")
