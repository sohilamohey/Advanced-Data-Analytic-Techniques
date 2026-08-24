# modeling.py
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# ===================================================
# TASK 4.1: BASELINE MODEL
# ===================================================
# ===================================================
# TASK 4.1.1: BASELINE MODEL (Logistic Regression)
# ===================================================
print("\n" + "="*80)
print("TASK 4.1: BASELINE MODEL (Logistic Regression)")
print("="*80 + "\n")

def train_baseline(X_train, y_train, X_val, y_val, X_test, y_test, name):

    print(f"\n[Baseline] Training Logistic Regression on {name} ...")

    # 1) Handle sparse inputs safely (convert to dense if needed)
    if sp.issparse(X_train):
        X_train = X_train.toarray()
    if sp.issparse(X_val):
        X_val = X_val.toarray()
    if sp.issparse(X_test):
        X_test = X_test.toarray()

    model = LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        solver='liblinear',
        C=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    proba_val = model.predict_proba(X_val)
    is_binary = (proba_val.shape[1] == 2)

    if is_binary:
        val_probs = proba_val[:, 1]
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            preds_val = (val_probs >= thr).astype(int)
            f1_macro = f1_score(y_val, preds_val, average='macro')
            if f1_macro > best_f1:
                best_f1, best_thr = f1_macro, thr
    else:
        best_thr = 0.5

    proba_test = model.predict_proba(X_test)
    if is_binary:
        y_probs = proba_test[:, 1]
        y_pred = (y_probs >= best_thr).astype(int)
        roc_val = roc_auc_score(y_test, y_probs)
    else:
        y_pred = np.argmax(proba_test, axis=1)
        try:
            roc_val = roc_auc_score(y_test, proba_test, multi_class='ovr', average='weighted')
        except Exception:
            roc_val = np.nan

    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print(f"[Baseline:{name}] Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | "
          f"Recall: {rec*100:.2f}% | F1 (weighted): {f1_w*100:.2f}% | "
          f"ROC-AUC: {0 if np.isnan(roc_val) else roc_val*100:.2f}%")

    save_confusion_matrix_plot(y_test, y_pred, " Logistic Regression ", name)
    return {
        'Model': 'Logistic Regression (Baseline)',
        'Embedding': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1_w,
        'ROC-AUC': roc_val
    }

# ===================================================
# Task 4.1.2: BASELINE MODEL (Naive Bayes)
# ===================================================
def train_nb(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"Training Naive Bayes ({name})...")

    # Detect problem type
    is_multiclass = len(np.unique(y_train)) > 2

    # Helper: check non-negativity (for MNB/CNB)
    def is_non_negative(X):
        if sp.issparse(X):
            return X.min() >= 0
        return np.min(X) >= 0

    best_model = None
    best_score = -1.0
    best_thr = 0.5  # used for binary

    # Case A: prefer MNB/CNB when features are non-negative (TF-IDF or similar)
    if (sp.issparse(X_train) and is_non_negative(X_train)) or (not sp.issparse(X_train) and is_non_negative(X_train)):
        candidates = []
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            candidates.append(MultinomialNB(alpha=alpha))
            candidates.append(ComplementNB(alpha=alpha))
        for model in candidates:
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)
            if not is_multiclass:
                probs = proba_val[:, 1]
                best_thr_p, best_combo_p = 0.5, -1.0
                for thr in np.linspace(0.1, 0.9, 17):
                    preds = (probs >= thr).astype(int)
                    combo = (accuracy_score(y_val, preds) + f1_score(y_val, preds, average='weighted')) / 2
                    if combo > best_combo_p:
                        best_combo_p, best_thr_p = combo, thr
                score = best_combo_p
                thr_for_model = best_thr_p
            else:
                preds = np.argmax(proba_val, axis=1)
                score = f1_score(y_val, preds, average='weighted')
                thr_for_model = None
            if score > best_score:
                best_score = score
                best_model = model
                if not is_multiclass:
                    best_thr = float(thr_for_model)
    else:
        # Case B: dense with negatives (e.g., BERT) -> GaussianNB (requires dense)
        if sp.issparse(X_train):
            X_train = X_train.toarray()
            X_val = X_val.toarray()
            X_test = X_test.toarray()
        for vs in [1e-9, 1e-8, 1e-7]:
            model = GaussianNB(var_smoothing=vs)
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)
            if not is_multiclass:
                probs = proba_val[:, 1]
                best_thr_p, best_combo_p = 0.5, -1.0
                for thr in np.linspace(0.1, 0.9, 17):
                    preds = (probs >= thr).astype(int)
                    combo = (accuracy_score(y_val, preds) + f1_score(y_val, preds, average='weighted')) / 2
                    if combo > best_combo_p:
                        best_combo_p, best_thr_p = combo, thr
                score = best_combo_p
                thr_for_model = best_thr_p
            else:
                preds = np.argmax(proba_val, axis=1)
                score = f1_score(y_val, preds, average='weighted')
                thr_for_model = None
            if score > best_score:
                best_score = score
                best_model = model
                if not is_multiclass:
                    best_thr = float(thr_for_model)

    # Predict on test
    proba_test = best_model.predict_proba(X_test)
    if not is_multiclass:
        y_proba = proba_test[:, 1]
        y_pred = (y_proba >= best_thr).astype(int)
        auc_score = roc_auc_score(y_test, y_proba)
    else:
        y_pred = np.argmax(proba_test, axis=1)
        try:
            auc_score = roc_auc_score(y_test, proba_test, multi_class='ovr', average='weighted')
        except Exception:
            auc_score = np.nan

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {0 if np.isnan(auc_score) else auc_score*100:.2f}%")

    # Hooks (defined externally)
    save_confusion_matrix_plot(y_test, y_pred, "NaiveBayes", name)
    save_model(best_model, f"nb_{name}")

    return {
        'Model': 'Naive Bayes',
        'Embedding': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc_score
    }


# ===================================================
# Task 4.2: TRADITIONAL ML MODELS (SVM → RF → XGBoost)
# ===================================================
# ===================================================
# Task 4.2.1: TRADITIONAL ML MODELS (SVM)
# ===================================================
def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"Training SVM ({name})")

    # Step 1: Train linear SVM on train data
    base_svm = LinearSVC(C=1.0, class_weight='balanced', random_state=42, dual='auto')
    base_svm.fit(X_train, y_train)

    # Step 2: Calibrate on validation to enable predict_proba
    model = CalibratedClassifierCV(base_svm, method='sigmoid', cv='prefit')
    model.fit(X_val, y_val)

    # Step 3: Evaluate on test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    try:
        if len(np.unique(y_test)) > 2:
            auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
    except Exception:
        auc_score = np.nan

    print(f"Params: {{'C': 1.0, 'kernel': 'linear (approx)', 'calibration': 'sigmoid'}}")
    print(f"Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {0 if np.isnan(auc_score) else auc_score*100:.2f}%")

    # Keep your exact saving workflow
    save_confusion_matrix_plot(y_test, y_pred, "SVM", name)
    save_model(model, f"svm_{name}")

    return {'Model': 'SVM', 'Embedding': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc_score}

# ===================================================
# Task 4.2.2: TRADITIONAL ML MODELS (RF)
# ===================================================
def train_rf(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"Training Random Forest ({name})...")

    # Sparse → dense if needed
    if sp.issparse(X_train):
        X_train, X_val, X_test = X_train.toarray(), X_val.toarray(), X_test.toarray()

    # Candidate params (strong but still fast)
    candidates = [
        {"n_estimators": 400, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 25,   "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 15,   "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": 0.5},
    ]

    is_multiclass = (len(np.unique(y_train)) > 2)
    best_params = None
    best_score = -1.0
    best_thr = 0.5  # for binary threshold tuning

    # Lightweight validation selection
    for p in candidates:
        rf = RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            class_weight='balanced_subsample',
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            oob_score=False
        )
        rf.fit(X_train, y_train)

        if is_multiclass:
            y_val_pred = rf.predict(X_val)
            score = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            thr_for_p = None
        else:
            val_proba = rf.predict_proba(X_val)[:, 1]
            best_thr_p, best_combo_p = 0.5, -1.0
            for thr in np.linspace(0.05, 0.95, 19):
                preds = (val_proba >= thr).astype(int)
                combo = (accuracy_score(y_val, preds) + f1_score(y_val, preds, average='weighted')) / 2
                if combo > best_combo_p:
                    best_combo_p, best_thr_p = combo, thr
            score = best_combo_p
            thr_for_p = best_thr_p

        if score > best_score:
            best_score = score
            best_params = p.copy()
            if not is_multiclass:
                best_thr = float(thr_for_p)

    # Final model on TRAIN with best params
    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        class_weight='balanced_subsample',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on TEST (with tuned threshold for binary)
    y_proba_full = model.predict_proba(X_test)
    if is_multiclass:
        y_pred = np.argmax(y_proba_full, axis=1)
        try:
            auc_score = roc_auc_score(y_test, y_proba_full, multi_class='ovr', average='weighted')
        except Exception:
            auc_score = np.nan
    else:
        y_proba = y_proba_full[:, 1]
        y_pred  = (y_proba >= best_thr).astype(int)
        auc_score = roc_auc_score(y_test, y_proba)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Params: {best_params}")
    print(f"Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {0 if np.isnan(auc_score) else auc_score*100:.2f}%")

    save_confusion_matrix_plot(y_test, y_pred, "RandomForest", name)
    save_model(model, f"rf_{name}")

    return {'Model': 'Random Forest', 'Embedding': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc_score}

# ===================================================
# Task 4.2.3: TRADITIONAL ML MODELS (XGBoost)
# ===================================================
def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"Training XGBoost ({name})...")

    # Dense conversion if needed
    if sp.issparse(X_train):
        X_train = X_train.toarray()
        X_val   = X_val.toarray()
        X_test  = X_test.toarray()

    # GPU if available (optional)
    tree_method = "hist"
    predictor   = "auto"
    try:
        if torch.cuda.is_available():
            tree_method = "gpu_hist"
            predictor   = "gpu_predictor"
    except Exception:
        pass

    # Binary vs multiclass
    is_multiclass = (len(np.unique(y_train)) > 2)

    # scale_pos_weight (binary)
    if not is_multiclass:
        pos = int((y_train == 1).sum())
        neg = int((y_train != 1).sum())
        scale_pos_weight = float(max(neg / max(pos, 1), 1.0))
    else:
        scale_pos_weight = 1.0

    # Fast stratified subset for tuning
    from sklearn.model_selection import StratifiedShuffleSplit
    def stratified_subset(X, y, max_samples=3000, random_state=42):
        if len(y) <= max_samples:
            return X, y
        sss = StratifiedShuffleSplit(n_splits=1, test_size=max_samples/len(y), random_state=random_state)
        _, idx = next(sss.split(X, y))
        return X[idx], y[idx]

    X_sub, y_sub = stratified_subset(X_train, y_train, max_samples=3000)

    # Compact, strong candidates (including n_estimators)
    candidates = [
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8},
    ]

    best_params = None
    best_score  = -1.0
    best_thr    = 0.5

    # Mini-tuning on subset (no early stopping)
    for p in candidates:
        model = XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"],
            reg_lambda=1.0,
            objective=("binary:logistic" if not is_multiclass else "multi:softprob"),
            num_class=(None if not is_multiclass else len(np.unique(y_train))),
            random_state=42,
            tree_method=tree_method,
            predictor=predictor,
            max_bin=256,
            n_jobs=-1,
            verbosity=0,
            eval_metric=("auc" if not is_multiclass else "mlogloss"),
            scale_pos_weight=(scale_pos_weight if not is_multiclass else 1.0)
        )

        model.fit(X_sub, y_sub)

        proba_val = model.predict_proba(X_val)
        if not is_multiclass:
            probs = proba_val[:, 1]
            best_thr_p, best_combo_p = 0.5, -1.0
            for thr in np.linspace(0.1, 0.9, 17):
                preds = (probs >= thr).astype(int)
                combo = (accuracy_score(y_val, preds) + f1_score(y_val, preds, average="weighted")) / 2
                if combo > best_combo_p:
                    best_combo_p, best_thr_p = combo, thr
            score = best_combo_p
        else:
            preds = np.argmax(proba_val, axis=1)
            score = f1_score(y_val, preds, average="weighted")

        if score > best_score:
            best_score = score
            best_params = p.copy()
            if not is_multiclass:
                best_thr = float(best_thr_p)

    # Final model on FULL train (single fit)
    final = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_lambda=1.0,
        objective=("binary:logistic" if not is_multiclass else "multi:softprob"),
        num_class=(None if not is_multiclass else len(np.unique(y_train))),
        random_state=42,
        tree_method=tree_method,
        predictor=predictor,
        max_bin=256,
        n_jobs=-1,
        verbosity=0,
        eval_metric=("auc" if not is_multiclass else "mlogloss"),
        scale_pos_weight=(scale_pos_weight if not is_multiclass else 1.0)
    )
    final.fit(X_train, y_train)

    # Test evaluation
    proba_test = final.predict_proba(X_test)
    if not is_multiclass:
        y_proba = proba_test[:, 1]
        y_pred  = (y_proba >= best_thr).astype(int)
        auc_score = roc_auc_score(y_test, y_proba)
    else:
        y_pred  = np.argmax(proba_test, axis=1)
        try:
            auc_score = roc_auc_score(y_test, proba_test, multi_class="ovr", average="weighted")
        except Exception:
            auc_score = np.nan

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Best Parameters: {best_params}")
    if len(np.unique(y_train)) <= 2:
        print(f"Best Threshold: {best_thr:.3f}")
    print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {0 if np.isnan(auc_score) else auc_score*100:.2f}%")

    save_confusion_matrix_plot(y_test, y_pred, "XGBoost", name)
    save_model(final, f"xgb_{name.lower()}")

    return {
        'Model': 'XGBoost',
        'Embedding': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc_score
    }


# ===================================================
# Task 4.3: DEEP LEARNING MODEL (FFNN → Simple DNN)
# ===================================================
# ===================================================
# Task 4.3.1: DEEP LEARNING MODEL (FFNN)
# ===================================================
def build_ffnn(input_dim, hidden1=512, hidden2=256):
    # Wider + BatchNorm for stability; same API & 2 outputs for CE loss
    return nn.Sequential(
        nn.Linear(input_dim, hidden1),
        nn.BatchNorm1d(hidden1),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(hidden1, hidden2),
        nn.BatchNorm1d(hidden2),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(hidden2, 2)
    ).to(device)


def train_ffnn(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"Training FFNN ({name})...")

    # --- Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=256
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=256
    )

    model = build_ffnn(X_train.shape[1])

    # --- class weights (robust) ---
    n_classes = int(np.max(y_train)) + 1
    counts = np.bincount(y_train, minlength=n_classes)
    weights = counts.max() / np.clip(counts, 1, None)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0
    best_model_state = None
    best_thr = 0.5
    patience = 7
    patience_counter = 0
    max_epochs = 30
    clip_norm = 1.0

    for _ in range(max_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        # ---- validation + threshold tuning (binary) ----
        model.eval()
        val_probs = []
        val_y_list = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                val_probs.extend(probs)
                val_y_list.extend(batch_y.numpy())

        val_probs = np.array(val_probs)
        val_y_np  = np.array(val_y_list)

        # scan thresholds to maximize weighted F1
        best_thr_epoch, best_f1_epoch = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (val_probs >= thr).astype(int)
            f1_val = f1_score(val_y_np, preds, average='weighted')
            if f1_val > best_f1_epoch:
                best_f1_epoch, best_thr_epoch = f1_val, thr

        scheduler.step(best_f1_epoch)

        if best_f1_epoch > best_val_f1:
            best_val_f1 = best_f1_epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = float(best_thr_epoch)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # ---- restore best state ----
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    # ---- test inference ----
    test_probs = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            logits = model(batch_X.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            test_probs.extend(probs)
    test_probs = np.array(test_probs)

    # use tuned threshold
    y_pred = (test_probs >= best_thr).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    auc_score = roc_auc_score(y_test, test_probs)

    print(f"Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {0 if np.isnan(auc_score) else auc_score*100:.2f}%")
    print(f"Best Val F1: {best_val_f1*100:.2f}% | Tuned Threshold: {best_thr:.3f}")

    save_confusion_matrix_plot(y_test, y_pred, "FFNN", name)
    torch.save(model.state_dict(), f"trained_models/ffnn_{name}.pt")
    print("Saved\n")

    return {'Model': 'FFNN (Deep Learning)', 'Embedding': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc_score}
# ===================================================
# Task 4.3.2: DEEP LEARNING MODEL (Simple DNN)
# ===================================================
def build_simple_dnn(input_dim, hidden=256, dropout=0.2):
    """
    One hidden layer MLP for speed.
    Output = 2 logits for CrossEntropyLoss (binary as 0/1).
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 2)
    ).to(device)


def train_dnn_simple(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"Training Simple DNN ({name})...")

    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=256
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=256
    )

    model = build_simple_dnn(X_train.shape[1], hidden=256, dropout=0.2)

    # Class weights (robust for imbalance)
    n_classes = int(np.max(y_train)) + 1
    counts = np.bincount(y_train, minlength=n_classes)
    weights = counts.max() / np.clip(counts, 1, None)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # slightly higher lr for shallow net
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0
    best_model_state = None
    best_thr = 0.5
    patience = 5
    patience_counter = 0
    max_epochs = 18
    clip_norm = 1.0

    for _ in range(max_epochs):
        # Train
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        # Validate and threshold tuning (binary)
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                val_probs.extend(probs)
                val_true.extend(batch_y.numpy())

        val_probs = np.array(val_probs)
        val_true = np.array(val_true)

        best_thr_epoch, best_f1_epoch = 0.5, -1.0
        for thr in np.linspace(0.1, 0.9, 17):
            preds = (val_probs >= thr).astype(int)
            f1_val = f1_score(val_true, preds, average='weighted', zero_division=0)
            if f1_val > best_f1_epoch:
                best_f1_epoch, best_thr_epoch = f1_val, thr

        scheduler.step(best_f1_epoch)

        if best_f1_epoch > best_val_f1:
            best_val_f1 = best_f1_epoch
            best_thr = float(best_thr_epoch)
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()

    # Test
    test_probs = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            logits = model(batch_X.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            test_probs.extend(probs)
    test_probs = np.array(test_probs)
    y_pred = (test_probs >= best_thr).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    auc_score = roc_auc_score(y_test, test_probs)

    print(f"Acc: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {0 if np.isnan(auc_score) else auc_score*100:.2f}%")
    print(f"Best Val F1: {best_val_f1*100:.2f}% | Tuned Threshold: {best_thr:.3f}")

    # Save (same hooks)
    save_confusion_matrix_plot(y_test, y_pred, "SimpleDNN", name)
    torch.save(model.state_dict(), f"trained_models/simple_dnn_{name}.pt")

    return {
        'Model': 'Simple DNN',
        'Embedding': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc_score
    }


# For cleaner imports
__all__ = [       
    "train_baseline",
    "train_nb", 
    "train_svm",
    "train_rf",
    "train_xgb",
    "build_ffnn",
    "train_ffnn",
    "build_simple_dnn",
    "train_dnn_simple",
]