# Konkurs Bitgrit 2025 – README

## 📁 Używane pliki danych

- `train.csv` – zbiór treningowy dostarczony przez organizatora
- `test.csv` – zbiór testowy dostarczony przez organizatora

---

## 🛠️ Przetwarzanie danych

1. Wczytanie danych za pomocą pandas.
2. Zakodowanie kolumn kategorycznych (jeśli użyty model nie obsługuje kategorii natywnie).
3. Inżynieria cech:
   - Konwersja kolumn typu "date" do wartości liczbowych
   - Grupowanie niskoczęstotliwościowych kategorii
   - Standaryzacja wartości numerycznych (jeśli wymagane)
4. Podział na `X_train`, `y_train`, `X_test`

---

## 🧠 Użyty algorytm

Model: `StackingClassifier`

Skład:
- `CatBoostClassifier`
- `LGBMClassifier`
- `XGBClassifier`

Model końcowy (meta-model): `LogisticRegression`

Hiperparametry:
- `LogisticRegression`: {
  "penalty": 'l2',
  "C": 3e-2,
  "class_weight": 'balanced',
  "tol": 1e-4,
  "warm_start": False,
  "solver": 'lbfgs',
  "max_iter": 80,
  "random_state": 42,
  "multi_class": 'multinomial'
}
- `LGBMClassifier`: {
  "boosting_type": gbdt,
  "class_weight": None,
  "colsample_bytree": 0.9333584928887282,
  "importance_type": split,
  "learning_rate": 0.23332896051177257,
  "max_depth": 12,
  "min_child_samples": 20,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 142,
  "n_jobs": -1,
  "num_leaves": 34,
  "objective": multiclass,
  "random_state": 81,
  "reg_alpha": 0.0,
  "reg_lambda": 1,
  "subsample": 0.9720591168744056,
  "subsample_for_bin": 200000,
  "subsample_freq": 0,
  "num_class": 3,
  "boosting": gbdt,
  "metric": None,
  "verbose": -1,
  "min_data_in_leaf": 6
}
- `XGBClassifier`: {
  "objective": multi:softprob,
  "base_score": None,
  "booster": None,
  "callbacks": None,
  "colsample_bylevel": None,
  "colsample_bynode": None,
  "colsample_bytree": None,
  "device": None,
  "early_stopping_rounds": None,
  "enable_categorical": True,
  "eval_metric": None,
  "feature_types": None,
  "feature_weights": None,
  "gamma": 0.6551776163530296,
  "grow_policy": None,
  "importance_type": None,
  "interaction_constraints": None,
  "learning_rate": 0.07315523763489679,
  "max_bin": 219,
  "max_cat_threshold": None,
  "max_cat_to_onehot": None,
  "max_delta_step": None,
  "max_depth": 10,
  "max_leaves": 23,
  "min_child_weight": None,
  "missing": nan,
  "monotone_constraints": None,
  "multi_strategy": None,
  "n_estimators": 118,
  "n_jobs": -1,
  "num_parallel_tree": None,
  "random_state": 42,
  "reg_alpha": 0.013705730913738874,
  "reg_lambda": 0.012219537217204616,
  "sampling_method": None,
  "scale_pos_weight": None,
  "subsample": 0.9988236950762315,
  "tree_method": hist,
  "validate_parameters": None,
  "verbosity": None,
  "use_label_encoder": None,
  "gpu_id": None,
  "predictor": None,
  "num_class": 3
}
- `CatBoostClassifier`: {
  "iterations": 137,
  "learning_rate": 0.10095947308778701,
  "depth": 7,
  "l2_leaf_reg": 0.01124311885824868,
  "loss_function": MultiClass,
  "random_seed": 42,
  "logging_level": Silent,
  "bagging_temperature": 1.0,
  "early_stopping_rounds": 3
}

Użyto `stack_method="predict_proba"` i `cv=6`.

---

## 🔁 Jak odtworzyć rozwiązanie?

1. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```

2. Uruchom pipeline w odpowiedniej kolejności:

   a) Notebook do przetwarzania danych:
   ```bash
   jupyter notebook know_data.ipynb
   ```

   b) Notebook do trenowania modeli i generowania predykcji:
   ```bash
   jupyter notebook models.ipynb
   ```

   Notatniki:
   - Wczytują dane
   - Przetwarzają cechy i dane tekstowe
   - Trenują modele
   - Generują plik predykcji `submission.csv`

---

## 🧪 Środowisko

- OS: Windows 11
- Python: 3.12
- RAM: 16 GB
- CPU: Intel i7-9700k
- GPU: (opcjonalnie) NVIDIA RTX 2080

---

## ❓ Pytania

- Jakie pliki są używane?
  - `train.csv`, `test.csv`
- Jak są przetwarzane?
  - Przez notebook `know_data.ipynb`
- Algorytm i jego hiperparametry?
  - Opisano powyżej
- Inne komentarze:
  - Ziarno losowe ustawione globalnie (`np.random.seed(42)`, `random_state=42`) dla powtarzalności wyników