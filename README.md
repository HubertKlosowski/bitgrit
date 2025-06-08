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
- `LogisticRegression`: C=0.03, solver='lbfgs', class_weight='balanced'
- `LGBMClassifier`: ...
- `XGBClassifier`: ...
- `CatBoostClassifier`: ...

Użyto `stack_method="predict_proba"` i `cv=6`.

---

## 🔁 Jak odtworzyć rozwiązanie?

1. Zainstaluj wymagane biblioteki:
   ```
   pip install -r requirements.txt
   ```

2. Uruchom pipeline:
   ```bash
   python main.py
   ```

   Ten skrypt:
   - Wczyta dane
   - Przetworzy cechy
   - Wytrenuje model
   - Wygeneruje plik predykcji `submission.csv`

---

## 🧪 Środowisko

- OS: Windows 11 / Ubuntu 22.04
- Python: 3.12
- RAM: 16 GB
- CPU: Intel i7 / AMD Ryzen 7
- GPU: (opcjonalnie) NVIDIA RTX 3060
- Dysk: min. 5 GB wolnego miejsca

---

## ❓ Pytania

- Jakie pliki są używane?
  - `train.csv`, `test.csv`
- Jak są przetwarzane?
  - Przez funkcję `preprocess_data()` w `main.py`
- Algorytm i jego hiperparametry?
  - Opisano powyżej
- Inne komentarze:
  - Ziarno losowe ustawione globalnie (`np.random.seed(42)`, `random_state=42`) dla powtarzalności wyników