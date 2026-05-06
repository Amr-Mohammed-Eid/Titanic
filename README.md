# Titanic Survival Prediction (XGBoost)

This project trains a Titanic survival model with a consistent train/test preprocessing flow and outputs `submission.csv` in Kaggle format.

## Workflow implemented in `titanic.ipynb`

1. Merge train and test into one dataframe for feature engineering.
2. Apply feature engineering in this order:
   - Title extraction from `Name`
   - Age imputation by title median
   - `Embarked` mode imputation
   - `Fare` imputation by `Pclass` median
   - `FamilySize`, `IsAlone`, `FarePerPerson`
   - Deck extraction from `Cabin`
   - `AgeBand` creation
   - Drop `Name`, `Ticket`, `Cabin`, `PassengerId`
3. One-hot encode categoricals on the combined dataframe.
4. Split back to train/test.
5. Evaluate baseline XGBoost with `StratifiedKFold` CV.
6. Tune with `GridSearchCV`.
7. Refit best model on full train and predict test.

## Setup

```bash
pip install -r requirements.txt
```

## Run

- Open `titanic.ipynb`
- Restart kernel
- Run all cells top-to-bottom

This will regenerate `submission.csv` with columns:

- `PassengerId`
- `Survived`

## Files

- `train.csv` / `test.csv`: Kaggle Titanic data
- `titanic.ipynb`: full training + tuning pipeline
- `submission.csv`: latest predictions
- `requirements.txt`: Python dependencies
