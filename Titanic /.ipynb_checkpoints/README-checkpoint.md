# Titanic Survival Prediction

```
> "The missing data wasn't a problem to clean — 
> it was a story to read."
```

## Problem
Binary classification: predict passenger survival
Business framing: demonstrates end-to-end ML pipeline
for imbalanced classification problems

## Data Challenges
- there was almost 77% of data was missing in cabin column
- there was a huge imbalance in the age column - middle aged(30-40)
- there was data missing in embark , age and majorly in cabin

## Approach
- Initial features → 82.01%
- Discovery: from the "Cabin" column there i discovered that persons with cabin has higher rate of surviving compared to those without it. 
- Feature engineered: I made a decision to not drop the cabin column , i made a new column named has_cabin in which i put 1 -> has cabin , 0 -> doesnt has cabin
- Algorithms compared: LR vs RF vs XGBoost
- Finding: All three algorithms hit the same ~82% ceiling.The bottleneck was data quality, not algorithm choice.Adding more complex models did not improve accuracy  better features would have more impact than better algorithms.

## Key Decisions
- Median over mean: Chose median over mean because Age distribution is right-skewed. outliers on the older end pull mean to 29.7 vs median 28.0.
  
- Dropped Cabin: Made has_cabin column in place of it , no need in keeping it 
- Has_Cabin feature:Preserved survival signal hidden in  missing data. Passengers   with cabin → 68% survival.  Without cabin → 29% survival. 2.3x difference.
- Stratified split: Without stratify → 82.12% (inflated, biased test set). With stratify → 79.80% (honest, balanced class ratio). Always use stratify=y for 
imbalanced classification.

## Results
| Algorithm | Accuracy |
|-----------|----------|
| Logistic Regression |  82.12% |
| Random Forest | 81.56% |
| XGBoost | 81.56% |

## What I'd improve
- Add cross-validation for more robust accuracy estimate
- Engineer Name title feature properly (Mr/Mrs/Miss 
  showed strong survival signal)
- Deploy as FastAPI endpoint with input validation
  and health check endpoint

## Model saved
titanic_model.pkl + titanic_scaler.pkl
Ready for API deployment

## Tech Stack
Python • Pandas • NumPy • Scikit-learn • Jupyter
