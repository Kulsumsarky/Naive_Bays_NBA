# Naive Bayes NBA Player Longevity Prediction

## Project Overview
This project builds a Gaussian Naive Bayes classifier to predict
whether an NBA player will last 5+ years in the league based on
their performance statistics.

## Dataset
- Size: 1,340 players, 11 features
- Target: target_5yrs (1=lasted 5+ years, 0=did not)
- Missing values: None
- Source: Engineered NBA performance dataset

## Modeling Approach
- Used Gaussian Naive Bayes for continuous features
- Split data 80/20 train/test
- Applied StandardScaler before training

## Model Performance
- Accuracy:  66%
- Precision: 86.1%
- Recall:    55.0%
- F1-Score:  67.1%

## Confusion Matrix
|                | Predicted Did Not Last | Predicted Lasted 5yrs |
|----------------|------------------------|----------------------|
| Actual Did Not Last | 84 (TN)          | 15 (FP)              |
| Actual Lasted 5yrs  | 76 (FN)          | 93 (TP)              |

## Independence Assumption Analysis
Naive Bayes assumes all features are independent.
However basketball stats ARE correlated:
- More minutes played = more points scored
- More shots taken = more points scored
This violates the independence assumption which explains
the model's moderate accuracy of 66%.

## Scouting Recommendations
1. Use model to shortlist players not make final decisions
2. High precision means recommended players are reliable
3. Low recall means scouts must still watch game film
4. Combine model with qualitative scouting insights
5. Consider Random Forest for better accuracy

## Environment Setup
pip install pandas numpy matplotlib seaborn scikit-learn

## Files
- naive_bayes_nba.ipynb - Main analysis notebook
- nba_engineered_data.csv - Dataset
