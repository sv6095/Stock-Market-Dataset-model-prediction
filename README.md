The model is evaluated based on the dataset nifty_500

Model Evaluation Results:

 Regression Models

| Model             | MAE   | R-squared |
|-------------------|-------|-----------|
| Random Forest     | 20.51 | 0.9977    |
| Simple Regression | 68.94 | 0.9922    |

 The Random Forest Regression model significantly outperforms the simpler regression model, showcasing superior predictive accuracy.

 Classification Models

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.9901   | 0.9833    | 1.0000 | 0.9916   |
| SVM            | 0.5842   | 0.5842    | 1.0000 | 0.7375   |
| Linear SVM     | 0.5842   | 0.5842    | 1.0000 | 0.7375   |

 The Random Forest Classifier demonstrates exceptional performance with high accuracy, precision, recall, and F1-Score. It significantly surpasses the SVM models in overall classification capability.


 Potential issues:

1.Data Imbalance-In svm and lsvm,maybe both the models are biased towards the majority dataset(1) 
Tried smote to resolve the issue but it gave perfect scores for random forest model(which is not very possible in practical except a few cases)


