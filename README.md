# Stock Price Prediction Using Machine Learning

## Overview
This project demonstrates the application of various machine learning models to predict stock price movements using historical stock data. The dataset used is the Google stock price dataset, which includes features such as `Open`, `Close`, `High`, `Low`, and `Volume`. The goal is to predict whether the stock price will increase (`1`) or decrease (`0`) based on the percentage change in the closing price.

The following machine learning models are implemented and evaluated:
1. **K-Nearest Neighbors (KNN)**
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**
4. **Decision Tree**
5. **Naive Bayes**

The project includes:
- Data preprocessing and feature engineering.
- Model training and evaluation.
- Performance comparison using accuracy, precision, recall, F1-score, and ROC-AUC curves.
- Visualization of results, including confusion matrices and classification reports.

---

## Key Features
1. **Data Preprocessing:**
   - The dataset is loaded and preprocessed to calculate percentage changes in the closing price.
   - The target variable (`target`) is created based on whether the price increased (`1`) or decreased (`0`).

2. **Feature Engineering:**
   - Features such as `Close` and `pct_change` (percentage change in closing price) are used for training the models.
   - Manual feature scaling (standardization) is applied to normalize the data.

3. **Model Implementation:**
   - **KNN:** Euclidean distance is used to find the nearest neighbors, and predictions are made based on majority voting.
   - **Logistic Regression:** L2 regularization is applied to prevent overfitting.
   - **SVM:** A linear kernel is used, and the model is trained using hinge loss.
   - **Decision Tree:** A custom implementation of a decision tree with Gini index for splitting.
   - **Naive Bayes:** Gaussian Naive Bayes is implemented to calculate probabilities and make predictions.

4. **Model Evaluation:**
   - Accuracy, precision, recall, and F1-score are calculated for each model.
   - Confusion matrices are plotted to visualize true positives, false positives, true negatives, and false negatives.
   - ROC-AUC curves are generated to compare the performance of the models.

5. **Visualization:**
   - Bar plots are used to compare the accuracy of different models.
   - ROC-AUC curves are plotted to evaluate the trade-off between true positive rate (TPR) and false positive rate (FPR).

---

## Results
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| KNN                  | 92.59%   | 88.93%    | 98.96% | 93.68%   |
| Logistic Regression  | 93.40%   | 89.37%    | 100%   | 94.38%   |
| SVM                  | 99.79%   | 99.79%    | 100%   | 99.89%   |
| Decision Tree        | 55.44%   | 55.44%    | 100%   | 71.33%   |
| Naive Bayes          | 78.82%   | 72.36%    | 100%   | 83.96%   |

### Key Insights:
- **SVM** outperformed all other models with near-perfect accuracy (99.79%) and F1-score (99.89%).
- **Logistic Regression** and **KNN** also performed well, with accuracy scores above 92%.
- **Decision Tree** and **Naive Bayes** had lower accuracy, indicating they may not be suitable for this dataset without further tuning.

---

## How to Run the Code
1. **Prerequisites:**
   - Python 3.x
   - Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

2. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn
   ```

3. **Run the Code:**
   - Download the dataset (`google.csv`) and update the file path in the code.
   - Execute the Python script to train and evaluate the models.

---

## Code Structure
- **Data Loading and Preprocessing:**
  - Load the dataset and reset the index.
  - Convert the `Date` column to datetime format.
  - Calculate percentage change in closing price and create the target variable.

- **Feature Engineering:**
  - Split the data into training and testing sets.
  - Apply manual feature scaling (standardization).

- **Model Implementation:**
  - Custom implementations of KNN, Logistic Regression, SVM, Decision Tree, and Naive Bayes.
  - Functions for training, prediction, and evaluation.

- **Evaluation and Visualization:**
  - Calculate accuracy, precision, recall, and F1-score.
  - Plot confusion matrices and ROC-AUC curves.

---

## Future Improvements
1. **Hyperparameter Tuning:**
   - Use grid search or random search to optimize hyperparameters for each model.
2. **Feature Engineering:**
   - Add more features such as moving averages, technical indicators, or sentiment analysis.
3. **Ensemble Methods:**
   - Implement ensemble models like Random Forest or Gradient Boosting to improve performance.
4. **Handling Class Imbalance:**
   - Address class imbalance using techniques like SMOTE or class weights.
5. **Cross-Validation:**
   - Use cross-validation to ensure the models generalize well to unseen data.

---

## Conclusion
This project demonstrates the application of machine learning models to predict stock price movements. SVM and Logistic Regression are the top-performing models, while Decision Tree and Naive Bayes require further tuning. By incorporating additional features and techniques, the performance of the models can be further improved.



**Note:** This project is for educational purposes only and should not be used for actual stock trading decisions.
