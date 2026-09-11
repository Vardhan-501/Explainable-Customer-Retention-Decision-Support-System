# Customer Churn Prediction: Technical Viva Questions and Answers

This guide is designed for the final-year project viva. The answers are based on the implementation in this repository, including the notebook, Scikit-learn pipeline, Streamlit application, supporting modules, tests, and deployment structure.

## 1. Project understanding and motivation

### Q1. What is the title of your project?

**Answer:** My project is titled **Explainable Customer Churn Prediction and Retention Decision-Support System for Telecom Customers**. It predicts the probability that a telecom customer may churn and presents the result through an interactive Streamlit application.

### Q2. What problem does the project solve?

**Answer:** Customer churn can reduce recurring revenue and increase the cost of acquiring new customers. My project helps identify customers who may be at elevated risk of leaving so that a retention team can prioritize further investigation. It is a decision-support system, not an automated system for deciding how customers should be treated.

### Q3. Why did you choose customer churn prediction?

**Answer:** Customer churn is a realistic business problem where machine learning can support a measurable decision. The dataset contains demographic, account, service, contract, billing, and payment information, which makes it suitable for demonstrating data cleaning, exploratory analysis, classification, imbalanced-learning techniques, explainability, and deployment.

### Q4. What are the inputs and output of the system?

**Answer:** The input consists of four numeric variables—`SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`—along with categorical variables such as contract type, internet service, payment method, phone service, and optional services. The output includes the predicted churn class, churn probability, and a provisional Low, Medium, or High risk band.

### Q5. What is the target variable?

**Answer:** The target variable is `Churn`. In the source dataset it contains `Yes` and `No`. In the modeling workflow, these values are mapped to 1 and 0 respectively, where 1 represents churn.

## 2. Dataset and preprocessing

### Q6. Which dataset did you use?

**Answer:** I used the IBM Telco Customer Churn dataset. The repository contains 7,043 customer records. It includes demographic information, account details, service subscriptions, contract information, billing information, payment method, and the churn outcome.

### Q7. What data-cleaning steps did you perform?

**Answer:** I removed duplicate rows, converted `TotalCharges` from text to numeric values, converted invalid numeric entries to missing values and replaced them with zero in the loading workflow, removed the identifier column `customerID` from the model inputs, separated the target variable, and identified numeric and categorical columns.

### Q8. Why was `customerID` removed?

**Answer:** `customerID` is an identifier rather than a meaningful behavioral or service feature. Including it could cause the model to learn accidental patterns associated with identifiers and would not provide a useful basis for retention decisions.

### Q9. How did you handle categorical variables?

**Answer:** I used `OneHotEncoder(handle_unknown="ignore")` inside a Scikit-learn `ColumnTransformer`. This converts categorical values into numerical indicator columns while allowing the model to handle categories that were not seen during training.

### Q10. How did you handle numerical variables?

**Answer:** I identified the numeric columns and applied `StandardScaler` within the preprocessing pipeline. Although Random Forest does not strictly require scaling, using a consistent preprocessing structure makes the pipeline reusable and supports comparison with algorithms such as logistic regression.

### Q11. What is data leakage, and how did you prevent it?

**Answer:** Data leakage occurs when information from outside the training data influences the model during training, leading to overly optimistic evaluation. I reduced this risk by placing preprocessing and classification inside one Scikit-learn `Pipeline`. The transformations are fitted on the training portion and then applied to the test portion, rather than fitting preprocessing on the entire dataset before splitting.

### Q12. Why did you use a stratified train-test split?

**Answer:** The target classes are imbalanced, so a random split could produce different churn proportions in the training and test sets. Stratification preserves the approximate class distribution in both sets. The notebook uses an 80/20 split with `random_state=42`, resulting in 5,634 training rows and 1,409 test rows.

## 3. Algorithm and model selection

### Q13. Which algorithm did you use?

**Answer:** I used a class-weighted `RandomForestClassifier` inside a Scikit-learn pipeline. The configured model uses 300 trees, `max_depth=10`, `random_state=42`, and `class_weight="balanced"`.

### Q14. Why did you choose Random Forest?

**Answer:** Random Forest is suitable for structured tabular data and can capture nonlinear relationships and interactions between variables such as tenure, contract, monthly charges, and services. It is relatively robust to noise, supports class weighting for imbalanced classification, and provides feature-importance information that is useful when explaining the model to stakeholders.

### Q15. Why did you use `class_weight="balanced"`?

**Answer:** The number of customers who stayed is larger than the number who churned. Without adjustment, the model could focus too strongly on the majority class. `class_weight="balanced"` gives more importance to the minority class during training, helping the model identify more potential churners.

### Q16. What does `n_estimators=300` mean?

**Answer:** It means the Random Forest contains 300 decision trees. More trees can make the ensemble’s predictions more stable, although they also increase training and prediction time. I used a fixed value with `random_state=42` to make the experiment reproducible.

### Q17. What does `max_depth=10` do?

**Answer:** `max_depth=10` limits the maximum depth of each decision tree. This controls model complexity and can reduce overfitting compared with allowing trees to grow without a depth limit.

### Q18. Why not use logistic regression?

**Answer:** Logistic regression is an excellent interpretable baseline, but it models the relationship between the inputs and the log-odds of churn in a relatively simple form unless additional interactions or transformations are created. Random Forest can capture nonlinear effects and interactions more naturally. In a stronger experiment, I would compare both models using cross-validation, PR-AUC, recall, F1-score, calibration, and business-threshold performance.

### Q19. Why not use a neural network?

**Answer:** A neural network would be more complex than necessary for this relatively small tabular dataset. It would require more tuning and would generally be harder to explain to business users. I selected an algorithm whose complexity was appropriate for the dataset and project objective.

### Q20. What is the difference between classification and regression?

**Answer:** Classification predicts a discrete category, such as churn or no churn. Regression predicts a continuous numerical value, such as revenue or monthly spending. This project is a binary classification problem, although the model also produces a probability score between 0 and 1.

## 4. Evaluation and interpretation

### Q21. Which evaluation metrics did you use?

**Answer:** I used precision, recall, F1-score, accuracy, confusion-matrix analysis, and ROC-AUC. The reusable evaluation utilities also support PR-AUC, threshold analysis, cross-validation, false-positive and false-negative counts, and contact rate.

### Q22. What was the model performance?

**Answer:** On the stratified holdout set, the notebook reports an ROC-AUC of approximately 0.8391. For the churn class, precision was 0.55, recall was 0.72, and F1-score was 0.63. Overall accuracy was 0.77.

### Q23. Why is accuracy not enough for this project?

**Answer:** Accuracy can be misleading when the classes are imbalanced. A model could achieve a reasonable accuracy by predicting the majority class frequently while missing many churners. Recall measures how many actual churners were identified, precision measures how many predicted churners were actually churners, and F1-score balances the two.

### Q24. What is recall, and why is it important here?

**Answer:** Recall is the proportion of actual churners that the model correctly identifies. It is important because failing to identify a likely churner may mean losing an opportunity for retention. However, maximizing recall alone could generate too many false positives, so it must be balanced against precision and business capacity.

### Q25. What is ROC-AUC?

**Answer:** ROC-AUC measures how well the model ranks positive cases above negative cases across different probability thresholds. A value close to 0.5 indicates performance similar to random ranking, while a higher value indicates better separation. The project’s holdout ROC-AUC is approximately 0.8391.

### Q26. What is PR-AUC, and why might it be useful?

**Answer:** PR-AUC summarizes the precision-recall trade-off across thresholds. It can be particularly informative when the positive class is relatively uncommon, as in churn prediction. The project includes utilities for calculating PR-AUC, although the main notebook report focuses on ROC-AUC and the classification report.

### Q27. What is a confusion matrix?

**Answer:** A confusion matrix shows true negatives, false positives, false negatives, and true positives. In this project, a false negative is a customer who churned but was predicted not to churn, while a false positive is a customer predicted to churn who actually stayed. The business cost of these two errors may be different.

### Q28. What is a probability threshold?

**Answer:** A probability threshold converts a predicted probability into a class label. For example, a threshold of 0.60 can classify probabilities of 0.60 or higher as churn. The project’s risk bands use provisional boundaries of below 0.30 for Low, 0.30 to below 0.60 for Medium, and 0.60 or higher for High.

### Q29. How would you choose the final threshold in a real company?

**Answer:** I would use validation data and business information such as the cost of missing a churner, the cost of contacting a non-churner, the number of retention agents available, and the expected value of an intervention. I would compare precision, recall, F1-score, contact rate, false positives, false negatives, and calibration at several thresholds.

### Q30. Does feature importance prove causation?

**Answer:** No. Feature importance indicates which features contributed to the model’s predictions, but it does not prove that those features caused churn. For example, a feature may be correlated with another unobserved factor. Explanations should therefore be communicated as predictive associations and used with human review.

## 5. Application, architecture, and software engineering

### Q31. Why did you build a Streamlit application?

**Answer:** A notebook is useful for analysis, but a Streamlit application makes the model easier for a non-technical user to interact with. The application accepts customer inputs and presents a probability, risk band, explanation, recommendations, segments, performance information, and monitoring results in one interface.

### Q32. How does the application make a prediction?

**Answer:** The application loads the serialized pipeline from `models/churn_model.pkl`. It extracts the model schema, collects numeric and categorical values through the form, validates the inputs, constructs a one-row DataFrame in the expected feature order, calls `predict_proba` and `predict`, and displays the probability, class, and risk band.

### Q33. Why did you save the entire pipeline instead of only the classifier?

**Answer:** Saving the entire pipeline preserves the preprocessing steps together with the classifier. This ensures that new inputs are transformed in exactly the same way as the training data. It also reduces the risk of applying inconsistent preprocessing during deployment.

### Q34. How does input validation work?

**Answer:** The prediction module checks that all required fields are present, verifies that numeric fields can be converted to numbers, rejects NaN values and invalid ranges, and checks that categorical values are among the categories learned by the model. The tests verify missing fields, negative charges, invalid categories, and an impossible tenure value.

### Q35. What is the purpose of the `src` directory?

**Answer:** The `src` directory contains the reusable application and machine-learning modules. Separating these modules from the Streamlit entry point improves organization, makes the code easier to test, and allows the same prediction and evaluation logic to be reused outside the interface.

### Q36. What is the purpose of the tests?

**Answer:** The tests check prediction validity, input validation, explanation output, recommendation reasons, segmentation behavior, data-quality summaries, drift-related metrics, and labeled performance metrics. They help detect regressions when the project structure or model-serving code changes.

### Q37. What deployment issue did you face?

**Answer:** The application initially imported modules from `src`, but the files were stored at the repository root. It also expected the dataset in `data` and the model in `models`. I fixed the issue by reorganizing the repository into the expected package layout, adding `src/__init__.py`, updating test paths, and pinning Scikit-learn to the model’s training version.

## 6. Responsible machine learning and limitations

### Q38. What are the main limitations of your project?

**Answer:** The dataset is historical and may not represent current products, prices, service quality, or customer populations. The holdout evaluation is not the same as repeated cross-validation or temporal validation. The risk bands are provisional, the probabilities have not been fully calibrated, and the explanations show associations rather than causes.

### Q39. Could this model be used directly in production?

**Answer:** It is a strong prototype, but I would not deploy it directly for high-impact operational decisions without additional validation. I would add temporal validation, repeated cross-validation, probability calibration, threshold selection based on business cost, monitoring, fairness checks, schema validation, model versioning, and a model card.

### Q40. How could the project be improved?

**Answer:** I would compare Random Forest with logistic regression and gradient boosting, report repeated cross-validation and PR-AUC, calibrate probabilities, perform threshold analysis, add batch scoring, monitor data and prediction drift, test the model on newer data, add fairness analysis where appropriate, and track experiments and model versions.

### Q41. Is the model making an ethical decision?

**Answer:** No. The model provides a prediction based on historical patterns. It should support human review and should not be used to deny service, penalize customers, or automatically assign unfair treatment. Any retention action should be evaluated for effectiveness and fairness.

## 7. Strong closing answer

> My project demonstrates the complete machine-learning lifecycle: defining a business problem, preparing real-world data, building a leakage-resistant preprocessing and classification pipeline, evaluating an imbalanced classifier, explaining predictions, integrating the model into an application, testing the implementation, and deploying it. The important point is that the system is not only a model; it is an explainable decision-support prototype with clear limitations and responsible-use considerations.
