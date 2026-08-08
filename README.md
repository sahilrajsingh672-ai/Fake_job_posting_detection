# Fake_job_posting_detection
This code implements a machine learning model to detect fraudulent job postings using the Random Forest classification algorithm. It begins by importing essential libraries such as pandas and numpy for data handling, and various modules from sklearn for preprocessing, model building, and evaluation.

The dataset is loaded from a CSV file named "fake_job_postings.csv" into a DataFrame. Basic information about the dataset, including its shape, first few rows, and column names, is displayed to understand the data structure.

A set of initial numerical features—telecommuting, has_company_logo, and has_questions—are selected. Since the dataset contains categorical variables such as employment type, experience level, education, industry, and function, LabelEncoder is used to convert these text-based columns into numerical form. Missing values in these columns are replaced with 'None' before encoding to avoid errors.

These newly encoded columns are then added to the feature list. The input features (X) are selected from the dataset using this updated feature list, while the target variable (y) is set as the 'fraudulent' column, which indicates whether a job posting is fake or genuine.

The target variable is also encoded into numerical format using LabelEncoder. The dataset is then split into training and testing sets using an 80-20 ratio, ensuring class balance through stratified sampling.

Next, StandardScaler is applied to normalize the feature values, which helps improve model performance. A RandomForestClassifier model is initialized with 200 trees, a maximum depth of 10, and balanced class weights to handle any class imbalance. The model is then trained using the scaled training data.

After training, predictions are made on the test dataset. The model's performance is evaluated using accuracy score, classification report (which includes precision, recall, and F1-score), and confusion matrix.

Feature importance is calculated to determine which features contribute most to the prediction. This helps in understanding the model's decision-making process.

Finally, the model is tested on a single sample from the test set. The prediction and its probability are displayed, indicating whether the job posting is fraudulent or not, along with confidence scores.

Overall, this code demonstrates a complete machine learning pipeline including data preprocessing, feature engineering, model training, evaluation, and prediction.
