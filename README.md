Customer Churn Prediction using Feature Selection and Neural Networks in PyTorch
Overview
This project implements a customer churn prediction pipeline using feature selection techniques and a neural network model built with PyTorch. The pipeline includes data preprocessing, feature selection via a RandomForest model, and evaluation using several performance metrics such as loss, accuracy, confusion matrix, ROC curve, and AUC score.

Key Steps and Tasks
1. Data Preprocessing
Loaded the customer churn dataset and prepared the features and target labels.
Applied OneHotEncoder to convert categorical columns to numerical form and used StandardScaler to normalize numeric columns.
Handled missing values and ensured consistency in data types for numeric features.
2. Feature Selection
Trained a RandomForestClassifier to compute feature importance.
Selected top contributing features using SelectFromModel based on the importance scores from the RandomForest model.
3. Neural Network Construction
Built a neural network model using PyTorch with the following architecture:
Three fully connected layers with ReLU activation functions.
Sigmoid output layer for binary classification (churn or no churn).
Trained the neural network on the selected top features using Binary Cross-Entropy Loss as the loss function and Adam optimizer.
4. Model Evaluation
Tracked and plotted the training loss and accuracy across epochs to monitor model performance.
Generated a confusion matrix to evaluate the model's classification performance on the test set.
Plotted the ROC curve and computed the AUC score to assess the model's ability to distinguish between churn and non-churn customers.
(Optional) Visualized the Precision-Recall curve for a detailed analysis of model performance, especially useful for imbalanced datasets.
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install the required dependencies: Install the dependencies listed in requirements.txt (PyTorch, scikit-learn, matplotlib, etc.):

bash
Copy code
pip install -r requirements.txt
Run the model: Execute the model by running the Python script:

bash
Copy code
python churn_prediction.py
View Results: The training loss, accuracy curves, confusion matrix, ROC curve, and other evaluation metrics will be displayed.

Plots and Visualizations
Training Loss and Accuracy: Shows how the model converges during training.
Confusion Matrix: Visualizes true positives, true negatives, false positives, and false negatives.
ROC Curve: Shows the tradeoff between true positive rate and false positive rate.
Precision-Recall Curve (Optional): Useful for understanding the model performance in cases of imbalanced classes.
