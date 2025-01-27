# Import all the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading the dataset
data = pd.read_csv(r'C:\Users\sammo\Desktop\Artificial Intelligence\Assessment_2\creditcard.csv')

# Assistance of chat-gpt was used for the alignment and implementation of the model
# Preprocessing the data
X = data.drop(['Class'], axis=1)
y = data['Class']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Training Logistic Regression with balanced class weights
logistic_model = LogisticRegression(max_iter=5000, class_weight='balanced')
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]

# Adjusting the threshold to improve precision
threshold = 0.8  
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Adjusted model evaluation
accuracy_adjusted = accuracy_score(y_test, y_pred_adjusted)
precision_adjusted = precision_score(y_test, y_pred_adjusted)
recall_adjusted = recall_score(y_test, y_pred_adjusted)
f1_adjusted = f1_score(y_test, y_pred_adjusted)
roc_auc_adjusted = roc_auc_score(y_test, y_pred_proba)
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)

# Printing the evaluation metrics
print("Logistic Regression Evaluation Metrics:")
print(f"Accuracy: {accuracy_adjusted:.4f}")
print(f"Precision: {precision_adjusted:.4f}")
print(f"Recall: {recall_adjusted:.4f}")
print(f"F1 Score: {f1_adjusted:.4f}")
print(f"ROC AUC Score: {roc_auc_adjusted:.4f}")

# Displaying the Confusion Matrix
print("\nConfusion Matrix for Logistic Regression:")
print("True Legitimate (0) | True Fraud (1)")
print(conf_matrix_adjusted)

# Plotting Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix for Logistic Regression', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Adding the text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plotting the confusion matrix
plot_confusion_matrix(conf_matrix_adjusted, classes=['Legitimate', 'Fraud'])

# Plotting the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc_adjusted:.4f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression (Adjusted Threshold)')
plt.legend()
plt.show()
