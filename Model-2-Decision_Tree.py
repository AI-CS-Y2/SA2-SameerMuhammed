# Import all the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import graphviz

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

# Training the Decision Tree with class weights
decision_tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
decision_tree_model.fit(X_train, y_train)

# Make predictions
y_pred = decision_tree_model.predict(X_test)
y_pred_proba = decision_tree_model.predict_proba(X_test)[:, 1]

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# Printing the evaluation metrics
print("Decision Tree Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Displaying Confusion Matrix in Terminal
print("\nConfusion Matrix:")
print("True Legitimate (0) | True Fraud (1)")
print(conf_matrix)

# Plotting the Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix for Decision Tree', cmap=plt.cm.Greens):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Adding text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(conf_matrix, classes=['Legitimate', 'Fraud'])

# Plotting the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='green', label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend()
plt.show()

# Graphviz Visualization of Decision Tree
feature_names = data.drop(['Class'], axis=1).columns  
class_names = ['Legitimate', 'Fraud']  

# Exporting the decision tree to a DOT format
dot_data = export_graphviz(
    decision_tree_model,
    out_file=None,  
    feature_names=feature_names,
    class_names=class_names,
    filled=True,  
    rounded=True, 
    special_characters=True  
)

# Creating a Graphviz Source and rendering it
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saving it as a PDF file 
graph.view()  
