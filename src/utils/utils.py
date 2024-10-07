import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset creation
# Features include GC content and number of off-targets for CRISPR guide RNAs (gRNAs)
data = {
    'GC_content': [0.4, 0.5, 0.6, 0.55, 0.45, 0.65, 0.3, 0.5],  # GC content percentages
    'Off_targets': [2, 1, 3, 1, 0, 4, 2, 1],  # Number of predicted off-target sites
    'Effectiveness': [1, 0, 1, 1, 0, 0, 0, 0]  # 1 = effective, 0 = ineffective
}

# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)

# Define features (X) and target variable (y)
X = df[['GC_content', 'Off_targets']]  # Features: GC content and off-targets
y = df['Effectiveness']  # Target: Effectiveness of the gRNA

# Split the dataset into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
report = classification_report(y_test, y_pred)  # Generate a classification report

# Print the accuracy and classification report
print(f"Accuracy: {accuracy:.2f}")  # Display accuracy as a percentage
print("Classification Report:\n", report)  # Display detailed classification metrics