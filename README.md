# DECISION-TREE-IMPLEMENTATION
"COMPANY" : CODTECH IT SOLUTIONS
"NAME" : SRIMURUGAN
"INTERN ID" : CT08VRH
"DOMAIN" : MACHINE LEARNING
"DURATION" : 4 WEEKS
"MENTOR" : Muzammil Ahmed


This Python script demonstrates  to implement a Decision Tree Classifier using the Iris dataset, a well-known dataset for classification tasks. The script follows a structured workflow that includes data loading, preprocessing, model training, evaluation, and visualization.

Key Steps in the Code:
Importing Required Libraries:   numpy and pandas for numerical and data manipulation.
matplotlib.pyplot for visualizing the decision tree.
sklearn.model_selection.train_test_split for splitting the dataset.
sklearn.tree.DecisionTreeClassifier for training a decision tree model.
sklearn.metrics for evaluating the model.
sklearn.datasets for loading the built-in Iris dataset.
Loading and Preparing the Data:

The Iris dataset is loaded using datasets.load_iris().
A DataFrame is created with feature columns.
The target variable (species) is added to the DataFrame.
Splitting the Dataset:

The dataset is split into training (80%) and testing (20%) sets using train_test_split().
Training the Decision Tree Model:

A DecisionTreeClassifier is initialized with max_depth=3 to prevent overfitting.
The model is trained on the training data using fit().
Model Prediction and Evaluation:

The trained model predicts labels for the test set.
The accuracy of the model is computed using accuracy_score().
A classification report is printed to show performance metrics such as precision, recall, and F1-score.
Visualizing the Decision Tree:

The decision tree structure is plotted using plot_tree(), which helps in understanding decision splits.
