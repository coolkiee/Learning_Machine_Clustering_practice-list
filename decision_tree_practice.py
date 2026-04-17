import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("--- Steps 1, 2 & 3: Loading and Splitting Data ---")
# ... kodun geri kalanı aynı şekilde devam ediyor ...

data = load_iris()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print(f"Total samples: {X.shape[0]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

print("\n--- Steps 4, 5, 6 & 7: Model Training, Prediction and Accuracy ---")

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, Y_train)

train_predictions = dt_model.predict(X_train)
test_predictions = dt_model.predict(X_test)

train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

print(f"Training Data Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Data Accuracy: {test_accuracy * 100:.2f}%")

print("\n--- Step 9: Performance Test with 10 Different 'random_state' Values ---")

accuracies = []

for i in range(10):
    X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(X, Y, test_size=0.25, random_state=i)

    dt_cv = DecisionTreeClassifier(random_state=42)
    dt_cv.fit(X_train_cv, Y_train_cv)

    preds = dt_cv.predict(X_test_cv)
    acc = accuracy_score(Y_test_cv, preds)
    accuracies.append(acc)

    print(f"Accuracy for Random State {i}: {acc * 100:.2f}%")

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\n--- Statistical Results ---")
print(f"Mean of 10 trials: {mean_accuracy * 100:.2f}%")
print(f"Standard Deviation of 10 trials: {std_accuracy * 100:.2f}%")

print("\n--- Step 10: Testing Different Split Ratios ---")

test_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
ratio_accuracies = []

for ratio in test_ratios:
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=ratio, random_state=42)

    model_ratio = DecisionTreeClassifier(random_state=42)
    model_ratio.fit(X_tr, Y_tr)

    score = accuracy_score(Y_te, model_ratio.predict(X_te))
    ratio_accuracies.append(score)
    print(f"Test Ratio: {ratio} -> Accuracy: {score * 100:.2f}%")

# --- Chart 1: Accuracy vs Test Ratio ---
plt.figure(figsize=(8, 5))
plt.plot(test_ratios, ratio_accuracies, marker='o', linestyle='-', color='b', markersize=8)
plt.title('Model Accuracy by Test Data Ratio', fontsize=14)
plt.xlabel('Test Data Ratio (test_size)', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(test_ratios)
plt.show()

print("\n--- Step 11: Plotting the Decision Tree ---")

# --- Chart 2: Decision Tree Visualization ---
plt.figure(figsize=(16, 10))
plot_tree(dt_model,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True,
          rounded=True,
          fontsize=10)

plt.title("Iris Dataset - Decision Tree Model", fontsize=16, fontweight='bold')
plt.show()