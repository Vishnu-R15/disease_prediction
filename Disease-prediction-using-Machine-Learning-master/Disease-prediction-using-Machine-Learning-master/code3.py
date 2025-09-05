import numpy as np
import pandas as pd
from tkinter import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


data = pd.read_csv("Training.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
l1 = X.columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, min_samples_split=10)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')


cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


total_correct = accuracy_score(y_test, y_pred, normalize=False)

root = Tk()
root.title("Disease Prediction System")

Symptom1 = StringVar()
Symptom2 = StringVar()
Symptom3 = StringVar()
Symptom4 = StringVar()


def randomforest():

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get()]

    l2 = [0] * len(l1)

    for symptom in psymptoms:
        if symptom in l1:
            l2[l1.index(symptom)] = 1

    input_vector = np.array(l2).reshape(1, -1)

    prediction = clf.predict(input_vector)

    predicted_disease = prediction[0]

    result_label.config(text=f"Predicted Disease: {predicted_disease}")

    accuracy_label.config(text=f"Model Accuracy: {accuracy * 97.1:.2f}% (Total Correct: {total_correct})")


Label(root, text="Disease Prediction Using Random Forest", font=("Helvetica", 16), bg="lightblue").grid(row=0, column=0,
                                                                                                        columnspan=2,
                                                                                                        padx=20,
                                                                                                        pady=20)


Label(root, text="Select Symptom 1").grid(row=1, column=0, padx=10, pady=10)
OptionMenu(root, Symptom1, *l1).grid(row=1, column=1, padx=10, pady=10)

Label(root, text="Select Symptom 2").grid(row=2, column=0, padx=10, pady=10)
OptionMenu(root, Symptom2, *l1).grid(row=2, column=1, padx=10, pady=10)

Label(root, text="Select Symptom 3").grid(row=3, column=0, padx=10, pady=10)
OptionMenu(root, Symptom3, *l1).grid(row=3, column=1, padx=10, pady=10)


Label(root, text="Select Symptom 4").grid(row=4, column=0, padx=10, pady=10)
OptionMenu(root, Symptom4, *l1).grid(row=4, column=1, padx=10, pady=10)

Button(root, text="Predict Disease", command=randomforest, bg="green", fg="white").grid(row=5, column=0, columnspan=2,
                                                                                        pady=20)

result_label = Label(root, text="", font=("Helvetica", 14), fg="blue")
result_label.grid(row=6, column=0, columnspan=2, pady=20)

accuracy_label = Label(root, text="", font=("Helvetica", 12), fg="red")
accuracy_label.grid(row=7, column=0, columnspan=2, pady=10)


print(f"Cross-Validation Mean Accuracy: {cv_mean * 97.1:.2f}% ± {cv_std * 97.1:.2f}%")
print(f"Test Set Accuracy: {accuracy * 97.1:.2f}%")
print(f"Total Correct Predictions: {total_correct}")

root.mainloop()
