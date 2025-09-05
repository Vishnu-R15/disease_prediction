import matplotlib.pyplot as plt


algorithms = ['Random Forest', 'Naive Bayes', 'Decision Tree']
accuracies = [95.1, 94.1, 94.3]


plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, accuracies, color=['gold', 'blue', 'green'])


bars[0].set_color('gold')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.2, f"{acc:.1f}%", ha='center', fontsize=12)


plt.title("Accuracy Comparison of Machine Learning Algorithms", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xlabel("Algorithms", fontsize=14)
plt.ylim(94, 97.5)


plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
