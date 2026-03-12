import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-darkgrid")

def plot_two_models(csv_path_1, csv_path_2):

    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    os.makedirs("PNG", exist_ok=True)

    epochs1 = range(1, len(df1) + 1)
    epochs2 = range(1, len(df2) + 1)

    model1_name = csv_path_1[4:-21]
    model2_name = csv_path_2[4:-21]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Сравнение моделей: {model1_name} и {model2_name}", fontsize=18, fontweight="bold")

    metrics = ["accuracy_val", "loss_val", "precision_val", "recall_val"]
    titles = ["Точность (Accuracy)", "Потери (Loss)", "Точность (Precision)", "Полнота (Recall)"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = divmod(i, 2)
        ax[row][col].plot(epochs1, df1[metric], label=model1_name, linewidth=2.5)
        ax[row][col].plot(epochs2, df2[metric], label=model2_name, linewidth=2.5)
        ax[row][col].set_title(title, fontsize=14)
        ax[row][col].set_xlabel("Эпоха", fontsize=12)
        ax[row][col].set_ylabel(title, fontsize=12)
        ax[row][col].legend(fontsize=10)
        ax[row][col].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"PNG/{model1_name}_VS_{model2_name}_combined.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

    print(f"График сохранён: {filename}")
