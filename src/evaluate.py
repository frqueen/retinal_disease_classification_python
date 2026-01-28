import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_dataset, class_names):
    """
    Evaluates a trained model on a test dataset and
    returns classification report and confusion matrix.
    """

    # Convert test dataset to numpy arrays
    test_iter = test_dataset.as_numpy_iterator()
    images = []
    labels = []

    while True:
        try:
            batch = next(test_iter)
            images.append(batch[0])
            labels.append(batch[1])
        except StopIteration:
            break

    X_test = np.concatenate(images, axis=0)
    y_true = np.concatenate(labels, axis=0)

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return report, cm


def plot_confusion_matrix(cm, class_names):
    """
    Plots confusion matrix using seaborn.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
