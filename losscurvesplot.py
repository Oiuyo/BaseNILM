import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_curve(train_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    epochs = train_data['Step']
    train_loss = train_data['Value']

    test_loss = test_data['Value']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_loss, label='Validation Accuracy', color='red')

    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss_curve('logs/run-train-tag-epoch_lossMetric.csv', 'logs/run-validation-tag-epoch_lossMetric.csv')
