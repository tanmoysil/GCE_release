import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gce.early_stopping import EarlyStopping
from gce.utils import *


def save_train_test_data(traindata_loader, testdata_loader, save_dir):
    """
    Save the train/test split data for later analysis
    """
    # Extract data from dataloaders
    X_train_list, y_train_list = [], []
    for inputs, targets in traindata_loader:
        X_train_list.append(inputs.numpy())
        y_train_list.append(targets.numpy())

    X_test_list, y_test_list = [], []
    for inputs, targets in testdata_loader:
        X_test_list.append(inputs.numpy())
        y_test_list.append(targets.numpy())

    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.hstack(y_test_list)

    # Save
    data_path = os.path.join(save_dir, "classifier_train_test_split.npz")
    np.savez(
        data_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
    )
    print(f"✓ Saved train/test split to {data_path}")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(
        f"  Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}"
    )

    return X_train, y_train, X_test, y_test


def evaluate_binary_classifier(model, dataloader, device, dataset_name="Test"):
    """
    Evaluate binary classifier and return predictions, probabilities, and targets
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            _, prediction_probabilities = model(inputs)
            predictions = torch.argmax(prediction_probabilities, dim=-1)

            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            # For binary classification, use probability of positive class
            all_probabilities.append(prediction_probabilities[:, 1].cpu().numpy())

    y_true = np.hstack(all_targets)
    y_pred = np.hstack(all_predictions)
    y_prob = np.hstack(all_probabilities)

    accuracy = (y_pred == y_true).sum() / len(y_true)

    print(f"\n{dataset_name} Set Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["OFF", "ON"], digits=4))

    return y_true, y_pred, y_prob, accuracy


def plot_roc_curves_combined(
    y_train_true, y_train_prob, y_test_true, y_test_prob, save_path
):
    """
    Plot train and test ROC curves on the same plot for binary classification
    """
    # Calculate train ROC
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_prob)
    auc_train = auc(fpr_train, tpr_train)

    # Calculate test ROC
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_prob)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 6))

    # Plot train curve
    plt.plot(
        fpr_train,
        tpr_train,
        color="blue",
        lw=2.5,
        label=f"Train ROC (AUC = {auc_train:.3f})",
    )

    # Plot test curve
    plt.plot(
        fpr_test,
        tpr_test,
        color="darkorange",
        lw=2.5,
        label=f"Test ROC (AUC = {auc_test:.3f})",
    )

    # Plot chance line
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Train vs Test", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.savefig(save_path.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close()

    return auc_train, auc_test


def plot_confusion_matrix(y_true, y_pred, save_path, dataset_name="Test"):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["OFF", "ON"],
        yticklabels=["OFF", "ON"],
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"Confusion Matrix - {dataset_name} Set", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.savefig(save_path.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_history(train_accuracies, test_accuracies, save_path):
    """
    Plot training and test accuracy over epochs
    """
    epochs = range(1, len(train_accuracies) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        train_accuracies,
        "b-o",
        label="Train Accuracy",
        linewidth=2,
        markersize=4,
    )
    plt.plot(
        epochs, test_accuracies, "r-s", label="Test Accuracy", linewidth=2, markersize=4
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Training History", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.savefig(save_path.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_summary(metrics, save_path):
    """
    Save metrics to text file
    """
    with open(save_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CLASSIFIER EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Train Accuracy: {metrics['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy:  {metrics['test_accuracy']:.4f}\n\n")

        f.write(f"Train AUC: {metrics['train_auc']:.4f}\n")
        f.write(f"Test AUC:  {metrics['test_auc']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("TRAINING CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(metrics["train_report"] + "\n\n")

        f.write("=" * 70 + "\n")
        f.write("TEST CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(metrics["test_report"] + "\n")


def train(config, seed):
    """
    Given a set of config parameters, train a neural network
    :config: dictionary hyperparameters specified in the config file
    :param seed: seed for reproducibility
    :return: nn.module: model
    """

    # Create output directory
    output_dir = config.get("output_dir", "./classifier_results_halfamp_train")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 70}\n")

    device = torch.device(config["device"])

    # Select the dataloaders for the right dataset
    traindata_loader, testdata_loader = select_dataloader(config)

    # Save train/test split
    X_train, y_train, X_test, y_test = save_train_test_data(
        traindata_loader, testdata_loader, output_dir
    )

    # Select model, optimizer, scheduler and loss function
    model = select_classifier(config).to(device)

    optimizer = select_optimizer(config, model)
    loss_module = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, min_lr=1e-6, patience=3
    )

    early_stopping = EarlyStopping(patience=20, delta=0.001)

    # Record current time to record training time
    start_time = time.time()

    # Track training history
    train_accuracies = []
    test_accuracies = []

    for epoch in range(config["epochs"]):
        batch_count = 0
        train_accuracy = 0
        model.train()

        for inputs, targets in traindata_loader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate model output and loss
            predictions, prediction_probabilities = model(inputs)
            loss = loss_module(predictions, targets)

            accuracyPredictions = torch.argmax(prediction_probabilities, dim=-1)
            correct = (accuracyPredictions == targets).sum().item()
            train_accuracy += correct / len(targets)

            # Execute backwards propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not config["no_print"]:
                print(
                    "[Train Epoch %d/%d] [Batch %d] time: %4.4f [loss: %f]"
                    % (
                        epoch + 1,
                        config["epochs"],
                        batch_count,
                        time.time() - start_time,
                        loss.item(),
                    )
                )

            batch_count += 1

        # Calculate test accuracy
        model.eval()
        with torch.no_grad():
            total_comparisons = 0
            total_correct = 0
            for inputs, targets in testdata_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                _, prediction_probabilities = model(inputs)

                predictions = torch.argmax(prediction_probabilities, dim=-1)
                total_comparisons += len(targets)
                total_correct += (predictions == targets).sum().item()

            test_accuracy = total_correct / total_comparisons
            train_acc_epoch = train_accuracy / batch_count

            train_accuracies.append(train_acc_epoch)
            test_accuracies.append(test_accuracy)

            print(
                "[Test Epoch %d/%d] [test accuracy: %f, train accuracy: %f]"
                % (epoch + 1, config["epochs"], test_accuracy, train_acc_epoch)
            )

        scheduler.step(test_accuracy)
        early_stopping(test_accuracy, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save model
    if config["save_classifier"]:
        model_path = os.path.join(output_dir, config["model_name"])
        torch.save(model.state_dict(), model_path)

    # =========================================================================
    # FINAL EVALUATION AND PLOTTING
    # =========================================================================

    # Evaluate on train set
    y_train_true, y_train_pred, y_train_prob, train_acc_final = (
        evaluate_binary_classifier(model, traindata_loader, device, "Train")
    )

    # Evaluate on test set
    y_test_true, y_test_pred, y_test_prob, test_acc_final = evaluate_binary_classifier(
        model, testdata_loader, device, "Test"
    )

    # Combined ROC curve (train + test)
    train_auc, test_auc = plot_roc_curves_combined(
        y_train_true,
        y_train_prob,
        y_test_true,
        y_test_prob,
        os.path.join(output_dir, "roc_curves_combined.svg"),
    )

    # Generate confusion matrices
    plot_confusion_matrix(
        y_train_true,
        y_train_pred,
        os.path.join(output_dir, "confusion_matrix_train.svg"),
        "Train",
    )

    plot_confusion_matrix(
        y_test_true,
        y_test_pred,
        os.path.join(output_dir, "confusion_matrix_test.svg"),
        "Test",
    )

    # Plot training history
    plot_training_history(
        train_accuracies,
        test_accuracies,
        os.path.join(output_dir, "training_history.svg"),
    )

    # Save metrics summary
    metrics = {
        "train_accuracy": train_acc_final,
        "test_accuracy": test_acc_final,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_report": classification_report(
            y_train_true, y_train_pred, target_names=["OFF", "ON"], digits=4
        ),
        "test_report": classification_report(
            y_test_true, y_test_pred, target_names=["OFF", "ON"], digits=4
        ),
    }

    save_metrics_summary(metrics, os.path.join(output_dir, "metrics_summary.txt"))

    return model


if __name__ == "__main__":
    # Create parser to get hyperparameters from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pd_all_save.yml")
    parser.add_argument(
        "--output_dir",
        default="./classifier_results_halfamp_train",
        help="Directory to save all outputs",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))
    config = to_classifier_config(config)

    # Add output directory to config
    config["output_dir"] = args.output_dir

    print(config)
    train(config, 1)
