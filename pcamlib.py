import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,\
roc_curve, roc_auc_score, classification_report, precision_score, recall_score

def load_pcam():
    pcam, pcam_info = tfds.load("patch_camelyon", with_info=True)
    print(pcam_info)
    return pcam, pcam_info

def convert_sample(sample):
    # Credit: Geert Litjens
    image, label = sample['image'], sample['label']
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, 2, dtype=tf.float32)
    return image, label

def build_pipelines(pcam):
    # Credit: Geert Litjens
    train_pipeline = pcam['train'].map(convert_sample, num_parallel_calls=8).shuffle(1024).repeat().batch(64).prefetch(2)
    valid_pipeline = pcam['validation'].map(convert_sample, num_parallel_calls=8).repeat().batch(128).prefetch(2)
    test_pipeline = pcam['test'].map(convert_sample, num_parallel_calls=8).batch(128).prefetch(2)

    return train_pipeline, valid_pipeline, test_pipeline

def save_history(history, filepath):
    # Save the history of the model to a pandas dataframe
    hist_df = pd.DataFrame(history.history)

    # Export the dataframe to a csv so it can be read into pandas later without re-training the model
    # filepath example: 'cnn1_history.csv'
    hist_csv_file = filepath
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def load_history(filepath):
    hist_df = pd.read_csv(filepath, index_col=0)

    return hist_df

def plot_history(hist_df, figsize=(10,4), title=None, save=False, filepath=None):
    # Create subplots
    plt.subplots(1, 2, figsize=figsize)

    plt.suptitle(title, fontsize=24)

    # Plot accuracies for train and validation sets
    plt.subplot(1, 2, 1)
    plt.plot(hist_df['accuracy'], label='Train', marker='o')
    plt.plot(hist_df['val_accuracy'], label='Validation', marker='o')
    plt.title('Training and Validation Accuracy', size=20)
    plt.xlabel('Epoch', size=16)
    plt.ylabel('Accuracy', size=16)
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(hist_df['loss'], label='Train', marker='o')
    plt.plot(hist_df['val_loss'], label='Validation', marker='o')
    plt.title('Training and Validation Loss', size=20)
    plt.xlabel('Epoch', size=16)
    plt.ylabel('Loss', size=16)
    plt.legend()

    # This ensures the subplots do not overlap
    plt.tight_layout()

    if save:
        # Sample filepath: 'plots/cnn1_acc_loss_plot.png'
        plt.savefig(filepath)

    # Show the subplots
    plt.show()

def plot_examples(pcam, save=False, filepath=None):
    train_iterator = pcam['train'].__iter__()

    plt.subplots(3, 3, figsize=(10,10))

    for i in range(9):

        plt.subplot(3, 3, i+1)

        train_image = train_iterator.get_next()

        image = train_image['image']
        label = int(train_image['label'])

        plt.imshow(train_image['image'])
        plt.title('Class Label: '+ str(label), size=16)

        # Create a Rectangle patch
        #     rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

        #     # Add the patch to the Axes
        #     ax.add_patch(rect)

    plt.tight_layout()

    if save:
        # Sample filepath: 'data/plots/example_images.png'
        plt.savefig(filepath)

    plt.show()

def plot_cf_matrix(y_true, y_pred, normalize=True, save=False, filepath=None):
    cf_matrix = confusion_matrix(y_true, y_pred)

    if normalize:
        cf_matrix = cf_matrix / cf_matrix.sum(axis=1)

    ConfusionMatrixDisplay(cf_matrix, display_labels=['Healthy (0)', 'Cancer (1)']).plot()

    if save:
        # Sample filepath: 'plots/cnn1_cf_matrix.png'
        plt.savefig(filepath)

    plt.show()

def plot_roc_curve(y_true, y_proba, save=False, filepath=None):

    if y_proba.shape[1] == 2:
        # y_proba is still one-hot encoded, so grab only the class 1 probabilities
        y_proba = np.array([i[1] for i in y_proba])

    fprs, tprs, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, color='darkorange',
             lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)', size=16)
    plt.ylabel('True Positive Rate (TPR)', size=16)
    plt.title('ROC Curve for Cancer Detection', size=20)
    plt.legend(loc="best")

    if save:
        # Sample filepath: 'plots/cnn1_roc.png'
        plt.savefig(filepath)

    plt.show()

    print(f'Area under curve (AUC):{roc_auc}')
    print(fprs.shape)
    print(tprs.shape)

def generate_y_true(pcam):
    # Initialize iterator so it starts from the beginning
    test_iterator = pcam['test'].__iter__()

    # Create an empty list to store the test labels
    y_true = []

    # There are 32768 images in the test set
    for i in range(32768):
        y_true.append(int(test_iterator.get_next()['label']))

    return np.array(y_true)

def generate_y_proba(model, test_pipeline, class_1=True, save=False, filepath=None):
    y_proba = model.predict(test_pipeline)

    if class_1:
        # Return just the class_1 predictions
        y_proba = np.array([i[1] for i in y_proba])

    if save:
        y_proba_df = pd.DataFrame(y_proba)

        y_proba_csv_file = filepath
        with open(y_proba_csv_file, mode='w') as f:
            y_proba_df.to_csv(f)

    return y_proba

def load_y_proba(filepath):
    y_proba = pd.read_csv(filepath, index_col=0).to_numpy()

    return y_proba

def generate_y_pred(y_proba, threshold=0.5):

    if y_proba.shape[1] == 2:
        # y_proba is still one-hot encoded, so grab only the class 1 probabilities
        y_proba = np.array([i[1] for i in y_proba])

    # Predict the positive class when the probability exceeds the given threshold
    y_pred = np.where(y_proba >= threshold, 1, 0)

    return y_pred

def print_test_accuracy(model, test_pipeline, steps=128, verbose=0):
    print("Test set accuracy is {0:.4f}".format(model.evaluate(test_pipeline, steps=steps, verbose=verbose)[1]))

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))

def plot_misclassified_images(pcam, y_true, y_pred, save=False, filepath=None):
    test_iterator = pcam['test'].__iter__()
    i = 0
    j = 0

    plt.subplots(3, 3, figsize=(10,10))

    while True:
        next_image = test_iterator.get_next()

        image = next_image['image']
        label = int(next_image['label'])

        # If the image was misclassified
        if y_true[i] != y_pred[i]:
            print(i)

            plt.subplot(3, 3, j+1)

            plt.imshow(image)

            title = f'Predicted Label: {str(y_pred[i])} ({str(label)})'
            plt.title(title, size=16)

            j += 1

        # Stop the loop after 9 images are plotted
        if j == 9:
            break

        i += 1

    plt.tight_layout()

    if save:
        # Sample filepath: 'data/plots/cnn1_misclassified_images.png'
        plt.savefig(filepath)

    plt.show()
