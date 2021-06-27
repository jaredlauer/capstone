import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# TODO: import only necessary tensorflow functions
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,\
roc_curve, roc_auc_score, classification_report, accuracy_score, precision_score, recall_score

# TODO: Add docstrings

# Loads the Patch Camelyon dataset
def load_pcam(data_dir=None):
    pcam, pcam_info = tfds.load("patch_camelyon", with_info=True, data_dir=data_dir)
    print(pcam_info)
    return pcam, pcam_info

# Converts images to prepare them for modelling
def convert_sample(sample):
    # Credit: Geert Litjens
    image, label = sample['image'], sample['label']
    
    image = tf.image.convert_image_dtype(image, tf.float32)  
    label = tf.one_hot(label, 2, dtype=tf.float32)
    return image, label

# Alternative to convert_sample which also converts images to grayscale
def convert_sample_grayscale(sample):
    image, label = sample['image'], sample['label']
    
    image = tf.image.rgb_to_grayscale(image, name=None)
    image = tf.image.convert_image_dtype(image, tf.float32)  
    
    label = tf.one_hot(label, 2, dtype=tf.float32)
    
    return image, label        

# Substitute for ImageDataGenerator which gets along with the TensorFlow Dataset object
def build_pipelines(pcam, grayscale=False):
    # Uses the grayscale version of convert_sample
    if grayscale:
        train_pipeline = pcam['train'].map(convert_sample_grayscale, num_parallel_calls=8).shuffle(1024).repeat().batch(64).prefetch(2)
        valid_pipeline = pcam['validation'].map(convert_sample_grayscale, num_parallel_calls=8).repeat().batch(128).prefetch(2)
        test_pipeline = pcam['test'].map(convert_sample_grayscale, num_parallel_calls=8).batch(128).prefetch(2)
    
    # Uses the normal version of convert_sample
    else:
        # Credit: Geert Litjens 
        train_pipeline = pcam['train'].map(convert_sample, num_parallel_calls=8).shuffle(1024).repeat().batch(64).prefetch(2)
        valid_pipeline = pcam['validation'].map(convert_sample, num_parallel_calls=8).repeat().batch(128).prefetch(2)
        test_pipeline = pcam['test'].map(convert_sample, num_parallel_calls=8).batch(128).prefetch(2)

    return train_pipeline, valid_pipeline, test_pipeline

# Export the training history to a .csv file
def save_history(hist_df, filepath):
    
    # Sample filepath: 'data/models/history/cnn1_history.csv'
    hist_csv_file = filepath
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    
# Loads model training history .csv into a pandas dataframe
def load_history(filepath):
    
    # Sample filepath: 'data/models/history/cnn1_history.csv'
    hist_df = pd.read_csv(filepath, index_col=0)

    return hist_df

# Plot the training accuracy and loss from training history
def plot_history(hist_df, figsize=(10,4), title=None, save=False, filepath=None):
    # Create subplots
    plt.subplots(1, 2, figsize=figsize)

    # Creates a title for the whole plot
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
        # Sample filepath: 'data/plots/cnn1_acc_loss_plot.png'
        plt.savefig(filepath)

    # Show the subplots
    plt.show()

# Plot the confusion matrix for a model
def plot_cf_matrix(y_true, y_pred, normalize=True, save=False, filepath=None):
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Turns the values in the confusion matrix into percentages
    if normalize:
        cf_matrix = cf_matrix / cf_matrix.sum(axis=1)

    ConfusionMatrixDisplay(cf_matrix, display_labels=['Healthy (0)', 'Cancer (1)']).plot()

    if save:
        # Sample filepath: 'data/plots/cnn1_cf_matrix.png'
        plt.savefig(filepath)

    plt.show()

# Plot the ROC curve and calculate AUC
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
        # Sample filepath: 'data/plots/cnn1_roc.png'
        plt.savefig(filepath)

    plt.show()

    print(f'Area under curve (AUC):{roc_auc}')

# Create a list of ground truth labels from a specified data split
def generate_y_true(pcam, split='test'):
    # Initialize iterator so it starts from the beginning
    iterator = pcam[split].__iter__()

    # Create an empty list to store the labels
    y_true = []

    if split == 'train':
        # There are 262144 images in the training set
        for i in range(262144):
            y_true.append(int(iterator.get_next()['label']))
    else:
        # There are 32768 images in the validation and test sets
        for i in range(32768):
            y_true.append(int(iterator.get_next()['label']))

    return np.array(y_true)

# Get predictions as probabilities from a trained model
def generate_y_proba(model, test_pipeline, class_1=False, save=False, filepath=None):
    y_proba = model.predict(test_pipeline)

    if class_1:
        # Return just the class_1 predictions rather than one-hot encoded predictions
        y_proba = np.array([i[1] for i in y_proba])

    # Save y_proba to a .csv file to load later without training the model
    if save:
        y_proba_df = pd.DataFrame(y_proba)
        
        # Sample filepath: 'data/models/cnn1_y_proba.csv'
        y_proba_csv_file = filepath
        with open(y_proba_csv_file, mode='w') as f:
            y_proba_df.to_csv(f)

    return y_proba

# Load y_proba from a .csv file
def load_y_proba(filepath):
    # Sample filepath: 'data/models/cnn1_y_proba.csv'
    y_proba = pd.read_csv(filepath, index_col=0).to_numpy()

    return y_proba

# Get predictions based on y_proba with the ability to change the decision threshold
def generate_y_pred(y_proba, threshold=0.5):

    if y_proba.shape[1] == 2:
        # y_proba is still one-hot encoded, so grab only the class 1 probabilities
        y_proba = np.array([i[1] for i in y_proba])

    # Predict the positive class when the probability exceeds the given threshold
    y_pred = np.where(y_proba >= threshold, 1, 0)

    return y_pred

# Print test set accuracy score
def print_test_accuracy(y_true, y_pred):
    print(accuracy_score(y_true, y_pred))

# Print the percentage the pathologist's workload has been reduced by pre-screening healthy images
def print_workload_reduction(y_pred):
    size_of_test_set = 32768
    
    # Cancerous images are class 1 predictions
    cancer_images = np.count_nonzero(y_pred)
    
    # Healthy images are class 0 predictions and are discarded
    healthy_images = size_of_test_set - cancer_images
    
    # Workload reduction is the percent of predicted healthy images expressed as a percentage of the test set
    workload_reduction = round((100*healthy_images / size_of_test_set), 1) 
    
    print(f'{workload_reduction}%')

# Print the classification report to get precision, accuracy, and f1 score to 4 decimal places
def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))

# Plot a 3x3 grid of sample images from a given data split, with the option for grayscale and saving the figure
def plot_examples(pcam, split='train', grayscale=False, save=False, filepath=None):
    iterator = pcam[split].__iter__()

    fig, ax = plt.subplots(3, 3, figsize=(10,10))
    
    # Plot title
    plt.suptitle(split + ' set samples', size=20)

    for i in range(9):
        
        ax = plt.subplot(3, 3, i+1)
        
        # Get the next image from the iterator
        sample_image = iterator.get_next()
        
        # Extract the image and its label
        image = sample_image['image']
        label = int(sample_image['label'])
        
        # Convert the image to grayscale if specified
        if grayscale:
            image = tf.image.rgb_to_grayscale(image)
            print(image.shape)
             
            # Need to change the colormap of matplotlib to 'Greys_r' or else the images look yellow/green when plotted
            ax.imshow(image, cmap='Greys_r')
        
        else:
            ax.imshow(image)
            
        plt.title('Class Label: '+ str(label), size=16)

        # Create a green rectangle patch to highlight the central 32 x 32 pixel region 
        # I couldn't find documentation for how the linewidth is extended, it's possible I've covered a couple pixels of the central region
        rect = patches.Rectangle((31, 31), 32, 32, linewidth=3, edgecolor='g', facecolor='none')

        # Add the patch to the axes
        ax.add_patch(rect)

    # Need to specify values for rect=[left, bottom, right, top] to ensure suptitle isn't overlapping the images
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        # Sample filepath: 'data/plots/example_images.png'
        plt.savefig(filepath)

    plt.show()

# Plot a 3x3 grid of images misclassified by the model, with the option to view different samples of images
def plot_misclassified_images(pcam, y_true, y_pred, grayscale=False, image_index=0, save=False, filepath=None):

    # Create an iterator object to iterate through images in the test set
    test_iterator = pcam['test'].__iter__()
    
    images_plotted = 0
    
    # If image_index is set to a value, iterate through the images to start from the specified index 
    # i.e. if image_index = 10, we iterate through the first 9 images here so when we load the next image inside the loop, we will load the 10th image
    for i in range(image_index):
        next_image = test_iterator.get_next()

    fig, ax = plt.subplots(3, 3, figsize=(10,10))
    
    # Title for the entire plot
    plt.suptitle('Misclassified Images from the Test Set', size=20)

    while True:
        next_image = test_iterator.get_next()

        image = next_image['image']
        label = int(next_image['label'])

        # If the image was misclassified
        if y_true[image_index] != y_pred[image_index]:
            
            ax = plt.subplot(3, 3, images_plotted+1)
            
            if grayscale:
                image = tf.image.rgb_to_grayscale(image)
             
                # Need to change the colormap of matplotlib to 'Greys_r' or else the images look yellow/green when plotted
                ax.imshow(image, cmap='Greys_r')
        
            else:
                ax.imshow(image)
                
            # Title format for image #1 which was predicted class 1 but is really class 0: 
            #       Image 1 
            # Predicted Label: 1 (0) 
            title = f'Image {str(image_index)}\nPredicted Label: {str(y_pred[image_index])} ({str(label)})'
            plt.title(title, size=16)

            # Create a green rectangle patch to highlight the central 32 x 32 pixel region 
            # I couldn't find documentation for how the linewidth is extended, it's possible I've covered a couple pixels of the central region
            rect = patches.Rectangle((31, 31), 32, 32, linewidth=3, edgecolor='g', facecolor='none')

            # Add the patch to the axes
            ax.add_patch(rect)

            images_plotted += 1

        # Stop the loop after 9 images are plotted
        if images_plotted == 9:
            break

        image_index += 1
        
    # Need to specify values for rect=[left, bottom, right, top] to ensure suptitle isn't overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        # Sample filepath: 'data/plots/cnn1_misclassified_images.png'
        plt.savefig(filepath)

    plt.show()