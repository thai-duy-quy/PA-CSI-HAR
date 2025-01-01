import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def draw_acc(epochs,train,val):
    xpoints = np.arange(0, epochs,dtype=int)
    train = np.array(train)
    val = np.array(val)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(xpoints, train)
    plt.plot(xpoints, val)
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

def draw_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def draw_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix
    target_names=['No movement', 'Falling', 'Sitting down/standing up', 'Walking', 'Turning', 'Picking up']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def draw_confusion_matrix_2(y_true, y_pred):
    # Define the mapping from numeric labels to string labels
    label_mapping = {
        0: 'No movement',
        1: 'Falling',
        2: 'Sitting down / Standing up',
        3: 'Walking',
        4: 'Turning',
        5: 'Picking up a pen'
    }

    # Convert the numeric labels to string labels for display purposes
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Generate confusion matrix with numeric values
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_mapping.keys()))

    # Convert confusion matrix to percentage format (row-wise)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Set up the figure and axes
    plt.figure(figsize=(8, 6))

    # Plot the heatmap with string labels
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)

    # Add axis labels and title
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Reference', fontsize=12)
    plt.title('Confusion Matrix of MultiEnv LOS (Office)', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

