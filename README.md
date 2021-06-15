# capstone
BrainStation Capstone Project - Histopathological Cancer Detection Using Convolutional Neural Networks
Author: Jared Lauer

Following is a description of the files contained in this project. I recommend starting from the EDA notebook before proceeding to the notebooks containing CNN models.

1. Capstone - EDA.ipynb
 
     Background and Exploratory Data Analysis for this project, which includes some background information on breast cancer, the PCam dataset, and why histopathological lymph node samples are relevant in cancer diagnosis. Next, I look at some sample images from the data
     
2. Capstone - CNN1.ipynb

    The first CNN for this project is based on the architecture proposed by one of the authors of the dataset, Geert Litjens. I make an effort to improve the performance of this model through hyperparameter optimization and changing the architecture by adding different layer types.
    
3. Capstone - ResNet50.ipynb

    The first transfer learning model for this project is ResNet50 which has been pretrained on the ImageNet database. I add some dense layers to the output of this model and examine its performance.
    
4. Capstone - VGG16.ipynb

    The second transfer learning model for this project is VGG16 which has been pretrained on the ImageNet database. Because the model in CNN1 is loosely based on this model architecture, I decided to examine it as well. I add some dense layers to the output of this model and examine its performance.
    
5. pcamlib.py

    A python script containing helper functions which appear in all of the various Jupyter notebooks, mainly used for loading the PCam dataset, saving and loading model outputs, and making the various plots I use to analyze the models.
    
6. GitHub Commands

    A Google Colab notebook containing the necessary commands to clone, push and pull data from my capstone repo on GitHub.
    
7. data folder

    A folder containing saved models and other data which can be loaded into a notebook to analyze the performance of a trained model without training it again, saving time and ensuring repeatablility of results. 
    Subfolders:
    
    data/models: saved copies of trained models created using the keras function model.save()
    data/plots: saved plots which may include confusion matrices, training histories, ROC curves, etc.
    data/y_proba: contains .csv files of trained model prediction probabilities. Especially useful for the transfer learning models which are too large to track using git when saving using model.save()
    
    

   