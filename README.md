## Detect_Logo
  
  This is a machine learning project to detect logos in images using a combination of ResNet50V2 model for feature extraction, Flatten for flattening the features, and a Keras model for training and prediction. 
  Additionally, the VGG16 model from Keras Applications is used to compare the performance of different feature extraction models. 
  The glob module is used for loading images from the dataset.
  
## Dataset

  The dataset used for this project is a custom dataset containing images of different logos. 
  The dataset is divided into two folders - train and test - and each folder contains subfolders for each class of logo.
  

## Requirements
  
  This project requires Python 3.x and the following libraries:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    keras
    tensorflow
    nltk
    
 ## You can install the required libraries using pip:
    
    pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow nltk
    
## Files

  The project consists of the following files:

    detect_logo.ipynb: Jupyter notebook containing the implementation of the ResNet50V2 model for logo detection, VGG16 model for logo detection
    dataset: folder containing the custom dataset.
    
## Conclusion

    In this project, we have seen how to use deep learning models to detect logos in images. We have also seen how to preprocess the dataset, extract image features, and evaluate the performance of the models. 
    The ResNet50V2 model has shown better results than the VGG16 model, but other models can also be used depending on the specific requirements of the problem. The glob module is useful for loading images from the dataset, and NLP techniques can be used to further improve the quality of the predictions.



