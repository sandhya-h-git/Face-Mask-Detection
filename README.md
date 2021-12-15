# Face-Mask-Detection
A Convolutional Neural Network model that classifies an image based on whether the person is wearing a face mask or not is built from scratch using Sequential API.

Dataset: https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset?select=New+Masks+Dataset

         The given dataset contains 3 directories:
         
          i.	  The train directory contains 600 images belonging to 2 classes (Mask, Non Mask – 300 files in each class).
          
          ii.	  The validation directory contains 306 images belonging to 2 classes (Mask, Non Mask – 153 files in each class)
          
          iii.	The test directory contains 100 images belonging to 2 classes (again Mask and Non Mask – 50 files in each class)
          
    
First the required libraries are imported. This includes _tensorflow_ – _keras_ for building the model (_ImageDataGenerator_ for data preprocessing, Sequential model, _ModelCheckpoint_ for callbacks, _RMSprop_ for model compilation etc.) and _pandas_, _matplotlib_ (_DataFrame_ and _pyplot_ for visualising model history)


The dataset is imported.


For data Preprocessing, _ImageDataGenerator_ is used.

The following has been performed on the training data:

    •	Rescaling
    
    •	Rotation (40)
    
    •	Width shift (0.2)
    
    •	Height shift (0.2)
    
    •	Shear (0.2)
    
    •	Zoom (0.2)
    
    •	Flip (horizontal)
    
    
The test data is also rescaled using _ImageDataGenerator_.


The train and validation data required for building the model are generated using _flow_from_directory()_ after _ImageDataGenerator_ is used. Here the target size is fixed as 150x150 and the batch size is fixed as 10.


The model is built using _Sequential()_ with the following layers:

    •	A Conv2D layer with kernel size 3x3, 100 filters with ReLU activation, taking in input of size 150x150x3 (input layer).
    
    •	A MaxPooling2D layer with window dimensions 2x2.
    
    •	A Conv2D layer with kernel size 3x3, 100 filters with ReLU activation.
    
    •	Another MaxPooling2D layer with window dimensions 2x2.
    
    •	A Flatten layer that converts 2D image data to 1D.
    
    •	A Dropout layer with keep probability of 0.5 to prevent overfitting.
    
    •	A Dense layer with 50 neurons and ReLU activation.
    
    •	A Dense layer with 2 neurons and SoftMax activation (output layer).
    

The model is then compiled with the following:

    •    _sparse_categorical_crossentropy_ loss function, since this is a categorical classification problem with the output taking integer values – 0 or 1.
    
    •    _RMSprop_ optimizer with learning rate of 2e-5.
    
    •    Accuracy is the performance metrics that is tested.
    

_checkpoint_ is initialised to make callbacks. Here the validation loss is being monitored and only the best model is saved after an epoch.


Now the model is fit with 50 epochs and the checkpoint.


Using _pandas.DataFrame_, the training loss, training accuracy, validation loss and validation accuracy are plotted for each epoch.


After clearing the session, the final best_model is loaded using _keras.models.load_model()_.


For performance evaluation, the test data is generated from the rescaled data using _flow_from_directory()_ with a target size of 150x150 and a batch size of 10. 
This is now evaluated using _model.evaluate()_ and it is observed that a **92% accuracy** on the generated test dataset is obtained.
