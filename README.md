# self-driving-car-simulator
The core technology behind Self Driving Cars today. Given the image of a road at a time frame, it can decide where to turn the steering and how much. I am working continuously to generalize it to as many different terrains as possible.    
It uses a Convolutional Neural Network to predict the motion of the steering given the image of a road at a time.  

Requirements: Python 3.5 ,Keras 2.0.2 , Tensorflow 1.2.1 , OpenCV 3.2, numpy 1.11.0   

## This approach uses Regression for predicting the angle of steering, and is clearly more successful and accurate than the classification approach which I used before. Regression provides flexibility to the results.    
### Regression approach
![alt tag](https://raw.githubusercontent.com/yugrocks/self-driving-car-simulator/master/demo3.gif)
### Classification approach
![alt tag](https://raw.githubusercontent.com/yugrocks/self-driving-car-simulator/master/demo2.gif)   

# How to:    
 ### Use:     
     $ python drive.py   
     You can use it either on a live video feed from the webcam, or a pre saved video on disk for the demo.    
 ### Train your own model:    
     Use the script cnn_train.py or train_cnn2.py(branch 2 named Regression-Approach). Make sure the datset is ready.
 ### Generate the dataset:
     Use the script generate_data.py to generate the dataset.    
     It requires the path of a video on disk from which training samples will be generated along with the action taken by the user.    
     It automatically puts a frame in the right folder(class) according to actions taken by user while generating data.     

## Contents /Scripts:  
  ### -cnn_train.py :    
        To train the  model.    
  ### -train_cnn2.py:    
        This file is in branch named "Regression-Approach". This is to train the regressive model.    
  ### -generate_data.py :   
        To generate the dataset from random videos.    
  ### - simulator_gui.py :    
        The class that provides the GUI for simulator.    
  ### -drive.py    
        The main script that starts the simulator.    
  ### -model2.json, model4.json :    
        The pre trained models on 4 differnt terrains. Note that the model2.json is different in both the branches.    
  ### -weights2.hdf5, weights4.hdf5 :    
        Weights of the corresponding models.    

# About The Model:    
The classification based model:    
![alt tag](https://raw.githubusercontent.com/yugrocks/self-driving-car-simulator/master/model.png)    
The regression based model:    
![alt tag](https://raw.githubusercontent.com/yugrocks/self-driving-car-simulator/master/model_regression.png)     

Trained using Backpropogation algorithm with stochastic gradint descent.    
## Accuracies after 10 epochs:  
### For classification based model:    
    -Train acc: 96.4665%    
    -Test acc : 88.5039%     
It may seem like it has been overfit. But no. It was the test set, which contained some wrong examples.    
### For regression based model:    
    -Train error: 2.0311  (Mean absolute error)     
    -Test error:  2.4532   
