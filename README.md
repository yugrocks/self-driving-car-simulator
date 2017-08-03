# self-driving-car-simulator
The core technology behind Self Driving Cars today. Given the image of a road at a time frame, it can decide where to turn the staring and how much. I am working continuously to generalize it to as many different terrains as possible.    
It uses a Convolutional Neural Network to predict the motion of the steering given the image of a road at a time.  

Requirements: Python 3.5 ,Keras 2.0.2 , Tensorflow 1.2.1 , OpenCV 3.2, numpy 1.11.0   

## However, some cute errors can be clearly seen with the current model in the demo.    

![alt tag](https://raw.githubusercontent.com/yugrocks/self-driving-car-simulator/master/demo2.gif)   

# How to:    
 ### Use:     
     $ python drive.py      
 ### Train your own model:    
     Use the script cnn_train.py . Make sure the datset is ready.
 ### Generate the dataset:
     Use the script generate_data.py to generate the dataset.    
     It requires the path of a video on disk from which training samples will be generated along with the action taken by the user.    
     It automatically puts a frame in the right folder(class) according to actions taken by user while generating data.     

## Contents /Scripts:  
  ### -cnn_train.py :    
        To train the model.    
  ### -generate_data.py :   
        To generate the dataset from random videos.    
  ### - simulator_gui.py :    
        The class that provides the GUI for simulator.    
  ### -drive.py    
        The main script that starts the simulator.    
  ### -model2.json, model4.json :    
        The pre trained models on 4 differnt terrains.
  ### -weights2.hdf5, weights4.hdf5 :    
        Weights of the corresponding models.    

# About The Model:    

![alt tag](https://raw.githubusercontent.com/yugrocks/self-driving-car-simulator/master/model.png)     

Trained using Backpropogation algorithm with stochastic gradint descent.    
## Accuracies after 10 epochs:    
    -Train acc: 96.4665%    
    -Test acc : 88.5039%     
It may seem like it has been overfit. But no. It was the test set, which contained some wrong examples.
