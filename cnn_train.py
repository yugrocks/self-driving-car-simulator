from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from numpy import array
from keras import regularizers
import cv2
from skimage import color, exposure
from keras.optimizers import SGD
from keras.utils import plot_model

img_rows=33
img_colms=50
img_channels=1 #1 for grayscale and 3 for RGB images


def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    img=np.array(img)
    img = cv2.resize(img, (img_colms,img_rows), interpolation = cv2.INTER_AREA)
    return img
        

def get_model():
    #init the model
    model= Sequential()
    
    #add conv layers and pooling layers (2 of each)
    model.add(Convolution2D(32,3,3, input_shape=(img_rows, img_colms, img_channels),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(32,3,3, input_shape=(img_rows, img_colms, img_channels),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    model.add(Dropout(0.5)) #to reduce overfitting
    
    model.add(Flatten())
    
    #Now two hidden(dense) layers:
    model.add(Dense(output_dim = 256, activation = 'relu',
                    kernel_regularizer=regularizers.l2(0.02)
                   ))
    
    model.add(Dropout(0.5))#again for regularization
    
    model.add(Dense(output_dim = 128, activation = 'relu',
                    kernel_regularizer=regularizers.l2(0.02)
                    ))
    
    
    model.add(Dropout(0.5))#last one lol
    
    """model.add(Dense(output_dim = 500, activation = 'relu',
                    kernel_regularizer=regularizers.l2(0.02)
                    ))
    
    model.add(Dense(output_dim = 128, activation = 'relu',
                    kernel_regularizer=regularizers.l2(0.01)
                    ))
    model.add(Dropout(0.5))"""
    
    #output layer
    model.add(Dense(output_dim = 7, activation = 'softmax'))
    
    lr = 0.001
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) #custom learning rate, with rate decay enabled
    #Now copile it
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary() #get summary
    return model
    

def init_dataset():
    #Prepare the images from the dataset
    train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False
                                 )

    test_datagen=ImageDataGenerator(rescale=1./255,
                                    horizontal_flip = False)
    
    training_set=train_datagen.flow_from_directory("Dataset/training_set",
                                                   target_size = (img_rows, img_colms),
                                                   color_mode='grayscale',
                                                   batch_size=10,
                                                   class_mode='categorical')
    
    test_set=test_datagen.flow_from_directory("Dataset/test_set",
                                                   target_size = (img_rows, img_colms),
                                                   color_mode='grayscale',
                                                   batch_size=10,
                                                   class_mode='categorical')
    
    return training_set, test_set


def train_CNN():
    #get model:
    model=get_model()
    
    #get datasets:
    training_set, test_set = init_dataset()
    
    #start training:
    history = model.fit_generator(training_set,
                         samples_per_epoch = 18676,
                         nb_epoch = 13,
                         validation_data = test_set,
                         nb_val_samples =4652)

    
    return history, model
    
    
input("Press enter to start training the model. Make sure the dataset is ready, and all files and folders are in place.")
history, model = train_CNN()
    
#accuracies over 13 epochs:
    #train acc: 75.2443%
    #test acc : 84.5039%

dec = str(input("Save weights, y/n?"))

if dec.lower() == 'y':
    #saving the weights
    model.save_weights("weights2.hdf5",overwrite=True)
    #saving the model itself in json format:
    model_json = model.to_json()
    with open("model2.json", "w") as model_file:
        model_file.write(model_json)
    print("Model has been saved.")
    

#check the model on a random image in test set
img = load_img('Dataset\\test_set\\e\\2.jpg',target_size=(33,50))
x=array(img)
img = cv2.cvtColor( x, cv2.COLOR_RGB2GRAY )
img=img.reshape((1,)+img.shape)
img=img.reshape(img.shape+(1,))
test_datagen = ImageDataGenerator(rescale=1./255)
m=test_datagen.flow(img,batch_size=1)
y_pred=model.predict_generator(m,1)

#save the model schema
plot_model(model, to_file='model.png', show_shapes = True)

