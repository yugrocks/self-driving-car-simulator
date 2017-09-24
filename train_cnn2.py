import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras import regularizers
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.preprocessing.image import load_img

labels = {'a':0, 'b':15, 'c':30, 'd':45,'e':60, 'f':75, 'g':90} #defining the values
train_filenames = []
train_values = []
test_filenames = []
test_values = []
train_dir = "Dataset/training_set"
test_dir = "Dataset/test_set"
batch_size = 32
nb_epochs = 10
img_rows=33
img_colms=50
img_channels=1 #1 for grayscale and 3 for RGB images
X_test = []
Y_test = []
X_train = []
Y_train = []


def preprocess_img(img):
    img = cv2.resize(img, (img_colms,img_rows), interpolation = cv2.INTER_AREA)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.
    return img


def load_data(train_dir, test_dir):
    global train_filenames, train_values, test_filenames, test_values
    for _ in os.listdir(train_dir):
        #for each directory in train_dir
        path = os.path.join(train_dir, _)
        for filename in os.listdir(path):
            #for each file in _
            train_filenames.append(os.path.join(path, filename))
            train_values.append(labels[_])
    randomize = np.arange(len(train_filenames))
    np.random.shuffle(randomize)
    train_filenames = np.array(train_filenames)[randomize]
    train_values = np.array(train_values)[randomize]
    #now test data
    for _ in os.listdir(test_dir):
        #for each directory in train_dir
        path = os.path.join(test_dir, _)
        for filename in os.listdir(path):
            #for each file in _
            test_filenames.append(os.path.join(path, filename))
            test_values.append(labels[_])
    randomize = np.arange(len(test_filenames))
    np.random.shuffle(randomize)
    test_filenames = np.array(test_filenames)[randomize]
    test_values = np.array(test_values)[randomize]

def get_test_data():
    global X_test, Y_test
    for i in range(len(test_filenames)):
        img = preprocess_img(cv2.imread(test_filenames[i]))
        X_test.append(img.flatten())
        Y_test.append(test_values[i])
    X_test = np.array(X_test)
    Y_test =  np.array(Y_test)
    
def get_train_data():
    global X_train, Y_train
    for i in range(len(train_filenames)):
        img = preprocess_img(cv2.imread(train_filenames[i]))
        X_train.append(img.flatten())
        Y_train.append(train_values[i])
    X_train = np.array(X_train)
    Y_train =  np.array(Y_train)
        
        

def generate_next_batch(current_index, upper_bound):
    X_train = []
    Y_train = []
    for i in range(current_index, upper_bound):
        img = preprocess_img(cv2.imread(train_filenames[i]))
        X_train.append(img.flatten())
        Y_train.append(train_values[i])
    return np.array(X_train), np.array(Y_train)


# For manual training per batch
def train_model(model, nb_epochs, batch_size = 32):
    #take one batch from the data, and train on it for nb_epochs
    global X_test
    for i in range(nb_epochs):
        print("Epoch {}/{}".format(i+1, nb_epochs))
        nb_batches = len(train_filenames) // batch_size #number of full batches
        last_batch_size = len(train_filenames) - (nb_batches * batch_size)
        #loop through all full batches
        for batch_id in range(0, nb_batches):
            X_train, Y_train = generate_next_batch(batch_id*batch_size, (batch_id+1)*batch_size)
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_colms, img_channels)
            error = model.train_on_batch(X_train, Y_train)
            print("train errror = ", error)
        #take care of last batch
        X_train, Y_train = generate_next_batch(len(train_filenames)-last_batch_size,len(train_filenames))
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_colms, img_channels)
        error = model.train_on_batch(X_train, Y_train)
        X_train = None; Y_train = None
        print("train error = ", error)
        #calculate test error now
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_colms, img_channels)/255.0
        test_error = model.test_on_batch(X_test, Y_test)
        print("validation error = ", test_error)
        return model


def get_model():
    #init the model
    model= Sequential()
    
    #add conv layers and pooling layers (2 of each)
    model.add(Convolution2D(32,3,3, input_shape=(img_rows, img_colms, img_channels),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #there's a lot of data so no considerable overfitting is expected
    model.add(Flatten())
    #Now one hidden(dense) layer:
    model.add(Dense(output_dim = 500, activation = 'relu',
                    #kernel_regularizer=regularizers.l2(0.01)
                   ))    
    #model.add(Dropout(0.01))#again for regularization
    #output layer
    model.add(Dense(output_dim = 1)) #outputs real values
    
    lr = .001
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) #custom learning rate, with rate decay enabled
    #Now copile it
    model.compile(optimizer="adam", loss='mean_absolute_error', metrics=['mae'])
    model.summary() #get summary
    return model
    

model = get_model()
load_data(train_dir, test_dir)
get_train_data()
get_test_data()
#model = train_model(model, nb_epochs = 1, batch_size = 256)



X_train = X_train.reshape(X_train.shape[0], img_rows, img_colms, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_colms, img_channels)

model.fit(X_train, Y_train, initial_epoch=28 ,batch_size=256, nb_epoch=30, verbose=1, validation_data=(X_test,Y_test))


model.save_weights("weights2.hdf5",overwrite=True)
#saving the model itself in json format:
model_json = model.to_json()
with open("model2.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")

#check the model on a random image in test set
img = load_img('Dataset\\training_set\\b\\1.jpg',target_size=(33,50))
x = np.array(img)
img = cv2.cvtColor( x, cv2.COLOR_BGR2GRAY )
img = img.reshape((1,)+img.shape+(1,))
img = img/255.
y_pred = model.predict(img)
print(y_pred)
