from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
#kernel_size=(3,3),stride=1
def cnnmodel(p,input_shape):
    classifier = Sequential()
    classifier.add(Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same', input_shape=input_shape, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(32, kernel_size=(3,3),strides=(1,1), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same', activation='relu')) #increasing filters 32->64
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(64, kernel_size=(3,3),strides=(1,1), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

#flatten
    classifier.add(Flatten())
#full connection
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(p))
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(p/2))
    classifier.add(Dense(1, activation='sigmoid'))
    
    classifier.compile(optimizer='Adam', 
                       loss='binary_crossentropy', matrics=['accuracy'])
    return classifier

def training(batchsize=32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory('dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    classifier = cnnmodel(p=0.2, input_shape=(64,64,3))
    classifier.fit_generator(train_set,
                             steps_per_epoch=8000/batchsize,
                             epochs=15,
                             validation_data=test_set,
                             validation_steps=2000/batchsize)
    #train_set.class_indices
    test_img = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', 
                          target_size=(64,64))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img,axis=0)
    classifier.predict(test_img)


training(batchsize=32, epochs=10)


