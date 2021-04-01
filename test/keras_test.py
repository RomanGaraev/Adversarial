import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    model = keras.models.load_model("resnet.h5")
    model.compile()
    #(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = np.load("X.npy")
    y_test = np.load("y.npy")
    #gen = keras.preprocessing.image.ImageDataGenerator()
    #test_iterator = gen.flow(x_test, y_test, batch_size=64)
    #acc = model.evaluate_generator(test_iterator, steps=len(test_iterator), verbose=0)
    #print((acc * 100))

    #x_test = x_test / 255.0
    #scaler = MinMaxScaler()
    #scaler.fit(x_train)
    x_test = [np.transpose(i, axes=(1, 2, 0)) for i in x_test]
    y_test = [np.array([i]) for i in y_test]
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)
    print(model.evaluate(test_dataset))
    pred = np.argmax(model.predict(test_dataset), axis=1)
    #for i in pred:
    #    print(i, sep=" ")
    #print(len([x for i, x in enumerate(pred) if y_test[i] == x]) / len(y_test))
    print(confusion_matrix(y_true=y_test, y_pred=pred))
