

import tensorflow as tf
from sklearn.preprocessing  import MinMaxScaler
import time
import VGG16_model
import VGG16_train
import os
import numpy as np

data_dir = "D:\data/"
extractor_folder = 'cifar-10-batches-bin'
def encode(y,k):   #y is label k is how many class
    one_hot =  np.zeros((y.shape[0],k))
    for idx, val in enumerate(y):
        one_hot[idx,val] = 1.0
    return one_hot

def load_testdata(): #n=1,2..5,data_batch_1.bin ~data_batch_5.bin
    test_path = os.path.join(data_dir,extractor_folder,'test_batch.bin')
    with open(test_path,'rb') as testpath:
        test_img = np.fromfile(testpath,dtype=np.uint8)
    return test_img

#test
def test(x_test_image,y_test_label):
    with tf.Graph().as_default() as g:


        x_ = tf.placeholder(tf.float32,[None,VGG16_train.n_input])
        x = tf.reshape(x_,shape=[-1,32,32,3])
        y = tf.placeholder(tf.float32,[None,10])

        #construct
        pred ,parameter= VGG16_model.model(x,1)   #dropout = 1


        #evaluate
        correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        test_batch_len = int(x_test_image.shape[0]/200)
        test_acc = []

        test_xs = np.reshape(x_test_image,(x_test_image.shape[0],32,32,3))

        batch_size = 200


        saver=tf.train.Saver()
        # restore all the variable
        with tf.Session() as sess :
            saver.restore(sess,"./VGG16/VGG16_cifar_model")

            for i in range(test_batch_len):
                _acc = sess.run(accuracy,feed_dict={x:test_xs[batch_size*i:batch_size*i+batch_size],y: y_test_label[batch_size*i:batch_size*i+batch_size]})
                test_acc.append(_acc)
                print ("Test batch ",i,":Testing Accuracy:",_acc)

            t_acc = tf.reduce_mean(tf.cast(test_acc,tf.float32))
            print("Average Testing Accuracy=",sess.run(t_acc))
            return
def main(argv=None):
    x_test_images = load_testdata()
    x_test_images = x_test_images.reshape(-1, 3073)  # reshape to (10000,3073)
    tempTest = x_test_images.copy()

    x_test_images = np.delete(x_test_images, 0, 1)
    x_test_images = x_test_images.reshape(-1, 3, 32, 32)
    x_test_images = x_test_images.transpose([0, 2, 3, 1])
    x_test_images = x_test_images.reshape(-1, 3072)

    tempTest = np.hsplit(tempTest, 3073)
    x_test_label = np.asarray(tempTest[0])
    x_test_label = x_test_label.reshape([10000, ])


    # normalization
    mms = MinMaxScaler()
    x_test_images = mms.fit_transform(x_test_images)

    #encode label
    x_test_label = encode(x_test_label, 10)

    print("X_test_image.shape=", x_test_images.shape)
    print("X_test_label.shape=", x_test_label.shape)
    print(x_test_label[0:50])

    test(x_test_images,x_test_label)

if __name__ == '__main__':
	main()