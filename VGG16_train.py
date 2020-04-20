
import  tensorflow as tf
import os
import struct
import numpy as np
import  cv2,csv
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import VGG16_model



data_dir = "D:\data/"
extractor_folder = 'cifar-10-batches-bin'
def encode_labels(y,k):   #y is label k is how many class
    one_hot =  np.zeros((y.shape[0],k))
    for idx, val in enumerate(y):
        one_hot[idx,val] = 1.0
    return one_hot
def load_train_data(n): #n=1,2..5,data_batch_1.bin ~data_batch_5.bin
    image_path = os.path.join(data_dir,extractor_folder,'data_batch_{}.bin'.format(n))
    with open(image_path,'rb') as imgpath:
        images = np.fromfile(imgpath,dtype=np.uint8)
    return images
def load_test_data(): #n=1,2..5,data_batch_1.bin ~data_batch_5.bin
    test_path = os.path.join(data_dir,extractor_folder,'test_batch.bin')
    with open(test_path,'rb') as testpath:
        test_img = np.fromfile(testpath,dtype=np.uint8)
    return test_img

#parameter
MODEL_SAVE_PATH ='./VGG16/'
MODEL_NAME = "VGG16_cifar_model"
learning_rate = 0.001
BATCH_SIZE = 200
display_step = 100
TRAIN_STEP = 8000
#network parameter
n_input = 3072 #cifar size
n_class = 10   #classes
dropout = 0.5

def train(x_train,y_train_label):
    shuffle = True
    batch_idx = 0
    batch_len = int(x_train.shape[0]/BATCH_SIZE)
    train_accuracy = []
    train_idx = np.random.permutation(batch_len) #打散batch 500 group e.g[2,50,36,152,....]

    # tf image input
    x_ = tf.placeholder(tf.float32,shape=[None,n_input])
    y_ = tf.placeholder(tf.float32,shape=[None,n_class])
    keep_prob = tf.placeholder(tf.float32)
    x = tf.reshape(x_,shape=[-1 ,32 ,32,3 ])

    #construct model
    _train ,parameter = VGG16_model.model(x,keep_prob)


    #define optimizer loss function cross entropy +l2 loss?


    #weight = tf.reshape(parameter[0],shape=[1,-1])
    #cost1 =tf.add_to_collection('losses',tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_train,labels=y_)))
    #cost2 =tf.add_to_collection('losses',tf.multiply(tf.nn.l2_loss(t=weight,name=None),0.004))
    #total_loss = tf.add_n(tf.get_collection('losses'))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_train,labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #evaluate model
    correct_pred = tf.equal(tf.arg_max(_train,1),tf.arg_max(y_,1)) #找出predict 出來機率最大和label相比 bool
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))#tf.cast  cast tensor to a new type

    #save model
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    #start
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        print("Starting training")
        #keep training until reach max iteration
        while step <TRAIN_STEP:
            if shuffle == True:    #洗牌ㄇ 打散batch順序
                batch_shuffle_idx = train_idx[batch_idx]  #train_idx 打散500group 從其第一個開始
                batch_xs = x_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]   #from 0*200:0*200+200 (first batch)
                batch_ys = y_train_label[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
            else:
                batch_xs = x_train[batch_idx * BATCH_SIZE:batch_idx * BATCH_SIZE + BATCH_SIZE]  # from 0*200:0*200+200 (first batch)
                batch_ys = y_train_label[batch_idx * BATCH_SIZE:batch_idx * BATCH_SIZE + BATCH_SIZE]
            if batch_idx < batch_len:
                batch_idx += 1
                if batch_idx == batch_len:
                        batch_idx = 0
            else:
                    batch_idx = 0
            reshaped_xs = np.reshape(batch_xs,(BATCH_SIZE,32,32,3))
            #optimize(backprop)
            sess.run(optimizer, feed_dict={x: reshaped_xs, y_: batch_ys,keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: reshaped_xs, y_: batch_ys,keep_prob: 1.})
            train_accuracy.append(acc)
            if step % display_step == 0:
                print("Step:"+str(step)+",Mini_batch Loss="+"{:.6f}".format(loss)+",Training ACC="+"{:.5f}".format(acc))
            step +=1
        print("Optimization Finished!")
        print("Save model...")
        # saver.save(sess, "./alexnet/alexnet_model")
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
def main(argv=None):

    ##Load Cifar-10 train image and label
    X_train_image1 = load_train_data(1)  # load data_batch_1.bin
    X_train_image2 = load_train_data(2)  # load data_batch_2.bin
    X_train_image3 = load_train_data(3)  # load data_batch_3.bin
    X_train_image4 = load_train_data(4)  # load data_batch_4.bin
    X_train_image5 = load_train_data(5)  # load data_batch_5.bin
    print(X_train_image1.shape)

    X_train_image = np.concatenate((X_train_image1, X_train_image2, X_train_image3, X_train_image4, X_train_image5),
                                   axis=0)
    print(X_train_image.shape)

    # reshape to (50000,3073)
    # in one Row ,the 1st byte is the label,other 3072byte =1024 Red +1024 green +1024 blue ch data
    X_train_image = X_train_image.reshape(-1, 3073)
    tempA = X_train_image.copy()
    X_train_image = np.delete(X_train_image, 0, 1)  # delete 1st column data
    X_train_image = X_train_image.reshape(-1, 3, 32, 32)  # (50000,3,32,32)
    X_train_image = X_train_image.transpose([0, 2, 3, 1])  # transfer to (10000,32,32,3)
    X_train_image = X_train_image.reshape(-1, 3072)  # (50000,3,32,32)

    # split to 3073 col,the first column is the label.
    tempA = np.hsplit(tempA, 3073)
    X_train_label = np.asarray(tempA[0])
    X_train_label = X_train_label.reshape([50000, ])

    print(X_train_image.shape)
    print(X_train_label.shape)
    print(X_train_label[0:50])

    mms = MinMaxScaler()
    X_train_image = mms.fit_transform(X_train_image)

    X_train_label = encode_labels(X_train_label, 10)
    print("y_train_lable.shape=", X_train_label.shape)

    train(X_train_image, X_train_label)

if __name__ == '__main__':
    main()

