import tensorflow as tf

def batch_norm(inputs,is_training,is_conv_out=True,decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])   #tf.nn.moment :comoute th mean and var of input  ,axes=[0,1,2](in conv means [batch,high,width,depth])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))  #tf.assign : assign a new number to pop_mean
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):                           #tf.control_dependencies(要先執行的operation):
                                                                                                #後來在執行
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, 0.001)

    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, 0.001)


def print_activation(t):
    print(t.op.name,'',t.get_shape().as_list())

def model(image,dropout):
    parameter = []
    #conv1
    with tf.variable_scope("layer1-conv1"):
        filter = tf.Variable(tf.truncated_normal([3,3,3,64],dtype=tf.float32,stddev=0.1),name="weights")
        conv = tf.nn.conv2d(image,filter,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),trainable=True,name="biases")
        bias = tf.add(conv,biases)
        bias = batch_norm(bias,True)
        conv1 = tf.nn.relu(bias)
        print_activation(conv1)
        parameter += [filter,biases]


        filter_b = tf.Variable(tf.truncated_normal([3,3,64,64],dtype=tf.float32,stddev=0.1),name="weights")
        conv_b = tf.nn.conv2d(conv1,filter_b,strides=[1,1,1,1],padding="SAME")
        biases_b = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[64]),trainable=True,name="biases")
        bias_b = tf.add(conv_b,biases_b)
        bias_b = batch_norm(bias_b,True)
        conv1_b =tf.nn.relu(bias_b)
        print_activation(conv1_b)
        parameter +=[filter_b,biases_b]

        #pool1
        pool1 = tf.nn.max_pool(conv1_b,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool1")
        print_activation(pool1)


    #conv2
    with tf.variable_scope("layer2-conv2"):
        filter = tf.Variable(tf.truncated_normal([3,3,64,128],dtype=tf.float32,stddev=0.1),name="weights")
        conv = tf.nn.conv2d(pool1,filter,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[128]),trainable=True,name="biases")
        bias = tf.add(conv,biases)
        bias = batch_norm(bias,True)
        conv2 = tf.nn.relu(bias)
        print_activation(conv2)
        parameter += [filter,biases]

        filter_b = tf.Variable(tf.truncated_normal([3,3,128,128],dtype=tf.float32,stddev=0.1),name="weights")
        conv_b = tf.nn.conv2d(conv2,filter_b,strides=[1,1,1,1],padding="SAME")
        biases_b = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[128]),trainable=True,name="biases")
        bias_b = tf.add(conv_b,biases_b)
        bias_b = batch_norm(bias_b,True)
        conv2_b = tf.nn.relu(bias_b)
        print_activation(conv2_b)
        parameter += [filter_b,biases_b]

        #pool2
        pool2 = tf.nn.max_pool(conv2_b,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool2")
        print_activation(pool2)


    #conv3
    with tf.variable_scope("layer3-conv3"):
        filter = tf.Variable(tf.truncated_normal([3,3,128,256],dtype=tf.float32,stddev=0.1),name="weights")
        conv = tf.nn.conv2d(pool2,filter,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name="biases")
        bias = tf.add(conv,biases)
        bias = batch_norm(bias,True)
        conv3 = tf.nn.relu(bias)
        print_activation(conv3)
        parameter += [filter,biases]


        filter_b = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=0.1),name="weights")
        conv_b = tf.nn.conv2d(conv3,filter_b,strides=[1,1,1,1],padding="SAME")
        biases_b = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name="biases")
        bias_b = tf.add(conv_b,biases_b)
        bias_b = batch_norm(bias_b,True)
        conv3_b = tf.nn.relu(bias_b)
        print_activation(conv3_b)
        parameter += [filter_b,biases_b]


        filter_c = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,stddev=0.1),name="weights")
        conv_c = tf.nn.conv2d(conv3_b,filter_c,strides=[1,1,1,1],padding="SAME")
        biases_c = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name="biases")
        bias_c = tf.add(conv_c,biases_c)
        bias_c = batch_norm(bias_c,True)
        conv3_c = tf.nn.relu(bias_c)
        print_activation(conv3_c)
        parameter += [filter_c,biases_c]

        #pool3
        pool3 = tf.nn.max_pool(conv3_c,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool3")
        print_activation(pool3)

    #conv4
    with tf.variable_scope("layer4-conv4"):
        filter = tf.Variable(tf.truncated_normal([3,3,256,512],dtype=tf.float32,stddev=0.1),name="weights")
        conv = tf.nn.conv2d(pool3,filter,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[512]),trainable=True,name="biases")
        bias = tf.add(conv,biases)
        bias = batch_norm(bias,True)
        conv4 = tf.nn.relu(bias)
        print_activation(conv4)
        parameter += [filter,biases]


        filter_b = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32,stddev=0.1),name="weights")
        conv_b = tf.nn.conv2d(conv4,filter_b,strides=[1,1,1,1],padding="SAME")
        biases_b = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[512]),trainable=True,name="biases")
        bias_b = tf.add(conv_b,biases_b)
        bias_b = batch_norm(bias_b,True)
        conv4_b = tf.nn.relu(bias_b)
        print_activation(conv4_b)
        parameter += [filter_b,biases_b]


        filter_c = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32,stddev=0.1),name="weights")
        conv_c = tf.nn.conv2d(conv4_b,filter_c,strides=[1,1,1,1],padding="SAME")
        biases_c = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[512]),trainable=True,name="biases")
        bias_c = tf.add(conv_c,biases_c)
        bias_c = batch_norm(bias_c,True)
        conv4_c = tf.nn.relu(bias_c)
        print_activation(conv4_c)
        parameter += [filter_c,biases_c]

        ##pool4
        #pool4 = tf.nn.max_pool(conv4_c,ksize=[1,3,3,1],strides=[1,2,2,1],name="pool4")
        #print_activation(pool4)
    #conv5
    with tf.variable_scope("layer5-conv5"):
        filter = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32,stddev=0.1),name="weights")
        conv = tf.nn.conv2d(conv4_c,filter,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[512]),trainable=True,name="biases")
        bias = tf.add(conv,biases)
        bias = batch_norm(bias,True)
        conv5 = tf.nn.relu(bias)
        print_activation(conv5)
        parameter += [filter,biases]


        filter_b = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32,stddev=0.1),name="weights")
        conv_b = tf.nn.conv2d(conv5,filter_b,strides=[1,1,1,1],padding="SAME")
        biases_b = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[512]),trainable=True,name="biases")
        bias_b = tf.add(conv_b,biases_b)
        bias_b = batch_norm(bias_b,True)
        conv5_b = tf.nn.relu(bias_b)
        print_activation(conv5_b)
        parameter += [filter_b,biases_b]


        filter_c = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=tf.float32,stddev=0.1),name="weights")
        conv_c = tf.nn.conv2d(conv5_b,filter_c,strides=[1,1,1,1],padding="SAME")
        biases_c = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[512]),trainable=True,name="biases")
        bias_c = tf.add(conv_c,biases_c)
        bias_c = batch_norm(bias_c,True)
        conv5_c = tf.nn.relu(bias_c)
        print_activation(conv5_c)
        parameter += [filter_c,biases_c]

        #pool5
        pool5 = tf.nn.max_pool(conv5_c,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool5")
        print_activation(pool5)
    #fc1
    with tf.variable_scope("layer6-fc1"):
        fc1_weights = tf.get_variable("weight",shape=[2048,4096],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc = tf.reshape(pool5,[-1,2048])
        fc1_biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),trainable=True,name="biases")
        fc1 = tf.add(tf.matmul(fc,fc1_weights),fc1_biases)
        fc1 = batch_norm(fc1,True,False)
        fc1 = tf.nn.relu(fc1)
        print_activation(fc1)
        parameter += [fc1_weights,fc1_biases]
        fc1 = tf.nn.dropout(fc1,dropout)

    #fc2
    with tf.variable_scope("layer7-fc2"):
        fc2_weights = tf.get_variable("weight",shape=[4096,1000],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[1000]),trainable=True,name="biases")
        fc2 = tf.add(tf.matmul(fc1,fc2_weights),fc2_biases)
        fc2 = batch_norm(fc2,True,False)
        fc2 = tf.nn.relu(fc2)
        print_activation(fc2)
        parameter += [fc2_weights,fc2_biases]
        fc2 =tf.nn.dropout(fc2,dropout)

    #output
    with tf.variable_scope("layer8-output"):
        out_weights = tf.get_variable("weight",shape=[1000,10],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        out_biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[10]),name="biases")
        output = tf.add(tf.matmul(fc2,out_weights),out_biases)
        parameter += [out_weights,out_biases]

    return output,parameter

