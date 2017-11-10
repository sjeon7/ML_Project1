import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
from tqdm import *
import os
slim = tf.contrib.slim

#os.system("python3 features.py --feature mfcc --training True")
#os.system("python3 features.py --feature mfcc --training False")

### properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)
batch_size = 32
epoch_num = 2
learning_rate = 0.0002

# fixed
metadata_path = 'dataset/track_metadata.csv'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = True
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint'
# 'track_genre_top' for project 1, 'listens' for project 2
label_column_name = 'listens'

# Placeholder and variables
# TODO : declare placeholder and variables
#row = n_mfcc
#col = int(valid_features.shape[1]/row)

x = tf.placeholder(tf.float32, [None, 20, 2498, 1], name='inputs')
y = tf.placeholder(tf.float32, [None, 3], name='class')
_is_training = tf.placeholder(tf.bool, name='is_training')

# Build model
# TODO : build your model here
batch_norm_params = {'is_training': _is_training, 'decay':0.9}
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn = tf.nn.relu,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    normalizer_fn = slim.batch_norm,
                    normalizer_params = batch_norm_params):
    net = slim.conv2d(x, 128, [1, 5], padding='same', scope='conv1')
    print('conv1:',net.get_shape())
    net = slim.max_pool2d(net, [1, 10], scope='pool1')
    print('pool1:',net.get_shape())
    net = slim.conv2d(net, 128, [1, 5], padding='same', scope='conv2')
    print('conv2:',net.get_shape())
    net = slim.max_pool2d(net, [1, 10], scope='pool2')
    print('pool2:',net.get_shape())
    net = slim.conv2d(net, 128, [1, 5], padding='same', scope='conv3')
    net = slim.max_pool2d(net, [1, 10], scope='pool3')
    net = slim.conv2d(net, 128, [1, 5], padding='same', scope='conv4')
    net = slim.max_pool2d(net, [1, 10], scope='pool4')
    net = slim.conv2d(net, 64, [1, 5], padding='same', scope='conv5')
    print('conv5:',net.get_shape())
    net = slim.max_pool2d(net, [1, 10], scope='pool5')
    print('true')
    print('pool5:',net.get_shape())
    test=tf.reshape(net,[-1,70,64])
    print(test.get_shape())
    #net = slim.flatten(net, scope='flatten5')
    #print('pool5:',net.get_shape())
    #net = slim.fully_connected(net, 3,
    #                           activation_fn = None,
    #                           weights_initializer = None,
    #                           normalizer_fn = None,
    #                           normalizer_params = None)
    #print('pool5:',net.get_shape())
net = tf.squeeze(net,[1])
#seq_len = np.ones(32)*70
gru = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(64)]*70)
#gru_out, state = tf.nn.dynamic_rnn(gru,tf.reshape(net,[-1,70,64]),sequence_length=seq_len,dtype=tf.float32, scope='gru')
gru_out, state = tf.nn.dynamic_rnn(gru,net,dtype=tf.float32, scope='gru')

gru2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(64)] * 70)
gru2_out, state = tf.nn.dynamic_rnn(gru2, gru_out, dtype=tf.float32, scope='gru2')
gru2_out = tf.transpose(gru2_out, [1, 0, 2])
gru2_out = tf.gather(gru2_out, int(gru2_out.get_shape()[0]) - 1)
dropout_5 = tf.nn.dropout(gru2_out, 0.3)

flat = tf.reshape(dropout_5, [-1, weights.get_shape().as_list()[0]])
#gru =  tf.nn.rnn_cell.BasicLSTMCell(128,forget_bias=1.0)
#gru_out, states = tf.nn.static_rnn(gru,tf.reshape(net,[-1,70,64]),dtype=tf.float32,scope='rnn')
weights = tf.Variable(tf.random_normal([64,3], stddev=0.01))
#output = tf.reshape(gru_out, [-1, weights.get_shape().as_list()[0]])
#print('gru_out',gru_out.get_shape())
#output = tf.reshape(gru_out, [-1, 64])
#print('output',output.get_shape())
#logit_out = tf.matmul(output, weights) + tf.Variable(tf.zeros([3]))
#print('logit_out',logit_out.get_shape())
#logit_out = tf.reshape(logit_out, [batch_size, -1, 3])
#print('logit_out',logit_out.get_shape())
#logit_out = tf.transpose(logit_out, (1,0,2))
#print('logit_out',logit_out.get_shape())
#pred = tf.nn.softmax(logit_out)
pred = tf.nn.softmax(tf.matmul(flat, weights)+tf.Variable(tf.zeros([3])))
print(pred.get_shape())
print(tf.argmax(pred,1))
# Loss and optimizer
# TODO : declare loss and optimizer operation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.contrib.layers.optimize_loss(loss=cost,global_step=tf.contrib.framework.get_global_step(),learning_rate=learning_rate,optimizer='Adam')
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train and evaluate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if is_train_mode:
        train_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size=batch_size,
                                      label_column_name=label_column_name, is_training=True)
        
        feature_train = pd.read_csv('dataset/mfcc_training.csv', header=None)
            
        tbar = tqdm(range(epoch_num))
        for epoch in tbar:
            total_batch = train_dataloader.num_batch
            total_cost = 0
            total_acc = 0

            for i in range(total_batch-1):
                # TODO: load csv file in main.py
                batch_x, batch_y = train_dataloader.next_batch(feature_train)
                batch_x = np.reshape(batch_x, (batch_size, 20, 2498, 1))
                # TODO:  do some train step code here
                feed_dict = {x: batch_x, y: batch_y, _is_training: True}
                _, loss, acc = sess.run([optimizer, cost, accuracy_op], feed_dict = feed_dict)

                total_cost += loss
                total_acc += acc
            desc = 'Epoch: {0:d}, Loss: {1:.3f}, Acc: {2:.3f}'.format(epoch, 
                                                                      total_cost / (total_batch-1), 
                                                                      total_acc / (total_batch-1))
            tbar.set_description(desc)
            train_dataloader.reset_pointer()
#            train_dataloader.shuffle_df()
            
        print('Training finished !')
        output_dir = checkpoint_path + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)
        saver.save(sess, output_dir)
        print('Model saved in file : %s'%output_dir)
    else:
        # skip training and restore graph for validation test
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))


    # Validation
    validation_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size = batch_size, label_column_name = label_column_name, is_training= False)
    
    feature_val = pd.read_csv('dataset/mfcc_validation.csv', header=None)

    average_val_cost = 0
    average_val_acc = 0
    for i in range(validation_dataloader.num_batch-1):
        batch_x, batch_y = validation_dataloader.next_batch(feature_val)
        batch_x = np.reshape(batch_x, (batch_size, 20, 2498, 1))
        
        # TODO : do some loss calculation here
        feed_dict = {x: batch_x, y: batch_y, _is_training: False}
        loss, acc = sess.run([cost, accuracy_op], feed_dict = feed_dict)
        
        average_val_cost += loss
        average_val_acc += acc
    average_val_cost /= (validation_dataloader.num_batch-1)
    average_val_acc /= (validation_dataloader.num_batch-1)
    
    print('Validation loss: {0:.3f}, Acc: {1:.3f}'.format(average_val_cost, average_val_acc))

    # accuracy test example
    # TODO :
    # pred = tf.nn.softmax(<your network output logit object>)
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # avg_accuracy = 0
    #for i in range(validation_dataloader.num_batch):
        #batch_x, batch_y = validation_dataloader.next_batch()
        #acc = accuracy_op.eval({x:batch_x, y: batch_y})
        # avg_accuracy += acc / validation_dataloader.num_batch
    # print("Average accuracy on validation set ", avg_accuracy)
