import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
from tqdm import tqdm
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
### properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)
batch_size = 32
epoch_num = 100


# fixed
metadata_path = 'dataset/track_metadata.csv'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
#is_train_mode = False
is_train_mode = True
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint'
# 'track_genre_top' for project 1, 'listens' for project 2
label_column_name = 'listens'


# Placeholder and variables
x = tf.placeholder(tf.float32, [None, 19, 20, 250], name = 'input_spectrogram')
y = tf.placeholder(tf.float32, [None, 3], name = 'label')
is_train = tf.placeholder(tf.bool, name='is_training') # for batch normalization

x_reshaped = tf.reshape(x, [-1, 20, 250])
x_expand = tf.expand_dims(x_reshaped, -1) 
# Build model
batch_norm_params = {'is_training':is_train, 'decay':0.9, 'updates_collections': None}

with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer(), normalizer_fn = slim.batch_norm, normalizer_params = batch_norm_params):
    net = slim.conv2d(x_expand, 32, kernel_size = [1,3], padding = 'SAME', scope = 'conv1')
    net = slim.max_pool2d(net, [1,3], scope = 'pool1')

    net = slim.conv2d(net, 32, kernel_size = [1,3], padding='same' , scope = 'conv2')
    net = slim.max_pool2d(net, [1,3], scope = 'pool2')

    net = slim.conv2d(net, 1, kernel_size = [1,1], padding = 'same', scope = 'conv3', activation_fn = None)
    net = slim.flatten(net, scope = 'flatten5')

#    net = slim.fully_connected(net, 3, activation_fn = None, weights_initializer = None, normalizer_fn = None, normalizer_params = None)
net = tf.reshape(net, [-1, 19, 305])

with tf.variable_scope('RNN'):
    rnn_input = tf.unstack(net, 19, axis=1)
    gru_cell = rnn.GRUCell(64)
    outputs, _ = rnn.static_rnn(gru_cell, rnn_input, dtype = tf.float32, scope = 'RNN')
    outputs = tf.transpose(tf.stack(outputs), [1,0,2]) # batch_size, 19, 64

net = slim.flatten(outputs, scope = 'flatten_rnn')
net = slim.fully_connected(net, 60, scope = 'fc1')
net = slim.fully_connected(net, 3, scope = 'fc2')
    

# Loss and optimizer
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = net))
loss = tf.reduce_mean(tf.losses.huber_loss(labels = y, predictions = tf.nn.softmax(net))) 
opt = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct = tf.equal(tf.argmax(y,1), tf.argmax(net,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train and evaluate
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep = 2)

    if is_train_mode:
        train_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size=batch_size,
                                      label_column_name=label_column_name, is_training=True)
        validation_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size = batch_size, label_column_name = label_column_name, is_training= False)
        
        max_acc = 0.0
        
        total_batch = train_dataloader.num_batch
        for epoch in range(epoch_num):
            total_loss, total_acc = 0.0, 0.0
            for i in tqdm(range(total_batch)):
                batch_x, batch_y = train_dataloader.next_batch()      
                feed_dict = {x : batch_x, y: batch_y, is_train : True}
                _, l, acc = sess.run([opt, loss, accuracy], feed_dict = feed_dict)
                total_loss += l
                total_acc += acc
            total_loss /= total_batch
            total_acc /= total_batch
            print("epoch {0} - loss : {1}, acc : {2}".format(epoch, total_loss, total_acc)) 
            
            if epoch % 5 == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(validation_dataloader.num_batch):
                    val_x, val_y = validation_dataloader.next_batch()
                    feed_dict = {x : val_x, y: val_y, is_train : True}
                    v_l, v_acc = sess.run([loss, accuracy], feed_dict = feed_dict)
                    val_loss += v_l
                    val_acc += v_acc
                val_loss /= validation_dataloader.num_batch
                val_acc /= validation_dataloader.num_batch
                print("validation - loss : {0}, acc : {1}".format(val_loss, val_acc)) 
                
                if val_acc > max_acc:
                    output_dir = checkpoint_path + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
                    if not gfile.Exists(output_dir):
                        gfile.MakeDirs(output_dir)
                    saver.save(sess, output_dir)
                    max_acc = val_acc
        print('Training finished !')
    else:
        # skip training and restore graph for validation test
        saver.restore(sess, checkpoint_path)


    # Validation
    validation_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size = batch_size, label_column_name = label_column_name, is_training= False)

    average_val_cost = 0.0
    average_val_acc = 0.0
    for i in range(validation_dataloader.num_batch):
        batch_x, batch_y = validation_dataloader.next_batch()
        feed_dict = {x:batch_x, y:batch_y, is_train : False}
        test_l, test_acc = sess.run([loss, accuracy], feed_dict = feed_dict)
        average_val_cost += test_l
        average_val_acc += test_acc
    
    average_val_cost /= validation_dataloader.num_batch
    average_val_acc /= validation_dataloader.num_batch

    print('Validation loss : {0} acc : {1}'.format(average_val_cost, average_val_acc))
