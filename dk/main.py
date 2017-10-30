import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time
import vgg
from tqdm import *
import os
slim = tf.contrib.slim

os.system("python3 feature.py --feature mfcc --training True")
os.system("python3 feature.py --feature mfcc --training False")

### properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)
batch_size = 32
epoch_num = 100
learning_rate = 0.0002

# fixed
metadata_path = 'dataset/track_metadata.csv'
# True if you want to train, False if you already trained your model
### TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = False
### TODO : IMPORTANT !!! Please specify the path where your best model is saved
### example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint'
# 'track_genre_top' for project 1, 'listens' for project 2
label_column_name = 'track_genre_top'

# Placeholder and variables
# TODO : declare placeholder and variables
#row = n_mfcc
#col = int(valid_features.shape[1]/row)

x = tf.placeholder(tf.float32, [None, 20, 2498, 1], name='inputs')
y = tf.placeholder(tf.float32, [None, 8], name='class')
_is_training = tf.placeholder(tf.bool, name='is_training')

# Build model
# TODO : build your model here
batch_norm_params = {'is_training': _is_training, 'decay':0.9, 'updates_collections':None}
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn = tf.nn.relu,
                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                    normalizer_fn = slim.batch_norm,
                    normalizer_params = batch_norm_params):
    net = slim.conv2d(x, 64, [3, 3], padding='same', scope='conv1')
    net = slim.max_pool2d(net, [2, 4], scope='pool1')
    net = slim.conv2d(net, 128, [3, 3], padding='same', scope='conv2')
    net = slim.max_pool2d(net, [2, 4], scope='pool2')
    net = slim.conv2d(net, 128, [3, 3], padding='same', scope='conv3')
    net = slim.max_pool2d(net, [2, 4], scope='pool3')
    net = slim.conv2d(net, 128, [3, 3], padding='same', scope='conv4')
    net = slim.max_pool2d(net, [2, 4], scope='pool4')
    net = slim.conv2d(net, 64, [3, 3], padding='same', scope='conv5')
#    net = slim.max_pool2d(net, [4, 4], scope='pool5')
    net = slim.flatten(net, scope='flatten5')
    net = slim.fully_connected(net, 8,
                               activation_fn = None,
                               weights_initializer = None,
                               normalizer_fn = None,
                               normalizer_params = None)
pred = tf.nn.softmax(net)

# Loss and optimizer
# TODO : declare loss and optimizer operation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
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