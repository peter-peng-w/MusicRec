#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:51:44 2020

@author: apple
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

music = pd.read_json('/Users/apple/Desktop/Digital_Music_5.json', lines = True)
music = music[['overall', 'reviewerID', 'asin', 'style']]

#groupby music 
rating_count = (music.
     groupby(by = ['asin'])['overall'].
     count().
     reset_index().
     rename(columns = {'overall': 'RatingCount_music'})
     [['asin', 'RatingCount_music']]
    )

threshold = 10
rating_count = rating_count.query('RatingCount_music >= @threshold')

#join回原表
rating_count_10 = pd.merge(rating_count, music , left_on='asin', right_on='asin', how='left')

#groupby reviewer 
user_count = (rating_count_10.
     groupby(by = ['reviewerID'])['overall'].
     count().
     reset_index().
     rename(columns = {'overall': 'RatingCount_reviewer'})
     [['reviewerID', 'RatingCount_reviewer']]
    )

user_count = user_count.query('RatingCount_reviewer >= @threshold')
#join 上表
data = pd.merge(user_count,rating_count_10 , left_on='reviewerID', right_on='reviewerID', how='left')

print('Number of unique music: ', data['asin'].nunique())
print('Number of unique reviewers: ', data['reviewerID'].nunique())






scaler = MinMaxScaler()
data['overall'] = data['overall'].values.astype(float)
rating_scaled = pd.DataFrame(scaler.fit_transform(data['overall'].values.reshape(-1,1)))
data['overall'] = rating_scaled


data = data.drop_duplicates(['reviewerID', 'asin'])

#构建评分矩阵
reviewer_music_matrix = data.pivot(index='reviewerID', columns='asin', values='overall')
reviewer_music_matrix.fillna(0, inplace=True)

reviewers = reviewer_music_matrix.index.tolist()
musics = reviewer_music_matrix.columns.tolist()

#turn df into array 纯数字
reviewer_music_matrix = reviewer_music_matrix.iloc[:,:].values


#train一个tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



num_input = data['asin'].nunique()
num_hidden_1 = 10
num_hidden_2 = 5


X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}


#build the encoder and decoder model
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2





encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
y_pred = decoder_op
y_true = X


loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)


init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()



#开始train
with tf.Session() as session:
    epochs = 100      
    batch_size = 35    

    session.run(init)
    session.run(local_init)

    num_batches = int(reviewer_music_matrix.shape[0] / batch_size) 
    reviewer_music_matrix = np.array_split(reviewer_music_matrix, num_batches) 
    
    for i in range(epochs):

        avg_cost = 0
        for batch in reviewer_music_matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches

        print("epoch: {} Loss: {}".format(i + 1, avg_cost))

    reviewer_music_matrix = np.concatenate(reviewer_music_matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={X: reviewer_music_matrix})

    pred_data = pred_data.append(pd.DataFrame(preds))

    pred_data = pred_data.stack().reset_index(name='overall')
    pred_data.columns = ['reviewerID', 'asin', 'overall']
    pred_data['reviewerID'] = pred_data['reviewerID'].map(lambda value: reviewers[value])
    pred_data['asin'] = pred_data['asin'].map(lambda value: musics[value])
    
    keys = ['reviewerID', 'asin']
    index_1 = pred_data.set_index(keys).index
    index_2 = data.set_index(keys).index

    top_ten_ranked = pred_data[~index_1.isin(index_2)]
    top_ten_ranked = top_ten_ranked.sort_values(['reviewerID', 'overall'], ascending=[True, False])
    top_ten_ranked = top_ten_ranked.groupby('reviewerID').head(10)
    
    

top_ten_ranked.loc[top_ten_ranked['reviewerID'] == 'AZXWUZ9PPSOTL']














