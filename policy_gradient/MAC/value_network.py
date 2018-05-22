import numpy as np
import tensorflow as tf
import sys
import random

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def lrelu(x, leak=0.3, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

class network:        
    weights=[]
    nb_epoch=0
    num_hidden_layers=0
    action_size=0
    state_size=0
    batch_size=64
    shuffle='every'
    sess=0
    def __init__(self,rep_size,action_size,num_hidden_layers,hidden_layer_size,learning_rate,nb_epoch,batch_size,shuffle,sess):
        self.sess=sess
        self.num_hidden_layers=num_hidden_layers
        self.batch_size=batch_size
        self.build(rep_size,action_size,num_hidden_layers,hidden_layer_size,learning_rate)
        self.nb_epoch=nb_epoch
        self.action_size
        self.state_size=rep_size
        self.shuffle=shuffle

    def build(self,rep_size,action_size,num_hidden_layers,hidden_layer_size,learning_rate):
        # initialize weights of the network
        for n in range(num_hidden_layers):
            if n==0:
                W = tf.Variable(xavier_init([rep_size, hidden_layer_size]))
            else:
                W = tf.Variable(xavier_init([hidden_layer_size, hidden_layer_size]))
            b = tf.Variable(tf.zeros(shape=[hidden_layer_size]))
            self.weights.append(W)
            self.weights.append(b)
        W=tf.Variable(xavier_init([hidden_layer_size, action_size]))
        b=tf.Variable(tf.zeros(shape=[action_size]))
        self.weights.append(W)
        self.weights.append(b)
        # initialize weights of the network
        # define network inputs
        self.state_input=tf.placeholder(tf.float32, shape=[None, rep_size], name='state')
        self.action_weights=tf.placeholder(tf.float32, shape=[None, action_size], name='action')
        # define network inputs
        # create input-output mapping
        self.Qs=self.Q_values(self.state_input)
        # create input-output mapping
        self.Qe=tf.reduce_sum(tf.multiply(self.Qs, self.action_weights), reduction_indices = 1)

        # create the optimizer
        self.labels = tf.placeholder(shape=[None],dtype=tf.float32)
        self.loss= tf.reduce_mean(tf.square(self.Qe-self.labels))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,var_list=self.weights)
        # create the optimizer
    def Q_values(self,states):

        assert len(states.get_shape())==2, "bad state input, perhaps you are doing image stuff?"
        assert len(self.weights)==2*self.num_hidden_layers+2, "model definition does not match what I expect"
        #define hidden layers
        temp=states
        for n in range(self.num_hidden_layers):
            W=self.weights[2*n+0]
            b=self.weights[2*n+1]
            temp=lrelu(tf.matmul(temp, W) + b)
        #define hidden layers
        #define output layer
        W=self.weights[2*self.num_hidden_layers+0]
        b=self.weights[2*self.num_hidden_layers+1]
        out=tf.matmul(temp, W) + b
        #define output layer
        return out

    def update(self,returns_list_orig,reps_list_orig,actions_list_orig):
        batch_size=self.batch_size
        #print(Qs)
        for epochs in range(self.nb_epoch):
            returns_list,reps_list,actions_list=self.shuffle_3_lists(returns_list_orig,reps_list_orig,actions_list_orig)#shuffle for every epoch

            offset=0
            losses=[]
            while True:
                #print(offset)
                X=np.vstack(reps_list)[offset:min(offset+batch_size,len(reps_list)),:]
                A=np.vstack(actions_list)[offset:min(offset+batch_size,len(reps_list)),:]
                Y=returns_list[offset:min(offset+batch_size,len(returns_list))]
                offset=min(offset+batch_size,len(reps_list))
                _, loss_val,Qs,Qe=self.sess.run([self.train_step,self.loss,self.Qs,self.Qe],
                        feed_dict={self.state_input: X, self.action_weights: A,self.labels: Y})
                if offset==len(reps_list):
                    break
                losses.append(loss_val)

    def shuffle_3_lists(self,returns_list,reps_list,actions_list):
        a=range(len(returns_list))
        random.shuffle(a)
        ret_shuffled,rep_shuffled,act_shuffled=[],[],[]
        for index in a:
            ret_shuffled.append(returns_list[index])
            rep_shuffled.append(reps_list[index])
            act_shuffled.append(actions_list[index])
        return ret_shuffled,rep_shuffled,act_shuffled
