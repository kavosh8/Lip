import numpy as np
import tensorflow as tf
import sys

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class network:
    policy=[]
    weights=[]
    num_hidden_layers=0
    def __init__(self,rep_size,action_size,q_network,num_hidden_layers,hidden_layer_size,learning_rate,sess):
        self.sess=sess
        self.num_hidden_layers=num_hidden_layers
        self.build(rep_size,action_size,q_network,num_hidden_layers,hidden_layer_size,learning_rate)
        
    def build(self,rep_size,action_size,q_network,num_hidden_layers,hidden_layer_size,learning_rate):
        # initialize weights of the network
        for n in range(num_hidden_layers):
            if n==0:
                W = tf.Variable(xavier_init([rep_size, hidden_layer_size]),name="w"+str(n))
            else:
                W = tf.Variable(xavier_init([hidden_layer_size, hidden_layer_size]),name="w"+str(n))
            b = tf.Variable(tf.zeros(shape=[hidden_layer_size]),name="b"+str(n))
            self.weights.append(W)
            self.weights.append(b)
        W=tf.Variable(xavier_init([hidden_layer_size, action_size]),name="w"+str(num_hidden_layers))
        b=tf.Variable(tf.zeros(shape=[action_size]),name="b"+str(num_hidden_layers))
        self.weights.append(W)
        self.weights.append(b)
        # initialize weights of the network
        # define network inputs
        self.state_input=tf.placeholder(tf.float32, shape=[None, rep_size], name='state')
        # define network inputs
        # create input-output mapping
        self.action_weights=self.probs(self.state_input)
        # create input-output mapping
        self.Qs=q_network.Q_values(self.state_input)
        # create the optimizer
        self.v_pi=tf.reduce_sum(tf.multiply(self.Qs, self.action_weights), reduction_indices = 1)

        self.loss= -tf.reduce_mean(self.v_pi)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,var_list=self.weights)
        # create the optimizer
    def probs(self,states):
        assert len(states.get_shape())==2 , "bad state input, perhaps you are doing image stuff?"
        assert states.get_shape()[1]==self.weights[0].get_shape()[0], "bad state input"
        assert len(self.weights)==2*self.num_hidden_layers+2, "model definition does not match what I expect"
        #define hidden layers
        temp=states
        for n in range(self.num_hidden_layers):
            W=self.weights[2*n+0]
            b=self.weights[2*n+1]
            temp=temp=tf.nn.relu(tf.matmul(temp, W) + b)
        #define hidden layers
        #define output layer
        W=self.weights[2*self.num_hidden_layers+0]
        b=self.weights[2*self.num_hidden_layers+1]
        out=tf.nn.softmax(tf.matmul(temp, W) + b)
        #define output layer
        return out
    def action_selection(self,s,sess):
        a_dist = sess.run(self.action_weights,feed_dict={self.state_input:[s]})
        a = np.random.choice(range(len(a_dist[0])),p=a_dist[0])
        return a,a_dist
    def update(self,reps_list_in):
        reps_list=list(reps_list_in)
        #reps_list=reps_list_in
        np.random.shuffle(reps_list)
        reps_list=np.vstack(reps_list)
        _, loss_val,action_weights,Qs,v_pi=self.sess.run([self.train_step,self.loss,self.action_weights,self.Qs,self.v_pi],
                        feed_dict={self.state_input: reps_list})


