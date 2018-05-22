import numpy 
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import optimizers
from keras.constraints import Constraint
from keras import backend as K
import tensorflow as tf


class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


class neural_transition_model:
    models=[]
    def __init__(self,model_params):
        self.K=model_params['lipschitz_constant']
        self.num_hidden_layers=model_params['num_hidden_layers']
        self.hidden_layer_nodes=model_params['hidden_layer_nodes']
        self.activation_fn=model_params['activation_fn']
        self.learning_rate=model_params['learning_rate']
        self.observation_size=model_params['observation_size']
        self.num_models=model_params['num_models']
        self.num_epochs=model_params['num_epochs']
        for n in range(self.num_models):
            self.models.append(self.create_model())

    def create_model(self):
    	input_state = keras.layers.Input(shape=(self.observation_size,))
        h=input_state
        for l in range(self.num_hidden_layers):
            h=Dense(self.hidden_layer_nodes
                    ,activation=self.activation_fn
                    ,kernel_constraint = WeightClip(self.K))(h)

    	output_state=Dense(self.observation_size)(h)
        model=keras.models.Model(inputs=input_state, outputs=output_state)
    	ad=optimizers.Adam(lr=self.learning_rate)
    	model.compile(loss='mean_squared_error',optimizer=ad)
        return model
    def regression(self,phi,y,w_li):
        assert len(w_li)==self.num_models, "number of weights not equal to number of functions {} {}".format(len(w_li),self.num_models)
        for index, w in enumerate(w_li):
            self.models[index].fit(x=phi,y=y,epochs=self.num_epochs, verbose=0,sample_weight=w)
    def predict(self,phi):
        return [x.predict(phi) for x in self.models]