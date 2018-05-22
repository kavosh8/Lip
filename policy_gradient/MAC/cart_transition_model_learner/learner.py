import csv, numpy
import keras
from keras.constraints import Constraint
from keras import backend as K
from keras.callbacks import ModelCheckpoint

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

print('preparing reps matrix ...')
with open("reps_matrix.csv") as f:
    reader = csv.reader(f)
    #next(reader) # skip header
    reps_matrix = [numpy.array([float(x) for x in r ]) for r in reader]

reps_matrix=numpy.array(reps_matrix)
#print(reps_matrix[0])
#print(reps_matrix.shape)

print('preparing reps_prime matrix ...')
with open("reps_prime_matrix.csv") as f:
    reader = csv.reader(f)
    #next(reader) # skip header
    reps_prime_matrix = [numpy.array([float(x) for x in r ]) for r in reader]

reps_prime_matrix=numpy.array(reps_prime_matrix)
#print(reps_prime_matrix[0])
#print(reps_prime_matrix.shape)

print('preparing actions matrix ...')
with open("actions_matrix.csv") as f:
    reader = csv.reader(f)
    #next(reader) # skip header
    actions_matrix = [numpy.array([float(x) for x in r ]) for r in reader]

actions_matrix=numpy.array(actions_matrix)


num_epochs=10
best_vals=[]

for k in [0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2]:
    each_k=[]
    for run in range(10):
        print('start training for k:',k)
        filepath="cartpole_learned_k"+str(k)+"_"+str(run)+".h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        state_input=keras.layers.Input(shape=(4,))
        action_input=keras.layers.Input(shape=(2,))
        added = keras.layers.Concatenate()([state_input, action_input])
        added=keras.layers.Dense(32,activation='relu',W_constraint = WeightClip(k))(added)
        added=keras.layers.Dense(32,activation='relu',W_constraint = WeightClip(k))(added)
        next_state=keras.layers.Dense(4,W_constraint = WeightClip(k))(added)
        model = keras.models.Model(inputs=[state_input, action_input], outputs=next_state)
        model.compile(optimizer='adam',
                      loss='mse')
        history=model.fit(x=[reps_matrix,actions_matrix], y=reps_prime_matrix, epochs=num_epochs, batch_size=64,validation_split=0.9,callbacks=[checkpoint])
        each_k.append(numpy.min(history.history['val_loss']))
        print(each_k)
    best_vals.append(each_k)
    print(best_vals)

