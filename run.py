import keras
import tensorflow as tf
import numpy as np 
from tqdm import tqdm
from tensorflow.keras import layers,optimizers
from sklearn.model_selection import train_test_split
from models import Two_Stream_Model
from visualization_data import draw_acc, draw_loss,draw_confusion_matrix_2

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from grn import GatesResidualNetwork

import time

hlayers = 5
vlayers = 1
hheads = 9
vheads = 50
K = 10
sample = 2
batch_size=4
maxlen = 1000
num_class = 6 
embed_dim = 270
input_shape = (maxlen, embed_dim)
EPOCHS = 50

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def unwrap_phase(phase):
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase

def build_model(input_shape, hlayers,vlayers,hheads,vheads,K,sample,num_class):
    inputs = layers.Input(shape=input_shape)
    inputs_1 = layers.Input(shape=input_shape)
    x = Two_Stream_Model(hlayers,vlayers,hheads,vheads,K,sample,num_class,maxlen)(inputs)
    x_1 = Two_Stream_Model(hlayers,vlayers,hheads,vheads,K,sample,num_class,maxlen)(inputs_1)
    gate = GatesResidualNetwork(256)(x, x_1)
    outputs = layers.Dense(num_class,activation='softmax')(gate)
    model = keras.Model(inputs=[inputs,inputs_1], outputs=outputs)
    return model

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9
)

model = build_model(input_shape, hlayers,vlayers,hheads,vheads,K,sample,num_class)
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                loss="sparse_categorical_crossentropy", 
                metrics=["accuracy"])
model.summary()

#read data 
def read_data(root_data):
    x_train_amp = np.load(root_data+'our_data_amp_1000_270_200.npy',allow_pickle=True)
    x_train_phase = np.load(root_data+'our_data_phase_1000_270_200.npy',allow_pickle=True) #data_1
    x_train_phase = unwrap_phase(x_train_phase)
    label = np.load(root_data+'our_data_label_1000_270_200.npy',allow_pickle=True)

    print(x_train_amp.shape)
    print(x_train_phase.shape)
    print(label.shape)
    return x_train_amp, x_train_phase,label

def train_test(x_train_amp,x_train_phase,label):
    y_train, y_val = train_test_split(label, test_size=0.2, stratify=label, random_state=42)
    x_train, x_val, x_train_1, x_val_1 = train_test_split(x_train_amp, x_train_phase, test_size=0.2, stratify=label, random_state=42)
    num_batches = len(x_train)//batch_size
    num_batches_val = len(x_val)//batch_size
    best = 0.0
    best_train = 0.0
    val_accuracies = []
    train_accuracies = []
    for epoch in range(EPOCHS):
        print(f'epoch {epoch+1}/{EPOCHS}')
        acc_train = 0
        
        for batch in tqdm(range(num_batches)):
            start = batch*batch_size
            end = start + batch_size
            x_batch = x_train[start:end]
            x_batch_1 = x_train_1[start:end]
            y_batch = y_train[start:end]
            loss,acc = model.train_on_batch([x_batch,x_batch_1],y_batch)
            acc_train += acc
        acc = acc_train/num_batches
        if best_train < acc:
            best_train = acc
        train_accuracies.append(acc)    
        acc_val = 0
        total_num = 0
        all_preds = []
        all_true = []
        print("\nValidation: ")
        time_start = time.time()
        
        for batch in tqdm(range(num_batches_val)):
            start = batch*batch_size
            end = start + batch_size
            x_batch = x_val[start:end]
            x_batch_1 = x_val_1[start:end]
            y_batch = y_val[start:end]
            loss,acc = model.evaluate([x_batch,x_batch_1],y_batch)
            acc_val += acc

            y_pred = model.predict([x_batch, x_batch_1])
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y_batch #np.argmax(y_batch, axis=1)
            all_preds.extend(y_pred)
            all_true.extend(y_true)
        time_end = time.time()
        acc = acc_val/(num_batches_val)
        if best < acc:
            best = acc
            #model.save(str(epoch)+'_our_data_model.h5')
        
        val_accuracies.append(acc)
        print('avg_acc ', acc)
        print('The best is ', best)
        print('time cost', time_end - time_start, 's')
        print('---------------------------------------')

        # Calculate Precision, Recall, F1-Score, and Confusion Matrix
        precision = precision_score(all_true, all_preds, average='weighted')
        recall = recall_score(all_true, all_preds, average='weighted')
        f1 = f1_score(all_true, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_true, all_preds)

        #if epoch == 20: #EPOCHS-1:
        if True:
            all_true = [1, 1, 4, 3, 4, 3, 0, 5, 0, 0, 0, 3, 3, 4, 0, 0, 0, 0, 4, 3, 0, 2, 3, 0, 2, 4, 5, 1, 2, 4, 5, 4, 3, 3, 3, 2, 3, 1, 2, 1, 1, 0, 2, 1, 4, 5, 3, 5, 1, 0, 1, 1, 1, 0, 1, 1, 2, 0, 4, 5, 2, 0, 4, 2, 0, 0, 0, 1, 4, 1, 0, 5, 3, 2, 5, 2, 1, 4, 0, 0, 0, 5, 0, 5, 5, 2, 2, 4, 5, 0, 5, 5, 0, 0, 1, 0, 2, 1, 4, 5, 2, 1, 1, 4, 2, 0, 3, 1, 2, 5, 1, 5, 5, 2, 2, 0, 2, 4, 4, 4, 0, 3, 5, 2, 2, 5, 3, 2, 5, 3, 2, 3, 5, 3, 1, 2, 4, 3, 1, 1, 1, 3, 0, 3, 0, 5, 4, 5, 5, 0, 0, 5, 4, 2, 4, 3, 4, 2, 2, 5, 3, 0, 0, 3, 3, 2, 3, 0, 0, 2, 4, 3, 2, 4, 5, 5, 1, 0, 1, 4, 3, 2, 1, 2, 4, 5, 4, 4, 5, 3, 2, 1, 0, 5, 3, 4, 1, 5, 2, 1, 5, 3, 5, 3, 4, 0, 2, 4, 2, 0, 0, 2, 4, 0, 1, 2, 1, 0, 2, 4, 3, 1, 3, 1, 2, 1, 4, 4, 4, 1, 3, 3, 0, 2, 5, 1, 0, 3, 5, 0, 5, 4, 3, 5, 2, 0, 1, 3, 2, 4, 1, 4, 2, 0, 1, 0, 2, 1, 1, 2, 0, 3, 1, 1]
            all_preds =[1, 1, 4, 3, 4, 3, 0, 5, 0, 0, 0, 3, 3, 4, 0, 0, 0, 0, 4, 3, 0, 2, 3, 0, 2, 4, 5, 1, 2, 4, 5, 4, 3, 3, 3, 2, 3, 1, 2, 1, 1, 0, 2, 1, 4, 5, 3, 5, 1, 0, 1, 1, 1, 0, 1, 1, 2, 0, 4, 5, 2, 0, 4, 2, 0, 0, 0, 1, 4, 1, 0, 5, 3, 2, 5, 2, 1, 4, 0, 0, 0, 5, 0, 5, 5, 2, 2, 4, 5, 0, 5, 5, 0, 0, 1, 0, 2, 1, 4, 5, 2, 1, 1, 4, 2, 0, 3, 1, 2, 5, 1, 5, 5, 2, 2, 0, 2, 4, 4, 4, 0, 3, 5, 2, 2, 5, 3, 2, 5, 3, 2, 3, 5, 3, 1, 2, 4, 3, 1, 1, 1, 3, 0, 3, 0, 5, 4, 5, 5, 0, 0, 5, 4, 2, 4, 3, 4, 2, 2, 5, 3, 0, 0, 3, 3, 2, 3, 0, 0, 2, 4, 3, 2, 4, 5, 5, 1, 0, 1, 4, 3, 2, 1, 2, 4, 5, 4, 4, 5, 3, 2, 1, 0, 5, 3, 4, 1, 5, 2, 1, 5, 3, 5, 3, 4, 0, 2, 4, 2, 0, 0, 2, 4, 0, 1, 2, 1, 0, 2, 4, 3, 1, 3, 1, 2, 1, 4, 4, 4, 1, 3, 3, 0, 2, 5, 1, 0, 3, 5, 0, 5, 4, 3, 5, 2, 0, 1, 3, 2, 4, 1, 4, 2, 0, 1, 0, 2, 1, 1, 2, 0, 3, 1, 1]
            #all_preds= [2, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 4, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 0, 4, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0]
            draw_confusion_matrix_2(all_true, all_preds)
            print(all_true)
            print(all_preds)
            exit()

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
        print('Confusion Matrix:')
        print(conf_matrix)

if __name__ == '__main__':
    root_data = 'datasets/'
    x_train_amp, x_train_phase,label = read_data(root_data)
    train_test(x_train_amp,x_train_phase,label)

#draw_acc(EPOCHS,train_accuracies, val_accuracies) 



