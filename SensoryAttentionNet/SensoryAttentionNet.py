import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
from keras.models import Model
from keras.layers import Input, Dense, Multiply, Dropout, Attention
from keras.losses import mean_squared_error
from keras.utils import plot_model
from keract import get_activations
from keras import backend as K
from sklearn.model_selection import train_test_split

def loadData(path='../Data/OilSourceGeochemicalData.xlsx', sheet_name='oil source correlation'):
    '''
    Load Dataset
    '''
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df[df['BD3']+df['BS12']+df['BS3']+df['QD3']+df['QS12']+df['QS3'] == 1]
    df = df.dropna(axis=1, thresh=0.9*df.shape[0]).reset_index(drop=True)
    x = df.iloc[:, 6:-7]
    df['D3'] = df['BD3'] + df['QD3']
    df['S12'] = df['BS12'] + df['QS12']
    df['S3'] = df['BS3'] + df['QS3']
    y = df.loc[:, ['D3', 'S12', 'S3']]

    print(x.shape, y.shape)
    return x, y

def Att(att_dim, inputs):
    '''
    Attention mechanism
    '''
    V = inputs
    QK = Dense(att_dim, bias_regularizer=None, activation='ReLU')(inputs)
    QK = Attention()([QK, QK])
    MV = Multiply()([V, QK])
    return MV

def build_model():
    '''
    Using the Attention mechanism
    '''
    inputs = Input(shape=(94,),)
    atts1 = Att(94, inputs)
    x1 = Dense(64, activation='sigmoid')(atts1)
    atts2 = Att(64, x1)
    x2 = Dense(32, activation='tanh')(atts2)
    atts3 = Att(32, x2)
    x3 = Dropout(0.5)(atts3)
    output = Dense(3, activation='softmax')(x3)
    model = Model(inputs=inputs, outputs=output)
    return model


def get_activations(model, input_data, print_shape_only=False, layer_name=None):
    '''
    Get the output of the specified layer
    '''
    if layer_name:
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        activations = intermediate_layer_model.weights
    else:
        activations = model.predict(input_data)
    if print_shape_only:
        print(activations.shape)
    else:
        return activations

def splitData(data_list, y_list, ratio=0.30):
    '''
    Split the sample data set according to the specified ratio
    ratio: ratio of test data
    '''
    X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio, random_state=50)
    print('=================split_data shape============================')
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    saveDir='result/'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    # Load data
    X, y = loadData()
    
    # Split data and build model
    X_train, X_test, y_train, y_test=splitData(X.values, y.values, ratio=0.30)
    model = build_model()

    
    # Visualization of model results
    try:
        plot_model(model,to_file=saveDir+"model_structure.png",show_shapes=True)
    except Exception as e:
        print('Exception: ', e)
    model.compile(optimizer='adam', loss=mean_squared_error, metrics=['mse'])
    print(model.summary())
    history=model.fit(X_train, y_train, epochs=65, batch_size=16, validation_split=0.2)
    
    # Loss curve MSE curve visualization
    print(history.history.keys())
    historyResult={}
    plt.clf()
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model MSE Cruve')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.legend(['train','test'], loc='upper left')
    plt.savefig(saveDir+'train_validation_MSE.png')
    
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss Cruve')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(saveDir+'train_validation_loss.png')
    
    plt.clf()
    
    # Get attention mechanism values for the 'attention_vec' layer
    attention_vector = np.array(get_activations(model, X_test, print_shape_only=False, layer_name='attention')[0]) 
    attention_weights = np.abs(attention_vector.sum(axis=1)) / np.sum(np.abs(attention_vector.sum(axis=1)))
    pd.Series(attention_weights).to_csv(saveDir+'attention_weights.csv', index=False, header=False)
    
    # Plot attention weights as a bar chart
    input_dimensions = np.arange(1, 95)
    df = pd.DataFrame({'Input Dimensions': input_dimensions, 'Attention Weight': attention_weights})

    # Increase the size of the figure and adjust x-axis label rotation
    plt.figure(figsize=(20, 6))
    ax = df.plot(x='Input Dimensions', y='Attention Weight', kind='bar', title='Attention Mechanism as a function of input dimensions.')
    plt.xlabel('Input Dimensions')

    # Set the x-axis tick locations to every 10 units
    ax.xaxis.set_major_locator(MultipleLocator(base=2))

    plt.savefig(saveDir+'attention.png')
    
    # Save model
    scores=model.evaluate(X_test,y_test,verbose=0)
    print('MSE: %.2f' % (scores[1]))
    historyResult['mse']=history.history['mse']
    historyResult['val_mse']=history.history['val_mse']
    historyResult['loss']=history.history['loss']
    historyResult['val_loss']=history.history['val_loss']
    model_json=model.to_json()
    with open(saveDir+'structure.json','w') as f:
        f.write(model_json)
    with open(saveDir+'history.json','w') as f:
        f.write(json.dumps(historyResult))
    model.save_weights(saveDir+'weight.h5')