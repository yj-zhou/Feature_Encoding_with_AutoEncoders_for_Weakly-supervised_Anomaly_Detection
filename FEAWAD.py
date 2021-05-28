# -*- coding: utf-8 -*- 
"""
@author：Xucheng Song
The algorithm was implemented using Python 3.6.12, Keras 2.3.1 and TensorFlow 1.13.1 based on the code (https://github.com/GuansongPang/deviation-network).
The major contributions are summarized as follows. 

This code adds a feature encoder to encode the input data and utilizes three factors, hidden representation, reconstruction residual vector,
and reconstruction error, as the new representation for the input data. The representation is then fed into an MLP based anomaly score generator,
similar to the code (https://github.com/GuansongPang/deviation-network), but with a twist, i.e., the reconstruction error is fed to each layer
of the MLP in the anomaly score generator. A different loss function in the anomaly score generator is also included. Additionally,
the pre-training procedure is adopted in the training process. More details can be found in our TNNLS paper as follows.

Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu,
Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection,
in IEEE Transactions on Neural Networks and Learning Systems, 2021, 12 pages,
which can be found in IEEE Xplore or arxiv (https://arxiv.org/abs/2105.10500).
"""

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Subtract,concatenate,Lambda,Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error

import argparse
import numpy as np
import sys
from scipy.sparse import vstack, csc_matrix
from toolsdev import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split

MAX_INT = np.iinfo(np.int32).max
data_format = 0


def auto_encoder(input_shape):
    x_input = Input(shape=input_shape)
    length = K.int_shape(x_input)[1]
  
    input_vector = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ain')(x_input)     
    en1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ae1')(input_vector)   
    en2 = Dense(64,kernel_initializer='glorot_normal', use_bias=True,activation='relu',name = 'ae2')(en1)
    de1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad1')(en2)
    de2 = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad2')(de1)

    model =  Model(x_input, de2)
    adm = Adam(lr=0.0001)
    model.compile(loss=mean_squared_error, optimizer=adm)
    return model

def dev_network_d(input_shape,modelname,testflag):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    length = K.int_shape(x_input)[1]
  
    input_vector = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ain')(x_input)     
    en1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ae1')(input_vector)   
    en2 = Dense(64,kernel_initializer='glorot_normal', use_bias=True,activation='relu',name = 'ae2')(en1)
    de1 = Dense(128, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad1')(en2)
    de2 = Dense(length, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'ad2')(de1)
    
    if testflag==0:
        AEmodel = Model(x_input,de2)
        AEmodel.load_weights(modelname)
        print('load autoencoder model')

        sub_result = Subtract()([x_input, de2])
        cal_norm2 = Lambda(lambda x: tf.norm(x,ord = 2,axis=1))
        sub_norm2 = cal_norm2(sub_result)
        sub_norm2 = Reshape((1,))(sub_norm2)
        division = Lambda(lambda x:tf.div(x[0],x[1]))
        sub_result = division([sub_result,sub_norm2])
        conca_tensor = concatenate([sub_result,en2],axis=1)

        conca_tensor = concatenate([conca_tensor,sub_norm2],axis=1)
    else:
        sub_result = Subtract()([x_input, de2])
        cal_norm2 = Lambda(lambda x: tf.norm(x,ord = 2,axis=1))
        sub_norm2 = cal_norm2(sub_result)
        sub_norm2 = Reshape((1,))(sub_norm2)
        division = Lambda(lambda x:tf.div(x[0],x[1]))
        sub_result = division([sub_result,sub_norm2])
        conca_tensor = concatenate([sub_result,en2],axis=1)

        conca_tensor = concatenate([conca_tensor,sub_norm2],axis=1)

    intermediate = Dense(256, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'hl2')(conca_tensor)
    intermediate = concatenate([intermediate,sub_norm2],axis=1)
    intermediate = Dense(32, kernel_initializer='glorot_normal',use_bias=True,activation='relu',name = 'hl3')(intermediate)
    intermediate = concatenate([intermediate,sub_norm2],axis=1)
    output_pre = Dense(1, kernel_initializer='glorot_normal',use_bias=True,activation='linear', name = 'score')(intermediate)
    dev_model = Model(x_input, output_pre)
    def multi_loss(y_true,y_pred):
        confidence_margin = 5. 

        dev = y_pred
        inlier_loss = K.abs(dev)
        outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))

        sub_nor = tf.norm(sub_result,ord = 2,axis=1) 
        outlier_sub_loss = K.abs(K.maximum(confidence_margin - sub_nor, 0.))
        loss1 =  (1 - y_true) * (inlier_loss+sub_nor) + y_true * (outlier_loss+outlier_sub_loss)

        return loss1

    adm = Adam(lr=0.0001)
    dev_model.compile(loss=multi_loss, optimizer=adm)
    return dev_model

def deviation_network(input_shape, network_depth,modelname,testflag):
    '''
    construct the deviation network-based detection model
    '''
    if network_depth == 4:
        model = dev_network_d(input_shape,modelname,testflag)
    elif network_depth == 2:
        model = auto_encoder(input_shape)

    else:
        sys.exit("The network depth is not set properly")
    return model

def auto_encoder_batch_generator_sup(x,inlier_indices, batch_size, nb_batch, rng):
    """auto encoder batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:
        if data_format == 0:
            ref, training_labels = AE_input_batch_generation_sup(x, inlier_indices,batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, inlier_indices,batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0
def AE_input_batch_generation_sup(train_x,inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = train_x.shape[1]
    ref = np.empty((batch_size, dim))
    training_labels = np.empty((batch_size, dim)) 
    n_inliers = len(inlier_indices)
    for i in range(batch_size):
        sid = rng.choice(n_inliers, 1)
        ref[i] = train_x[inlier_indices[sid]]
        training_labels[i] = train_x[inlier_indices[sid]]
    return np.array(ref), np.array(training_labels)
def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0
 
def input_batch_generation_sup(train_x, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = train_x.shape[1]
    ref = np.empty((batch_size, dim))
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if(i % 2 == 0):   
            sid = rng.choice(n_inliers, 1)
            ref[i] = train_x[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = train_x[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

def input_batch_generation_sup_sparse(train_x, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    '''      
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = train_x[ref, :].toarray()
    return ref, np.array(training_labels)

def load_model_weight_predict(model_name, input_shape, network_depth, test_x):
    '''
    load the saved weights to make predictions
    '''
    model = deviation_network(input_shape, network_depth,model_name,1)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)
    
    if data_format == 0:
        scores = scoring_network.predict(test_x)
    else:
        data_size = test_x.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = test_x[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores

def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()

def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    ''' 
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

def run_devnet(args):
    names = args.data_set.split(',')
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)
        filename = nm.strip()
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()    
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]
        n_outliers_org = outliers.shape[0]

        for i in np.arange(runs): 
            train_x, test_x, train_label, test_label = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)

            print(filename + ': round ' + str(i))
            outlier_indices = np.where(train_label == 1)[0]
            inlier_indices = np.where(train_label == 0)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, Number of outliers in Train data:: %d" % (train_x.shape[0], n_outliers))
            
            n_noise  = len(np.where(train_label == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)    
            
            rng = np.random.RandomState(random_seed)  
            if data_format == 0:                
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)
                    train_x = np.delete(train_x, remove_idx, axis=0)
                    train_label = np.delete(train_label, remove_idx, axis=0)
                    #ae_label = train_x
                noises = inject_noise(outliers, n_noise, random_seed)
                train_x = np.append(train_x, noises, axis = 0)
                train_label = np.append(train_label, np.zeros((noises.shape[0], 1)))
            
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)        
                    retain_idx = set(np.arange(train_x.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    train_x = train_x[retain_idx]
                    train_label = train_label[retain_idx]                               
                
                noises = inject_noise_sparse(outliers, n_noise, random_seed)
                train_x = vstack([train_x, noises])
                train_label = np.append(train_label, np.zeros((noises.shape[0], 1)))
            
            outlier_indices = np.where(train_label == 1)[0]
            inlier_indices = np.where(train_label == 0)[0]
            train_x_inlier = np.delete(train_x, outlier_indices, axis=0)
            print("Processed Train data number:",train_label.shape[0], "outliers number in Train data:",outlier_indices.shape[0],'\n',\
                "normal number in Train data:", inlier_indices.shape[0],"noise number:", n_noise)
            input_shape = train_x.shape[1:]
            n_samples_trn = train_x.shape[0]
            n_outliers = len(outlier_indices)

            n_samples_test = test_x.shape[0]
            test_outlier_indices = np.where(test_label == 1)[0]
            test_inlier_indices = np.where(test_label == 0)[0]
            print("Test data number:",test_label.shape[0],'\n',\
                "outliers number in Test data:",test_outlier_indices.shape[0],"normal number in Test data:",test_inlier_indices.shape[0])

            epochs = args.epochs
            batch_size = args.batch_size    
            nb_batch = args.nb_batch  

            AEmodel = deviation_network(input_shape,2,None,0)  #auto encoder model 预训练
            print('pre-training start....')
            print(AEmodel.summary())
            AEmodel_name = "auto_encoder_normalization"+".h5"
            ae_checkpointer = ModelCheckpoint(AEmodel_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = True)            
            AEmodel.fit_generator(auto_encoder_batch_generator_sup(train_x, inlier_indices, batch_size, nb_batch, rng),
                                             steps_per_epoch = nb_batch,
                                             epochs = 100,
                                             callbacks=[ae_checkpointer])

            print('load autoencoder model....')
            dev_model = deviation_network(input_shape, 4, AEmodel_name, 0)
            print('end-to-end training start....')
            dev_model_name = "./devnet_" + filename + "_" + str(args.cont_rate) + "cr_"  + str(args.batch_size) +"bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d.h5"
            checkpointer = ModelCheckpoint(dev_model_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = True) 
            dev_model.fit_generator(batch_generator_sup(train_x, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                                          steps_per_epoch = nb_batch,
                                          epochs = epochs,
                                          callbacks=[checkpointer])
            print('load model and print results....')
            scores = load_model_weight_predict(dev_model_name, input_shape, 4, test_x)

            rauc[i], ap[i] = aucPerformance(scores, test_label)     
        
        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)

        print("average AUC-ROC: %.4f  average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
        print("std AUC-ROC: %.4f  std AUC-PR: %.4f" % (std_auc, std_aucpr))

        writeResults(filename+'_'+'ae_devnet','training_samples = '+str(n_samples_trn), 'train_outliers = '+str(n_outliers),\
            'test_samples = '+str(n_samples_test), 'test_outliers = '+str(test_outlier_indices.shape[0]),'test_inliers = '+str(test_inlier_indices.shape[0]),\
            'avg_AUC_ROC = '+format(mean_auc,'.4f'), 'avg_AUC_PR = '+format(mean_aucpr,'.4f'), \
            'std_AUC_ROC = '+format(std_auc,'.4f'), 'std_AUC_PR = '+format(std_aucpr,'.4f'), path=args.output)

parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1','2', '4'], default='4', help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default = 512, help = "batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default =20,help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default = 30, help="the number of epochs")
parser.add_argument("--runs", type=int, default = 10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default = 30, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--data_set", type=str, default='nslkdd_normalization', help="a list of data set names")
parser.add_argument("--data_format", choices=['0','1'], default='0',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str, default='./proposed_devnet_auc_performance.csv', help="the output file path")
parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
args = parser.parse_args()
run_devnet(args)

