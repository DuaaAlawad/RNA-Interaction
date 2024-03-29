import math
import os
import sys
import time
import pandas as pd
from argparse import ArgumentParser
from functools import reduce
import configparser
import numpy as np
import tensorflow as tf
import matplotlib
import csv
matplotlib.use('Agg')
from keras import optimizers
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from utils.callbacks import AccHistoryPlot, EarlyStopping
from utils.basic_modules import *
from utils.sequence_encoder import ProEncoder, RNAEncoder

from tensorflow import keras
from tensorflow.keras import layers

# default program settings
DATA_SET = 'RPI488'
TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"

WINDOW_P_UPLIMIT = 3 
WINDOW_P_STRUCT_UPLIMIT = 3 
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
VECTOR_REPETITION_CNN = 1
RANDOM_SEED = 1
K_FOLD = 5
BATCH_SIZE = 150
FIRST_TRAIN_EPOCHS = [20]
SECOND_TRAIN_EPOCHS = [20]
PATIENCES = [10]
FIRST_OPTIMIZER = 'adam'
SECOND_OPTIMIZER = 'sgd'
SGD_LEARNING_RATE = 0.001
ADAM_LEARNING_RATE = 0.001
FREEZE_SUB_MODELS = True
CODING_FREQUENCY = True
MONITOR = 'acc'
MIN_DELTA = 0.0
SHUFFLE = True
VERBOSE = 2

# get the path of rpiter.py
script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
print("===============================================================================================================================",script_dir)
parent_dir = os.path.dirname(script_dir)
# set paths of data, results and program parameters
DATA_BASE_PATH = parent_dir + '/data/'
RESULT_BASE_PATH = parent_dir + '/result/'
INI_PATH = script_dir + '/utils/data_set_settings.ini'

metrics_whole = {'Conjoint-Struct-CNN-BLSTM': np.zeros(7)}

### plot ROC
#f = open(parent_dir + '/result/RPI488/sample3_structure+third structure.csv','w',encoding='utf-8')
#csv_writer = csv.writer(f)
#csv_writer.writerow(["tpr","fpr"])

print("hello")
parser = ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='The dataset you want to process.')
args = parser.parse_args()
if args.dataset != None:
    DATA_SET = args.dataset
print("Dataset: %s" % DATA_SET)

# gpu memory growth manner for TensorFlow
# to consider version compatibility
if tf.__version__< '2.0':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
else:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


# set result save path
result_save_path = RESULT_BASE_PATH + DATA_SET + "/" + DATA_SET + time.strftime(TIME_FORMAT, time.localtime()) + "/"
if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
out = open(result_save_path + 'result.txt', 'w')


def read_data_pair(path):
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        #/home/dmalawad/Research/RNA_Protein_Interaction/EDLMFC/data/RPI1807_pairs.txt
        # print("path+++++++",path)
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            if label == '1':
                pos_pairs.append((p, r))
            elif label == '0':
                neg_pairs.append((p, r))
    return pos_pairs, neg_pairs


def read_feature(path):
    features = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            features.append(line)
            #the feature from tertiary structure file_Numbers
    # print("features++++++++++++++++++++++++++++++++++++++++++++++++++",features)
    return features


def read_data_seq(path):
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    #return all seq and structure of RNA + Protein
    #print("seq_dict*******************************************************************",seq_dict)
    return seq_dict


# calculate the six metrics of Acc, Sn, Sp, Precision, MCC and AUC
def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    # print(con_matrix)
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P if P > 0 else 0
    Sp = TN / N if N > 0 else 0
    Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
    Pre = (TP) / (TP + FP) if (TP+FP) > 0 else 0
    F1_measure = (2*Sn*Pre)/(Sn+Pre)
    MCC = 0
    tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    #csv_writer.writerow([tpr,fpr])
    return Acc, Sn, Sp, Pre, F1_measure, MCC, AUC


def load_data(data_set):
    pro_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_protein_seq.fa')
    rna_seqs = read_data_seq(DATA_BASE_PATH + "sequence/" + data_set + '_rna_seq.fa')
    pro_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_protein_struct.fa')
    rna_structs = read_data_seq(DATA_BASE_PATH + "structure/" + data_set + '_rna_struct.fa')
    pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_pairs.txt')
    features = read_feature(DATA_BASE_PATH + data_set + '_tertiary_structure.txt')

    return pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs,features


def coding_pairs(pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, features, PE, RE, kind):
    samples = []
    if kind == 1:
        i = 0
    else:
        i = 42
    for pr in pairs:
        if pr[0] in pro_seqs and pr[1] in rna_seqs and pr[0] in pro_structs and pr[1] in rna_structs:
            feature_list = []
            for feature in features:
                feature_list.append(float(feature[i])) #float(feature[i])
            i += 1
            p_seq = pro_seqs[pr[0]]  # protein sequence
            r_seq = rna_seqs[pr[1]]  # rna sequence
            p_struct = pro_structs[pr[0]]  # protein structure
            r_struct = rna_structs[pr[1]]  # rna structure

            p_conjoint = PE.encode_conjoint(p_seq)
            r_conjoint = RE.encode_conjoint(r_seq)
            p_conjoint_struct = PE.encode_conjoint_struct(p_seq, p_struct)
            p_conjoint_struct = np.append(p_conjoint_struct, feature_list)
            r_conjoint_struct = RE.encode_conjoint_struct(r_seq, r_struct)
            r_conjoint_struct = np.append(r_conjoint_struct, feature_list[0])
            # DF44 = pd.DataFrame(p_conjoint_struct)
            # DF44.to_csv('p_conjoint_struct'+str(i)+'.csv')
            if p_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[0], pr))
            elif r_conjoint is 'Error':
                print('Skip {} in pair {} according to conjoint coding process.'.format(pr[1], pr))
            elif p_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[0], pr))
            elif r_conjoint_struct is 'Error':
                print('Skip {} in pair {} according to conjoint_struct coding process.'.format(pr[1], pr))

            else:
                samples.append([[p_conjoint, r_conjoint],
                                [p_conjoint_struct, r_conjoint_struct],
                                kind])
                # samples.append([[p_conjoint_struct, r_conjoint_struct],
                #                 kind])
        else:
            print('Skip pair {} according to sequence dictionary.'.format(pr))
    
    # convert array into dataframe
    # DF = pd.DataFrame(samples)
    # DF.to_csv("samples.csv")
    # # print("samples,*************************************************************",samples)
    size = len(samples)
    # print("size of samples&&&&&&&&&&&&&&&&&&&&&&&&",size)
    
    return samples


def standardization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def pre_process_data(samples, samples_pred=None):
    # np.random.shuffle(samples)

    p_conjoint = np.array([x[0][0] for x in samples])

    r_conjoint = np.array([x[0][1] for x in samples])
    p_conjoint_struct = np.array([x[1][0] for x in samples])
    r_conjoint_struct = np.array([x[1][1] for x in samples])
    y_samples = np.array([x[2] for x in samples])
    # DF_p_conjoint = pd.DataFrame(p_conjoint)
    # DF_p_conjoint.to_csv("p_conjoint.csv")

    # DF_r_conjoint = pd.DataFrame(r_conjoint)
    # DF_r_conjoint.to_csv("p_conjoint_struct.csv")


    p_conjoint, scaler_p = standardization(p_conjoint)
    r_conjoint, scaler_r = standardization(r_conjoint)
    p_conjoint_struct, scaler_p_struct = standardization(p_conjoint_struct)
    r_conjoint_struct, scaler_r_struct = standardization(r_conjoint_struct)

    p_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint])
    r_conjoint_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint])
    p_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct])
    r_conjoint_struct_cnn = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct])

    p_ctf_len = 7 ** WINDOW_P_UPLIMIT
    r_ctf_len = 4 ** WINDOW_R_UPLIMIT
    p_conjoint_previous = np.array([x[-p_ctf_len:] for x in p_conjoint])
    r_conjoint_previous = np.array([x[-r_ctf_len:] for x in r_conjoint])
    np.set_printoptions(threshold=np.inf)
    X_samples = [[p_conjoint, r_conjoint],
                 [p_conjoint_struct, r_conjoint_struct],
                 [p_conjoint_cnn, r_conjoint_cnn],
                 [p_conjoint_struct_cnn, r_conjoint_struct_cnn],
                 [p_conjoint_previous, r_conjoint_previous]
                 ]

    # X_samples2 = pd.DataFrame(X_samples)
    # X_samples2.to_csv("X_samples.tsv")
    # ABOUT this part????????????????????????????????????
    if samples_pred:
        # np.random.shuffle(samples_pred)

        p_conjoint_pred = np.array([x[0][0] for x in samples_pred])
        r_conjoint_pred = np.array([x[0][1] for x in samples_pred])
        p_conjoint_struct_pred = np.array([x[1][0] for x in samples_pred])
        r_conjoint_struct_pred = np.array([x[1][1] for x in samples_pred])
        y_samples_pred = np.array([x[2] for x in samples_pred])

        p_conjoint_pred = scaler_p.transform(p_conjoint_pred)
        r_conjoint_pred = scaler_r.transform(r_conjoint_pred)
        p_conjoint_struct_pred = scaler_p_struct.transform(p_conjoint_struct_pred)
        r_conjoint_struct_pred = scaler_r_struct.transform(r_conjoint_struct_pred)

        p_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_pred])
        r_conjoint_cnn_pred = np.array([list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_pred])
        p_conjoint_struct_cnn_pred = np.array(
            [list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in p_conjoint_struct_pred])
        r_conjoint_struct_cnn_pred = np.array(
            [list(map(lambda e: [e] * VECTOR_REPETITION_CNN, x)) for x in r_conjoint_struct_pred])

        p_conjoint_previous_pred = np.array([x[-p_ctf_len:] for x in p_conjoint_pred])
        r_conjoint_previous_pred = np.array([x[-r_ctf_len:] for x in r_conjoint_pred])

        X_samples_pred = [[p_conjoint_pred, r_conjoint_pred],
                          [p_conjoint_struct_pred, r_conjoint_struct_pred],
                          [p_conjoint_cnn_pred, r_conjoint_cnn_pred],
                          [p_conjoint_struct_cnn_pred, r_conjoint_struct_cnn_pred],
                          [p_conjoint_previous_pred, r_conjoint_previous_pred]
                          ]

        return X_samples, y_samples, X_samples_pred, y_samples_pred

    else:
        return X_samples, y_samples


def sum_power(num, bottom, top):
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))


def get_callback_list(patience, result_path, stage, fold, X_test, y_test):
    earlystopping = EarlyStopping(monitor=MONITOR, min_delta=MIN_DELTA, patience=patience, verbose=1,
                                  mode='auto', restore_best_weights=True)
    acchistory = AccHistoryPlot([stage, fold], [X_test, y_test], data_name=DATA_SET,
                                result_save_path=result_path, validate=0, plot_epoch_gap=10)

    return [acchistory, earlystopping]


def get_optimizer(opt_name):
    if opt_name == 'sgd':
        return keras.optimizers.sgd(lr=SGD_LEARNING_RATE, momentum=0.5)
    elif opt_name == 'adam':
        return keras.optimizers.Adam(lr=ADAM_LEARNING_RATE)
    else:
        return opt_name


def control_model_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

# load data settings
if DATA_SET in ['RPI1807', 'NPInter', 'RPI488']:
    config = configparser.ConfigParser()
    config.read(INI_PATH)
    WINDOW_P_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_UPLIMIT')
    WINDOW_P_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_P_STRUCT_UPLIMIT')
    WINDOW_R_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_UPLIMIT')
    WINDOW_R_STRUCT_UPLIMIT = config.getint(DATA_SET, 'WINDOW_R_STRUCT_UPLIMIT')
    VECTOR_REPETITION_CNN = config.getint(DATA_SET, 'VECTOR_REPETITION_CNN')
    RANDOM_SEED = config.getint(DATA_SET, 'RANDOM_SEED')
    K_FOLD = config.getint(DATA_SET, 'K_FOLD')
    BATCH_SIZE = config.getint(DATA_SET, 'BATCH_SIZE')
    PATIENCES = [int(x) for x in config.get(DATA_SET, 'PATIENCES').replace('[', '').replace(']', '').split(',')]
    FIRST_TRAIN_EPOCHS = [int(x) for x in
                          config.get(DATA_SET, 'FIRST_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    SECOND_TRAIN_EPOCHS = [int(x) for x in
                           config.get(DATA_SET, 'SECOND_TRAIN_EPOCHS').replace('[', '').replace(']', '').split(',')]
    FIRST_OPTIMIZER = config.get(DATA_SET, 'FIRST_OPTIMIZER')
    SECOND_OPTIMIZER = config.get(DATA_SET, 'SECOND_OPTIMIZER')
    SGD_LEARNING_RATE = config.getfloat(DATA_SET, 'SGD_LEARNING_RATE')
    ADAM_LEARNING_RATE = config.getfloat(DATA_SET, 'ADAM_LEARNING_RATE')
    FREEZE_SUB_MODELS = config.getboolean(DATA_SET, 'FREEZE_SUB_MODELS')
    CODING_FREQUENCY = config.getboolean(DATA_SET, 'CODING_FREQUENCY')
    MONITOR = config.get(DATA_SET, 'MONITOR')
    MIN_DELTA = config.getfloat(DATA_SET, 'MIN_DELTA')



# write program parameter settings to result file
settings = (
    """# Analyze data set {}\n
Program parameters:
WINDOW_P_UPLIMIT = {},
WINDOW_R_UPLIMIT = {},
WINDOW_P_STRUCT_UPLIMIT = {},
WINDOW_R_STRUCT_UPLIMIT = {},
VECTOR_REPETITION_CNN = {},
RANDOM_SEED = {},
K_FOLD = {},
BATCH_SIZE = {},
FIRST_TRAIN_EPOCHS = {},
SECOND_TRAIN_EPOCHS = {},
PATIENCES = {},
FIRST_OPTIMIZER = {},
SECOND_OPTIMIZER = {},
SGD_LEARNING_RATE = {},
ADAM_LEARNING_RATE = {},
FREEZE_SUB_MODELS = {},
CODING_FREQUENCY = {},
MONITOR = {},
MIN_DELTA = {},
    """.format(DATA_SET, WINDOW_P_UPLIMIT, WINDOW_R_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT,
               WINDOW_R_STRUCT_UPLIMIT, VECTOR_REPETITION_CNN,
               RANDOM_SEED, K_FOLD, BATCH_SIZE, FIRST_TRAIN_EPOCHS, SECOND_TRAIN_EPOCHS, PATIENCES, FIRST_OPTIMIZER,
               SECOND_OPTIMIZER, SGD_LEARNING_RATE, ADAM_LEARNING_RATE,
               FREEZE_SUB_MODELS, CODING_FREQUENCY, MONITOR, MIN_DELTA)
)

out.write(settings)

PRO_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT)
PRO_STRUCT_CODING_LENGTH = sum_power(7, 1, WINDOW_P_UPLIMIT) + sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT) + 5
RNA_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT)
RNA_STRUCT_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT) + sum_power(7, 1, WINDOW_R_STRUCT_UPLIMIT) + 1

############################################################################################# MAIN Code ##############################################################################################################
# read rna-protein pairs and sequences from data files
pos_pairs, neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, features = load_data(DATA_SET)


print("script_dir   ",script_dir) 
print("script_name ",script_name)
print("sys.argv[0] ",sys.argv[0])

# sequence encoder instances
PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)
RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)


print("Coding positive protein-rna pairs.\n")
np.set_printoptions(threshold=np.inf)
samples = coding_pairs(pos_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, features, PE, RE, kind=1)
# DF = pd.DataFrame(samples)
# DF.to_csv("positive.csv")
sizes = len(samples)
print("size of samples&&&&&&&&&&&&&&&&&&&&&&&&",sizes)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
positive_sample_number = len(samples)
# print("Coding negative protein-rna pairs.\n")
samples += coding_pairs(neg_pairs, pro_seqs, rna_seqs, pro_structs, rna_structs, features, PE, RE, kind=0)
# DF2= pd.DataFrame(samples)
# DF2.to_csv("ALLs2.tsv")
size = len(samples)
print("size of samples&&&&&&&&&&&&&&&&&&&&&&&&",size)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
negative_sample_number = len(samples) - positive_sample_number
sample_num = len(samples)

# positive and negative sample numbers
# print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))
out.write('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negative_sample_number))

X, y = pre_process_data(samples=samples)
size_X = len(X)

# K-fold CV processes
print('\n\nK-fold cross validation processes:\n')
out.write('\n\nK-fold cross validation processes:\n')
for fold in range(K_FOLD):
    train = [i for i in range(sample_num) if i%K_FOLD !=fold]
    test = [i for i in range(sample_num) if i%K_FOLD ==fold]

    # generate train and test data
    X_train_conjoint = [X[0][0][train], X[0][1][train]]
    X_train_conjoint_struct = [X[1][0][train], X[1][1][train]]
    X_train_conjoint_cnn = [X[2][0][train], X[2][1][train]]
    X_train_conjoint_struct_cnn = [X[3][0][train], X[3][1][train]]
    X_train_conjoint_previous = [X[4][0][train], X[4][1][train]]

    X_test_conjoint = [X[0][0][test], X[0][1][test]]
    X_test_conjoint_struct = [X[1][0][test], X[1][1][test]]
    X_test_conjoint_cnn = [X[2][0][test], X[2][1][test]]
    X_test_conjoint_struct_cnn = [X[3][0][test], X[3][1][test]]
    X_test_conjoint_previous = [X[4][0][test], X[4][1][test]]

    y_train_mono = y[train]
    y_train = np_utils.to_categorical(y_train_mono, 2)
    y_test_mono = y[test]
    y_test = np_utils.to_categorical(y_test_mono, 2)

    X_ensemble_train = X_train_conjoint_struct + X_train_conjoint_struct_cnn
    X_ensemble_test = X_test_conjoint_struct + X_test_conjoint_struct_cnn


    print(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    out.write(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
    model_metrics = {'Conjoint-Struct-CNN-BLSTM': np.zeros(7)}

    model_weight_path = result_save_path + 'weights.hdf5'

    module_index = 0

    # =================================================================
    # Conjoint-struct-CNN-LSTM module

    stage = 'Conjoint-Struct-CNN-BLSTM'
    print("\n# Module Conjoint-Struct-CNN-BLSTM part #\n")

    # create model
    print("PRO_STRUCT_CODING_LENGTH", PRO_STRUCT_CODING_LENGTH)
    print("RNA_STRUCT_CODING_LENGTH", RNA_STRUCT_CODING_LENGTH)
    print("VECTOR_REPETITION_CNN", VECTOR_REPETITION_CNN)

    
    model_conjoint_struct_cnn_blstm = conjoint_struct_cnn_blstm(PRO_STRUCT_CODING_LENGTH, RNA_STRUCT_CODING_LENGTH, VECTOR_REPETITION_CNN)
    callbacks = get_callback_list(PATIENCES[0], result_save_path, stage, fold, X_test_conjoint_struct_cnn,
                                  y_test)
    print("model_conjoint_struct_cnn_blstm",model_conjoint_struct_cnn_blstm)

    # first train
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model_conjoint_struct_cnn_blstm.compile(loss='categorical_crossentropy', optimizer= get_optimizer(FIRST_OPTIMIZER),
                                    metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = False
    model_conjoint_struct_cnn_blstm.fit(x=X_train_conjoint_struct_cnn,
                                y=y_train,
                                epochs=FIRST_TRAIN_EPOCHS[0],
                                batch_size=BATCH_SIZE,
                                verbose=VERBOSE,
                                shuffle=SHUFFLE,
                                callbacks=[callbacks[0]])



    # second train
    model_conjoint_struct_cnn_blstm.compile(loss='categorical_crossentropy', optimizer=opt,
                                    metrics=['accuracy'])
    callbacks[0].close_plt_on_train_end = True
    model_conjoint_struct_cnn_blstm.fit(x=X_train_conjoint_struct_cnn,
                                y=y_train,
                                epochs=SECOND_TRAIN_EPOCHS[0],
                                batch_size=BATCH_SIZE,
                                verbose=VERBOSE,
                                shuffle=SHUFFLE,
                                callbacks=callbacks)

    # model_conjoint_struct_cnn_blstm.save('model_conjoint_struct_cnn_blstm.h5')
    ################################################################################################Test
    # test
    y_test_predict = model_conjoint_struct_cnn_blstm.predict(X_test_conjoint_struct_cnn)
    model_metrics['Conjoint-Struct-CNN-BLSTM'] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
    print('Best performance for module Conjoint-Struct-CNN-BLSTM:\n'
          + 'ACC = ' + str(model_metrics['Conjoint-Struct-CNN-BLSTM'][0]) + ' ' + 'SN = ' + str(
        model_metrics['Conjoint-Struct-CNN-BLSTM'][1]) + ' '
          + 'SP = ' + str(model_metrics['Conjoint-Struct-CNN-BLSTM'][2]) + ' ' + 'PRE = ' + str(
        model_metrics['Conjoint-Struct-CNN-BLSTM'][3]) + ' '
          + 'F1_measure = ' + str(metrics_whole['Conjoint-Struct-CNN-BLSTM'][4]) + ' '
          + 'MCC = ' + str(model_metrics['Conjoint-Struct-CNN-BLSTM'][5]) + ' ' + 'AUC = ' + str(
        model_metrics['Conjoint-Struct-CNN-BLSTM'][6]) + '\n')



     # =================================================================

    for key in model_metrics:
        out.write(key + " : " +  'ACC = ' + str(model_metrics[key][0]) + ' ' + 'SN = ' + str(model_metrics[key][1]) + ' '
              + 'SP = ' + str(model_metrics[key][2]) + ' ' + 'PRE = ' + str(model_metrics[key][3]) + ' '
              + 'F1_measure = ' + str(model_metrics[key][4]) + ' '
              + 'MCC = ' + str(model_metrics[key][5]) + ' ' + 'AUC = ' + str(model_metrics[key][6]) + '\n')

    for key in model_metrics:
        metrics_whole[key] += model_metrics[key]


for key in metrics_whole.keys():
    metrics_whole[key] /= K_FOLD
    print('\nMean metrics in {} fold:\n'.format(K_FOLD)
          + 'ACC = ' + str(metrics_whole[key][0]) + ' ' + 'SN = ' + str(metrics_whole[key][1]) + ' '
          + 'SP = ' + str(metrics_whole[key][2]) + ' ' + 'PRE = ' + str(metrics_whole[key][3]) + ' '
          + 'F1_measure = ' + str(metrics_whole[key][4]) + ' ' + 'MCC = ' + str(metrics_whole[key][5]) + ' '
          + 'AUC = ' + str(metrics_whole[key][6]) + '\n')
    out.write('\nMean metrics in {} fold:\n'.format(K_FOLD)+ 'ACC = ' + str(metrics_whole[key][0]) + ' ' + 'SN = ' + str(metrics_whole[key][1]) + ' '
              + 'SP = ' + str(metrics_whole[key][2]) + ' ' + 'PRE = ' + str(metrics_whole[key][3]) + ' '
              + 'F1_measure = ' + str(metrics_whole[key][4]) + ' '
              + 'MCC = ' + str(metrics_whole[key][5]) + ' ' + 'AUC = ' + str(metrics_whole[key][6]) + '\n')
out.flush()
out.close()

