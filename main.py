import os
import argparse
import numpy as np
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D, Input, GRU, LSTM, Dense, Activation, Dropout, Reshape, Permute, BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import metrics
import utils
import keras.backend as K
K.set_image_data_format('channels_first')


def load_data(feat_folder, fold):
    feat_file_fold = os.path.join(feat_folder, f'mbe_fold{fold}.npz')
    dmp = np.load(feat_file_fold)
    X_train, Y_train, X_test, Y_test = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2'],  dmp['arr_3']
    return X_train, Y_train, X_test, Y_test


def get_model(data_in, data_out, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate):

    spec_start = Input(shape=(data_in.shape[-3], data_in.shape[-2], data_in.shape[-1]))
    spec_x = spec_start
    for i, cnt in enumerate(cnn_pool_size):
        spec_x = Conv2D(filters=cnn_nb_filt, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization(axis=1)(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, cnn_pool_size[i]))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((data_in.shape[-2], -1))(spec_x)

    for r in rnn_nb:
        spec_x = Bidirectional(
            LSTM(r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul'
        )(spec_x)

    for f in fc_nb:
        spec_x = TimeDistributed(Dense(f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(data_out.shape[-1]))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    model = Model(inputs=spec_start, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    model.summary()
    return model


def preprocess_data(X, Y, X_test, Y_test, _seq_len, _nb_ch):
    # split into sequences
    X = utils.split_in_seqs(X, _seq_len)
    Y = utils.split_in_seqs(Y, _seq_len)

    X_test = utils.split_in_seqs(X_test, _seq_len)
    Y_test = utils.split_in_seqs(Y_test, _seq_len)

    X = utils.split_multi_channels(X, _nb_ch)
    X_test = utils.split_multi_channels(X_test, _nb_ch)
    return X, Y, X_test, Y_test


def train(args):
    is_mono = True  # True: mono-channel input, False: binaural input

    feat_folder = args.feat_folder
    models_folder = args.models_folder

    nb_ch = 1 if is_mono else 2
    batch_size = args.batch_size    
    seq_len = 256       # Frame sequence length. Input to the CRNN.
    n_epochs = args.n_epochs      # Training epochs
    patience = int(0.25 * n_epochs)  # Patience for early stopping

    # Number of frames in 1 second, required to calculate F and ER for 1 sec segments.
    # Make sure the nfft and sr are the same as in feature.py
    sr = args.sr
    nfft = args.nfft
    frames_1_sec = int(sr/(nfft/2.0))

    # CRNN model definition
    cnn_nb_filt = 128           # CNN filter size
    cnn_pool_size = [5, 2, 2]   # Maxpooling across frequency. Length of cnn_pool_size =  number of CNN layers
    rnn_nb = [32, 32]           # Number of RNN nodes.  Length of rnn_nb =  number of RNN layers
    fc_nb = [32]                # Number of FC nodes.  Length of fc_nb =  number of FC layers
    dropout_rate = 0.5          # Dropout after each layer

    avg_er = list()
    avg_f1 = list()
    for fold in [1, 2, 3, 4]:
        print(f'Fold {fold}')
        # Load feature and labels, pre-process it
        X, Y, X_test, Y_test = load_data(feat_folder, fold)
        X, Y, X_test, Y_test = preprocess_data(X, Y, X_test, Y_test, seq_len, nb_ch)
        model = get_model(X, Y, cnn_nb_filt, cnn_pool_size, rnn_nb, fc_nb, dropout_rate)

        # Training
        best_epoch, pat_cnt, best_er, f1_for_best_er = 0, 0, 99999, None
        f1_overall_1sec_list, er_overall_1sec_list = [0] * n_epochs, [0] * n_epochs
        posterior_thresh = 0.5
        for i in range(n_epochs):
            print(f'Epoch {i}: ', end='')
            model.fit(
                X, Y,
                batch_size=batch_size,
                validation_data=[X_test, Y_test],
                epochs=1,
                verbose=2
            )

            # Calculate the predictions on test data, in order to calculate ER and F scores
            pred = model.predict(X_test)
            pred_thresh = pred > posterior_thresh
            score_list = metrics.compute_scores(pred_thresh, Y_test, frames_in_1_sec=frames_1_sec)

            f1_overall_1sec_list[i] = score_list['f1_overall_1sec']
            er_overall_1sec_list[i] = score_list['er_overall_1sec']
            pat_cnt = pat_cnt + 1

            # Calculate confusion matrix
            test_pred_cnt = np.sum(pred_thresh, 2)
            Y_test_cnt = np.sum(Y_test, 2)
            conf_mat = confusion_matrix(Y_test_cnt.reshape(-1), test_pred_cnt.reshape(-1))
            conf_mat = conf_mat / (utils.eps + np.sum(conf_mat, 1)[:, None].astype('float'))

            if er_overall_1sec_list[i] < best_er:
                best_er = er_overall_1sec_list[i]
                f1_for_best_er = f1_overall_1sec_list[i]
                model.save(os.path.join(models_folder, f'_fold{fold}_model.keras'))
                best_epoch = i
                pat_cnt = 0

            print(f'F1_overall: {f1_overall_1sec_list[i]}, ER_overall: {er_overall_1sec_list[i]}, best ER: {best_er}, best_epoch: {best_epoch}')

            if pat_cnt > patience:
                break
        avg_er.append(best_er)
        avg_f1.append(f1_for_best_er)
        print(f'Saved model for the best_epoch {best_epoch} with best_er: {best_er} f1_for_best_er: {f1_for_best_er}')

    print(f'\nMETRICS FOR ALL FOUR FOLDS:\n avg_er: {avg_er}, avg_f1: {avg_f1}')
    print(f'MODEL AVERAGE OVER FOUR FOLDS:\n avg_er: {np.mean(avg_er)}, avg_f1: {np.mean(avg_f1)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--feat_folder', type=str, default='feat/')
    parser.add_argument('--models_folder', type=str, default='models/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--nfft', type=float, default=2048, help='length of the windowed signal in FFT')
    parser.add_argument('--sr', type=int, default=44100, help='sampling rate')

    args = parser.parse_args()

    train(args)