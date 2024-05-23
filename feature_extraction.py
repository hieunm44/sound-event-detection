import argparse
import wave
import numpy as np
import librosa
import os
from sklearn import preprocessing


def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array
    Supports 24-bit wav-format
    
    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None


def load_desc_file(desc_file, class_labels):
    desc_dict = dict()
    for line in open(desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in desc_dict:
            desc_dict[name] = list()
        desc_dict[name].append([float(words[2]), float(words[3]), class_labels[words[-1]]])
    return desc_dict


def extract_mbe(y, sr, nfft, n_mels):
    spec = librosa.feature.melspectrogram(y=y, n_fft=nfft, hop_length=nfft//2, power=1, n_mels=n_mels)
    # mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=n_mels)
    return spec


def extract_features(args):
    # setup
    is_mono = True
    class_labels = {
        'brakes squeaking': 0,
        'car': 1,
        'children': 2,
        'large vehicle': 3,
        'people speaking': 4,
        'people walking': 5
    }

    folds_list = [1, 2, 3, 4]
    eval_setup_folder = args.eval_setup_folder
    audio_folder = args.audio_folder
    feat_folder = args.feat_folder

    # User set parameters
    nfft = args.nfft
    win_len = nfft
    hop_len = win_len // 2
    n_mels = args.n_mels
    sr = args.sr

    # Feature extraction and label generation
    # Load labels
    train_file = os.path.join(eval_setup_folder, 'street_fold1_train.txt')
    evaluate_file = os.path.join(eval_setup_folder, 'street_fold1_evaluate.txt')
    desc_dict = load_desc_file(train_file, class_labels)
    desc_dict.update(load_desc_file(evaluate_file, class_labels))
    # Extract features for all audio files, and save it along with labels
    for audio_filename in os.listdir(audio_folder):
        audio_filepath = os.path.join(audio_folder, audio_filename)
        print('Extracting features and label for : {}'.format(audio_filename))
        y, sample_rate = load_audio(audio_filepath, mono=is_mono, fs=sr)
        mbe = None

        if is_mono:
            mbe = extract_mbe(y, sr, nfft, n_mels).T
        else:
            for ch in range(y.shape[0]):
                mbe_ch = extract_mbe(y[ch, :], sr, nfft, n_mels).T
                if mbe is None:
                    mbe = mbe_ch
                else:
                    mbe = np.concatenate((mbe, mbe_ch), 1)

        label = np.zeros((mbe.shape[0], len(class_labels)))
        tmp_data = np.array(desc_dict[audio_filename])
        frame_start = np.floor(tmp_data[:, 0] * sr / hop_len).astype(int)
        frame_end = np.ceil(tmp_data[:, 1] * sr / hop_len).astype(int)
        se_class = tmp_data[:, 2].astype(int)
        for ind, val in enumerate(se_class):
            label[frame_start[ind]:frame_end[ind], val] = 1
        tmp_feat_file = os.path.join(feat_folder, f'{audio_filename}.npz')
        np.savez(tmp_feat_file, mbe, label)

    # Normalize features
    for fold in folds_list:
        train_file = os.path.join(eval_setup_folder, f'street_fold{fold}_train.txt')
        evaluate_file = os.path.join(eval_setup_folder, f'street_fold{fold}_evaluate.txt')
        train_dict = load_desc_file(train_file, class_labels)
        test_dict = load_desc_file(evaluate_file, class_labels)

        X_train, Y_train, X_test, Y_test = None, None, None, None
        for key in train_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, f'{key}.npz')
            dmp = np.load(tmp_feat_file)
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
            if X_train is None:
                X_train, Y_train = tmp_mbe, tmp_label
            else:
                X_train, Y_train = np.concatenate((X_train, tmp_mbe), 0), np.concatenate((Y_train, tmp_label), 0)

        for key in test_dict.keys():
            tmp_feat_file = os.path.join(feat_folder, f'{key}.npz')
            dmp = np.load(tmp_feat_file)
            tmp_mbe, tmp_label = dmp['arr_0'], dmp['arr_1']
            if X_test is None:
                X_test, Y_test = tmp_mbe, tmp_label
            else:
                X_test, Y_test = np.concatenate((X_test, tmp_mbe), 0), np.concatenate((Y_test, tmp_label), 0)

        # Normalize the training data, and scale the testing data using the training data weights
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        normalized_feat_file = os.path.join(feat_folder, f'mbe_fold{fold}.npz')
        np.savez(normalized_feat_file, X_train, Y_train, X_test, Y_test)
        print(f'Normalized feat file : {normalized_feat_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_folder', type=str, default='audio/street/')
    parser.add_argument('--eval_setup_folder', type=str, default='evaluation_setup')
    parser.add_argument('--feat_folder', type=str, default='feat/')
    parser.add_argument('--nfft', type=float, default=2048, help='length of the windowed signal in FFT')
    parser.add_argument('--n_mels', type=float, default=40, help='number of mel bands')
    parser.add_argument('--sr', type=int, default=44100, help='sampling rate')

    args = parser.parse_args()

    extract_features(args)
