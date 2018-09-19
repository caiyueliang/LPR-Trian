# encoding:utf-8
import os
import argparse

import cv2
import numpy as np
from keras.layers import *
from keras import backend as K
from keras.layers import Input, Activation, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         '港', '学', '使', '警', '澳', '挂', '军', '北', '南', '广',
         '沈', '兰', '成', '济', '海', '民', '航', '空',
         ]

# CHARS = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑",
#          u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤",
#          u"桂", u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁",
#          u"新",
#          u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9",
#          u"A", u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K",
#          u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V",
#          u"W", u"X", u"Y", u"Z",
#          u"港", u"学", u"使", u"警", u"澳", u"挂", u"军", u"北", u"南", u"广",
#          u"沈", u"兰", u"成", u"济", u"海", u"民", u"航", u"空"]

CHARS_DICT = {char.decode("utf-8"): i for i, char in enumerate(CHARS)}

NUM_CHARS = len(CHARS)


# The actual loss calc occurs here despite it not being
# an internal Keras loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    print('y_pred', y_pred.shape)
    # y_pred = y_pred[:, :, 0, :]             # !!!!!!!!! 基于GRU要注释掉这一行
    print('y_pred', y_pred.shape)

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# ======================================================================================================================
def build_model(width, num_channels):
    input_tensor = Input(name='the_input', shape=(width, 48, num_channels), dtype='float32')
    x = input_tensor
    base_conv = 32
    print('input_tensor', x.shape)

    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3), padding="same")(x)
        print('Conv2D', x.shape)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        print('MaxPooling2D', x.shape)

    x = Conv2D(256, (5, 5))(x)
    print('Conv2D', x.shape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1))(x)
    print('Conv2D', x.shape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(NUM_CHARS+1, (1, 1))(x)
    print('Conv2D', x.shape)
    x = Activation('softmax')(x)

    y_pred = x
    return input_tensor, y_pred


# 基于GRU车牌识别模型
def model_seq_rec():
    width, height, n_len, n_class = 164, 48, 7, NUM_CHARS + 1
    rnn_size = 256
    input_tensor = Input((164, 48, 3))
    x = input_tensor
    base_conv = 32

    for i in range(3):
        x = Conv2D(base_conv * (2 ** i), (3, 3))(x)
        # print('Conv2D', x.shape)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # print('MaxPooling2D', x.shape)          # [None, 18, 4, 128]

    conv_shape = x.get_shape()
    # print(conv_shape)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    # print('Reshape', x.shape)                   # [None, 18, 512]
    x = Dense(32)(x)
    # print('Dense', x.shape)                     # [None, 18, 32]
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    # print('gru_1', gru_1.shape)                 # [None, None, 256]
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    # print('gru_1b', gru_1b.shape)               # [None, None, 256]
    gru1_merged = add([gru_1, gru_1b])
    # print('gru1_merged', gru1_merged.shape)     # [None, None, 256]

    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    # print('gru_2', gru_2.shape)                 # [None, None, 256]
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    # print('gru_2b', gru_2b.shape)               # [None, None, 256]
    x = concatenate([gru_2, gru_2b])
    # print('concatenate', x.shape)               # [None, None, 512]
    x = Dropout(0.25)(x)
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
    # print('Dense', x.shape)                     # [None, 18, 84]

    # base_model = Model(inputs=input_tensor, outputs=x)
    # base_model.load_weights(model_path)
    # return base_model

    y_pred = x
    return input_tensor, y_pred


# ======================================================================================================================
def encode_label(s):
    s = s.decode("utf-8")
    # print(s)

    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        # print(c)
        label[i] = CHARS_DICT[c]

    print('label', label)
    return label


def parse_line(line):
    parts = line.split(':')
    filename = parts[0]
    label = encode_label(parts[1].strip().upper())
    return filename, label


class TextImageGenerator:
    def __init__(self, img_dir, label_file, batch_size, img_size, input_length, num_channels=3, label_len=5):
        self._img_dir = img_dir
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._input_len = input_length
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()

    def init(self):
        self.labels = []
        with open(self._label_file) as f:
            for line in f:
                filename, label = parse_line(line)
                self.filenames.append(filename)
                self.labels.append(label)
                self._num_examples += 1
        self.labels = np.float32(self.labels)

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end >= self._num_examples:
            self._next_index = 0
            self._num_epoches += 1
            end = self._num_examples
            batch_size = self._num_examples - start
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])
        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            img = cv2.imread(os.path.join(self._img_dir, fname))
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        input_length[:] = self._input_len
        label_length[:] = self._label_len
        outputs = {'ctc': np.zeros([batch_size])}
        inputs = {'the_input': images,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        return inputs, outputs

    def get_data(self):
        while True:
            yield self.next_batch()


def train(args):
    """Train the OCR model
    """
    ckpt_dir = os.path.dirname(args.c)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.log != '' and not os.path.isdir(args.log):
        os.makedirs(args.log)
    label_len = args.label_len

    # input_tensor, y_pred = build_model(args.img_size[0], args.num_channels)
    input_tensor, y_pred = model_seq_rec()

    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    pred_length = int(y_pred.shape[1])
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    if args.pre != '':
        model.load_weights(args.pre)

    print("args.ti: %s" % args.ti)
    print("args.tl: %s" % args.tl)
    print("batch_size: %s" % args.b)
    print("img_size: %s" % args.img_size)
    print("input_length: %s" % pred_length)
    print("num_channels: %s" % args.num_channels)
    print("label_len: %s" % label_len)
    train_gen = TextImageGenerator(img_dir=args.ti,
                                 label_file=args.tl,
                                 batch_size=args.b,
                                 img_size=args.img_size,
                                 input_length=pred_length,
                                 num_channels=args.num_channels,
                                 label_len=label_len)

    print("args.vi: %s" % args.vi)
    print("args.vl: %s" % args.vl)
    print("batch_size: %s" % args.b)
    print("img_size: %s" % args.img_size)
    print("input_length: %s" % pred_length)
    print("num_channels: %s" % args.num_channels)
    print("label_len: %s" % label_len)
    val_gen = TextImageGenerator(img_dir=args.vi,
                                 label_file=args.vl,
                                 batch_size=args.b,
                                 img_size=args.img_size,
                                 input_length=pred_length,
                                 num_channels=args.num_channels,
                                 label_len=label_len)

    checkpoints_cb = ModelCheckpoint(args.c, period=1)
    cbs = [checkpoints_cb]

    if args.log != '':
        tfboard_cb = TensorBoard(log_dir=args.log, write_images=True)
        cbs.append(tfboard_cb)

    print("train_gen.get_data: %s" % train_gen.get_data())
    print("steps_per_epoch: %d" % ((train_gen._num_examples+train_gen._batch_size-1) // train_gen._batch_size))
    print("val_gen.get_data: %s" % val_gen.get_data())
    print("validation_steps: %d" % ((val_gen._num_examples+val_gen._batch_size-1) // val_gen._batch_size))
    print("args.start_epoch: %d" % args.start_epoch)

    model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=(train_gen._num_examples+train_gen._batch_size-1) // train_gen._batch_size,
                        epochs=args.n,
                        validation_data=val_gen.get_data(),
                        validation_steps=(val_gen._num_examples+val_gen._batch_size-1) // val_gen._batch_size,
                        callbacks=cbs,
                        initial_epoch=args.start_epoch)


def train_1(args):
    """Train the OCR model
    """
    ckpt_dir = os.path.dirname(args.c)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.log != '' and not os.path.isdir(args.log):
        os.makedirs(args.log)
    label_len = args.label_len

    input_tensor, y_pred = build_model(args.img_size[0], args.num_channels)

    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    pred_length = int(y_pred.shape[1])
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.0, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

    if args.pre != '':
        model.load_weights(args.pre)

    train_gen = TextImageGenerator(img_dir=args.ti,
                                 label_file=args.tl,
                                 batch_size=args.b,
                                 img_size=args.img_size,
                                 input_length=pred_length,
                                 num_channels=args.num_channels,
                                 label_len=label_len)

    val_gen = TextImageGenerator(img_dir=args.vi,
                                 label_file=args.vl,
                                 batch_size=args.b,
                                 img_size=args.img_size,
                                 input_length=pred_length,
                                 num_channels=args.num_channels,
                                 label_len=label_len)

    checkpoints_cb = ModelCheckpoint(args.c, period=1)
    cbs = [checkpoints_cb]

    if args.log != '':
        tfboard_cb = TensorBoard(log_dir=args.log, write_images=True)
        cbs.append(tfboard_cb)

    model.fit_generator(generator=train_gen.get_data(),
                        steps_per_epoch=(train_gen._num_examples+train_gen._batch_size-1) // train_gen._batch_size,
                        epochs=args.n,
                        validation_data=val_gen.get_data(),
                        validation_steps=(val_gen._num_examples+val_gen._batch_size-1) // val_gen._batch_size,
                        callbacks=cbs,
                        initial_epoch=args.start_epoch)


def export(args):
    """Export the model to a hdf5 file
    """
    input_tensor, y_pred = build_model(None, args.num_channels)
    model = Model(inputs=input_tensor, outputs=y_pred)
    model.save(args.m)
    print('model saved to {}'.format(args.m))


def main():
    ps = argparse.ArgumentParser()
    ps.add_argument('-num_channels', type=int, help='number of channels of the image', default=3)
    subparsers = ps.add_subparsers()

    # Parser for arguments to train the model
    parser_train = subparsers.add_parser('train', help='train the model')
    parser_train.add_argument('-ti', help='训练图片目录', required=True)
    parser_train.add_argument('-tl', help='训练标签文件', required=True)
    parser_train.add_argument('-vi', help='验证图片目录', required=True)
    parser_train.add_argument('-vl', help='验证标签文件', required=True)
    parser_train.add_argument('-b', type=int, help='batch size', required=True)
    parser_train.add_argument('-img-size', type=int, nargs=2, help='训练图片宽和高', required=True)
    parser_train.add_argument('-pre', help='pre trained weight file', default='')
    parser_train.add_argument('-start-epoch', type=int, default=0)
    parser_train.add_argument('-n', type=int, help='number of epochs', required=True)
    parser_train.add_argument('-label-len', type=int, help='标签长度', default=7)
    parser_train.add_argument('-c', help='checkpoints format string', required=True)
    parser_train.add_argument('-log', help='tensorboard 日志目录, 默认为空', default='')
    parser_train.set_defaults(func=train)
    # parser_train.set_defaults(func=train_1)

    # Argument parser of arguments to export the model
    parser_export = subparsers.add_parser('export', help='将模型导出为hdf5文件')
    parser_export.add_argument('-m', help='导出文件名(.h5)', required=True)
    parser_export.set_defaults(func=export)

    args = ps.parse_args()
    args.func(args)


def test_model_layers():
    input_tensor, y_pred = build_model(160, 3)
    my_model = Model(inputs=input_tensor, outputs=y_pred)

    x = np.zeros([1, 160, 40, 3])
    print('x', x.shape)
    y = my_model.predict(x)  # 预测
    print('y', y.shape)


def test_model_layers_1():
    input_tensor, y_pred = model_seq_rec()
    my_model = Model(inputs=input_tensor, outputs=y_pred)

    x = np.zeros([1, 164, 48, 3])
    print('x', x.shape)
    y = my_model.predict(x)  # 预测
    print('y', y.shape)


if __name__ == '__main__':
    main()

    # test_model_layers()
    # test_model_layers_1()
