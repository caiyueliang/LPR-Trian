# encoding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
# from keras.layers import *
# from keras.layers import Input, Activation, Conv2D, BatchNormalization, Lambda, MaxPooling2D, Dropout


# # 基于GRU车牌识别模型
# def model_seq_rec():
#     width, height, n_len, n_class = 164, 48, 7, 83 + 1
#     rnn_size = 256
#     input_tensor = Input(name='the_input', shape=(164, 48, 3), dtype='float32')
#     x = input_tensor
#     base_conv = 32
#     print('input_tensor', x.shape)
#
#     for i in range(3):
#         x = Conv2D(base_conv * (2 ** i), (3, 3))(x)
#         print('Conv2D', x.shape)
#         x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#         print('MaxPooling2D', x.shape)          # [None, 18, 4, 128]
#
#     conv_shape = x.get_shape()
#     print(conv_shape)
#     x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
#     print('Reshape', x.shape)                   # [None, 18, 512]
#     x = Dense(32)(x)
#     print('Dense', x.shape)                     # [None, 18, 32]
#     x = BatchNormalization()(x)
#     print('BatchNormalization', x.shape)        # [None, 18, 32]
#     x = Activation('relu')(x)
#     print('Activation', x.shape)                # [None, 18, 32]
#
#     gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
#     print('gru_1', gru_1.shape)                 # [None, None, 256]
#     gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
#     print('gru_1b', gru_1b.shape)               # [None, None, 256]
#     gru1_merged = add([gru_1, gru_1b])
#     print('gru1_merged', gru1_merged.shape)     # [None, None, 256]
#
#     gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
#     print('gru_2', gru_2.shape)                 # [None, None, 256]
#     gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
#     print('gru_2b', gru_2b.shape)               # [None, None, 256]
#     x = concatenate([gru_2, gru_2b])
#     print('concatenate', x.shape)               # [None, None, 512]
#     x = Dropout(0.25)(x)
#     x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
#     print('Dense', x.shape)                     # [None, 18, 84]
#
#     y_pred = x
#     return input_tensor, y_pred


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3))
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        output = self.max_pool(x)
        return output


class CGRU(nn.Module):
    def __init__(self, width, height, n_class, n_c=3):
        super(CGRU, self).__init__()
        assert height % 16 == 0, 'imgH has to be a multiple of 16'
        n_base_conv = 32
        n_hide_size = 256

        self.conv_1 = MyConv2D(in_channels=n_c, out_channels=n_base_conv)
        self.conv_2 = MyConv2D(in_channels=n_base_conv, out_channels=n_base_conv * 2)
        self.conv_3 = MyConv2D(in_channels=n_base_conv * 2, out_channels=n_base_conv * 4)

        self.fc = nn.Linear(in_features=n_base_conv * 16, out_features=n_base_conv)
        self.bn = nn.BatchNorm1d(num_features=18)
        self.relu = nn.ReLU()

        self.gru_1 = nn.GRU(input_size=n_base_conv, hidden_size=n_hide_size, bidirectional=True)
        self.gru_2 = nn.GRU(input_size=n_base_conv * 16, hidden_size=n_hide_size, bidirectional=True)

        self.fc_end = nn.Linear(in_features=n_base_conv * 16, out_features=n_class)
        self.dropout = nn.Dropout(p=0.4)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        # print('input: ', input.size())      # (-1, 3, 48, 164)
        x = self.conv_1(input)
        # print('conv_1: ', x.size())         # (-1, 32, 23, 81)
        x = self.conv_2(x)
        # print('conv_2: ', x.size())         # (-1, 64, 10, 39)
        x = self.conv_3(x)
        # print('conv_3: ', x.size())         # (-1, 128, 4, 18)

        b, c, h, w = x.size()
        # print('b %d, c %d, h %d, w %d' % (b, c, h, w))  # b -1, c 128, h 4, w 18
        x = x.reshape([-1, w, c * h])
        # print('reshape: ', x.size())        # (-1, 18, 512)
        x = self.fc(x)
        # print('fc: ', x.size())             # (-1, 18, 32)
        x = self.bn(x)
        # print('bn: ', x.size())             # (-1, 18, 32)
        x = self.relu(x)

        x = x.permute(1, 0, 2)                # [w, b, c]
        # print('permute: ', x.size())        # (18, -1, 32)

        x, _ = self.gru_1(x)
        # print('gru_1: ', x.size())          # (18, -1, 512)
        x, _ = self.gru_2(x)
        # print('gru_2: ', x.size())          # (18, -1, 512)

        x = self.dropout(x)
        x = self.fc_end(x)
        # print('fc_end: ', x.size())         # (18, -1, 84)
        # x = self.softmax(x)                 # CRNN没有做归一化，同CRNN
        # print('softmax: ', x.size())        # (18, -1, 84)

        output = x
        return output


def softmax_test():
    data = Variable(torch.FloatTensor([[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
                                        [0, 5, 5], [0, 5, 5], [0, 5, 5], [0, 5, 5]],
                                       [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                        [0, 5, 5], [0, 5, 5], [0, 5, 5], [0, 5, 5]]]))
    sofmax = nn.Softmax(dim=2)

    print data
    print data.size()
    data = sofmax(data)
    print data


def gru_test():
    print('gru_test')
    model = CGRU(width=164, height=48, n_class=85)
    data = Variable(torch.randn(1, 3, 48, 164))
    print('input: ', data.size())
    output = model(data)
    print('output: ', output.size())
    # print(output)


if __name__ == '__main__':
    softmax_test()

    gru_test()
