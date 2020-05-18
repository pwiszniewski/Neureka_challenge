import torch.nn as nn


class OneDimCNNBlock(nn.Module):
    def __init__(self, in_shape, num_filt, stride_conv, kernel_conv, stride_maxp, kernel_maxp, padd, drop,
                 dbl_conv=False, batch_norm=False):
        super(OneDimCNNBlock, self).__init__()

        self.layer = nn.Sequential()
        self.layer.add_module(f'maxpool-k={kernel_conv},s={stride_conv}',
                              nn.Conv1d(in_shape[0], num_filt, kernel_size=kernel_conv, stride=stride_conv, padding=padd))
        convout_size = (in_shape[1] - kernel_conv + 2 * padd) // stride_conv + 1
        if dbl_conv:
            self.layer.add_module(f'maxpool-k={kernel_conv},s={stride_conv}',
                                  nn.Conv1d(num_filt, num_filt, kernel_size=kernel_conv, stride=stride_conv, padding=padd))
            convout_size = (convout_size - kernel_conv + 2 * padd) // stride_conv + 1
        if batch_norm:
            self.layer.add_module(f'batch_norm', nn.BatchNorm1d(num_filt))
        self.layer.add_module('relu', nn.ReLU())
        self.layer.add_module(f'maxpool-k={kernel_maxp},s={stride_maxp}', nn.MaxPool1d(kernel_size=kernel_maxp, stride=stride_maxp))
        if drop > 0:
            self.layer.add_module('dropout='+str(drop).replace('.', ','), nn.Dropout(drop))

        self.out_size = (convout_size - kernel_maxp) // kernel_maxp + 1

    def forward(self, x):
        return self.layer(x)

class OneDimCNNLayers(nn.Module):
    def __init__(self, input_shape, num_filters, stride_conv, kernel_conv, stride_maxpool, kernel_maxpool, padd, drop,
                 double_conv,  batch_norm):
        super(OneDimCNNLayers, self).__init__()
        layers = []
        num_levels = len(num_filters)
        for i in range(num_levels):
            layers.append(OneDimCNNBlock(input_shape, num_filters[i], stride_conv[i], kernel_conv[i], stride_maxpool[i],
                                         kernel_maxpool[i], padd[i], drop[i], double_conv[i], batch_norm[i]))
            input_shape = (num_filters[i], layers[-1].out_size)

        self.out_size = layers[-1].out_size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FullConnBlock(nn.Module):
    def __init__(self, input_size, out_size, drop, batch_norm=False):
        super(FullConnBlock, self).__init__()
        linear = [nn.Linear(input_size, out_size)]
        if batch_norm:
            linear.append(nn.BatchNorm1d(out_size))
        self.layer = nn.Sequential(
            *linear,
            nn.ReLU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.layer(x)


class FullConnLayers(nn.Module):
    def __init__(self, input_size, num_units, out_size, drop, batch_norm):
        super(FullConnLayers, self).__init__()
        layers = []
        num_levels = len(num_units)
        for i in range(num_levels):
            layers.append(FullConnBlock(input_size, num_units[i], drop[i], batch_norm[i]))
            input_size = num_units[i]
        layers.append(nn.Linear(input_size, out_size))
        if len(drop) == len(num_units) + 1 and drop[-1] > 0:
            layers.append(nn.Dropout(drop[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class OneDimCNNPytorch(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(OneDimCNNPytorch, self).__init__()
        in_drop = 0

        conv_num_filters = [32, 64, 128, 256]
        conv_stride = [1, 1, 1, 1]
        conv_kernel = [3, 3, 3, 3]
        conv_padd = [0, 0, 0, 0]
        conv_batch_norm = len(conv_num_filters)*[False]
        double_conv = len(conv_num_filters)*[True]
        maxpool_stride = [2, 2, 2, 2]
        maxpool_kernel = [2, 2, 2, 2]
        conv_layers_drop = [0, 0, 0, 0.1]

        fc_units = [512]
        fc_drop = [0, 0]
        fc_batch_norm = [False]

        self.in_dropout = nn.Dropout(in_drop)

        self.conv_layers = OneDimCNNLayers(input_shape, conv_num_filters, conv_stride, conv_kernel, maxpool_stride, maxpool_kernel,
                                     conv_padd, conv_layers_drop, double_conv, conv_batch_norm)

        fc_in_size = conv_num_filters[-1] * self.conv_layers.out_size
        self.fc = FullConnLayers(fc_in_size, fc_units, num_classes, fc_drop, fc_batch_norm)

    def forward(self, x):
        x = self.in_dropout(x)
        out = self.conv_layers(x)
        out = self.fc(out.view(out.size(0), -1))
        return out