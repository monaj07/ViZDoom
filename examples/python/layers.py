import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano_utils import *
import operator
import theano.printing
RNG = np.random.RandomState(1000)
RELU = lambda x: x * (x > 1e-6)


class ConvLayer(object):
    def __init__(self, input, filter_shape, input_shape, pool_size=(2, 2), activation=RELU):
        # filterShape: [num_filters, num_input_feature_maps, filt_height, filt_width]
        # input_shape: [minibatch_size, num_input_feature_maps, img_height, img_width]

        #input = theano.printing.Print('input')(input)
        # The number of feature maps of the input and the input dimension of the filters must be equal:
        #assert input_shape[1] == filter_shape[1]

        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        if pool_size is not None:
            fan_out = np.prod(filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size))
        else:
            fan_out = np.prod(filter_shape[0] * np.prod(filter_shape[2:]))
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(RNG.uniform(low=-w_bound, high=+w_bound, size=filter_shape), dtype=theano.config.floatX)
        self.w = theano.shared(value=w_values, borrow=True)
        b_values = np.zeros(filter_shape[0], dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(input=input, filters=self.w, filter_shape=filter_shape, input_shape=input_shape)
        if pool_size is not None:
            pooling_out = T.signal.pool.pool_2d(input=input, ds=pool_size, ignore_border=True)
        else:
            pooling_out = conv_out
        self.out = activation(pooling_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.w, self.b]
        self.weight_types = ['w', 'b']
        self.L1 = abs(self.w).sum()
        self.L2 = abs(self.w**2).sum()


class FCLayer(object):
    def __init__(self, input, fan_in, num_hidden, activation=RELU):  # Or you can put activation=T.nnet.relu
        fan_out = num_hidden
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(RNG.uniform(low=-w_bound, high=w_bound, size=(fan_in, fan_out)), dtype=theano.config.floatX)
        self.w = theano.shared(value=w_values, borrow=True)
        self.b = theano.shared(value=np.zeros(fan_out, dtype=theano.config.floatX), borrow=True)

        fc_out = T.dot(input, self.w) + self.b
        self.out = activation(fc_out) if activation is not None else fc_out
        self.params = [self.w, self.b]
        self.weight_types = ['w', 'b']
        self.L1 = abs(self.w).sum()
        self.L2 = abs(self.w**2).sum()

class DropoutLayer(object):
    # Borrowed from https://github.com/uoguelph-mlrg/theano_alexnet/blob/master/lib/layers.py
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []

    def __init__(self, inp, prob_drop=0.5):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=inp.shape)

        self.out = self.flag_on * T.cast(self.mask, theano.config.floatX) * inp + \
            self.flag_off * self.prob_keep * inp

        DropoutLayer.layers.append(self)
        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def setDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def setDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)

class SoftmaxLayer(object):
    def __init__(self, inp, n_classes):
        fan_in = T.shape(inp)[0]
        fan_out = n_classes
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(RNG.uniform(low=-w_bound, high=+w_bound, size=(fan_in, fan_out)), dtype=theano.config.floatX)
        self.w = theano.shared(value=w_values, borrow=True)
        self.b = theano.shared(value=np.zeros(fan_out, dtype=theano.config.floatX), borrow=True)
        self.weight_types = ['w', 'b']
        self.L1 = abs(self.w).sum()
        self.L2 = abs(self.w**2).sum()

        self.p_y_given_x = T.nnet.softmax(T.dot(inp, self.w) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.w, self.b]
        def negative_log_likelihood(self, y):
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def errors_top_x(self, y, num_top=5):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))
        else:
            raise NotImplementedError()

