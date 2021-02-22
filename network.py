import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import he_uniform

from keras.layers import \
    Input, Conv3D, Conv3DTranspose, ReLU, Cropping3D, ZeroPadding3D, Concatenate, \
    Add, Lambda, Reshape, Dense, GlobalAveragePooling3D

from layers import GroupNormalization
from libs.neuron.layers import SpatialTransformer

from keras.models import Model


class RegistrationNetwork():

    def __init__( self, input_shape ):

        self.input_shape = input_shape

    def convolution( self, x, n_filters, pool=None, name='conv' ):

        # define common parameters
        params = dict(kernel_initializer=he_uniform(seed=42), use_bias=False, padding='same')

        # apply correct convolutional operation
        op = None
        if pool == 'up':
            op = Conv3DTranspose(filters=n_filters, kernel_size=3, strides=2, name=name + '_conv', **params)
        if pool is None:
            op = Conv3D(filters=n_filters, kernel_size=3, name=name + '_conv', **params)
        if pool == 'down':
            op = Conv3D(filters=n_filters, kernel_size=3, strides=2, name=name + '_conv',  **params)

        # normalize and non linearlity
        x = op(x)
        x = GroupNormalization(groups=8, name=name + '_norm')(x)
        x = ReLU(name=name + '_relu')(x)

        return x

    def concat( self, x1, x2, name='concat' ):

        # adapt shape in case of mismatch, x1 is adapted to x2
        if any([not (s1 == s2) for s1, s2 in zip(K.int_shape(x1)[1:-1], K.int_shape(x2)[1:-1])]):
            diff = [s1 - s2 for s1, s2 in zip(K.int_shape(x2)[1:-1], K.int_shape(x1)[1:-1])]
            x1 = Cropping3D(cropping=tuple([(0, np.maximum(0, - d)) for d in diff]), name=name + '_crop')(x1)
            x1 = ZeroPadding3D(padding=tuple([(0, np.maximum(0, d)) for d in diff]), name=name + '_pad')(x1)

        x = Concatenate(name=name + '_concat')([x1, x2])

        return x

    def affine( self, fixed, moving ):

        outputs = Concatenate(axis=-1, name='affine_concat')([fixed, moving])
        features = 16
        while K.int_shape(outputs)[1] > 7:
            outputs = self.convolution(outputs, n_filters=features, pool='down', name='affine_block' + str(features))
            features = features * 2
        outputs = GlobalAveragePooling3D(name='affine_gpool')(outputs)
        outputs = Dense(1024, activation='relu', name='fc')(outputs)

        # build affine transform matrix
        W = Dense(9, name='affine_W')(outputs)
        W = Reshape((1, 3, 3, 1), name='affine_W_reshape')(W)
        b = Dense(3, name='affine_b')(outputs)
        b = Reshape((1, 3, 1, 1), name='affine_b_reshape')(b)

        # apply transform
        concat_fn = lambda x: K.squeeze(K.squeeze(K.concatenate(x, 3), 1), axis=-1)
        transform = Lambda(function=concat_fn, name='A')([W, b])
        wrapped = SpatialTransformer(name='wrapA')([moving, transform])

        return transform, wrapped

    def deformable( self, fixed, moving ):

        # using the output of the affine warp
        outputs = Concatenate(axis=-1, name='elastic_concat')([fixed, moving])

        # encoder
        outputs = skip_0 = self.convolution(outputs, n_filters=16, pool='down', name='elastic_downblock16')
        outputs = skip_1 = self.convolution(outputs, n_filters=32, pool='down', name='elastic_downblock32')
        outputs = skip_2 = self.convolution(outputs, n_filters=64, pool='down', name='elastic_downblock64')
        outputs = skip_3 = self.convolution(outputs, n_filters=80, pool='down', name='elastic_downblock80')
        
        h = self.convolution(outputs, n_filters=96, name='elastic_block96')

        # decoder
        outputs = self.concat(h, skip_3, name='elastic_upblock80')
        outputs = self.convolution(outputs, n_filters=80, pool='up', name='elastic_upblock80')
        outputs = self.concat(outputs, skip_2, name='elastic_upblock64')
        outputs = self.convolution(outputs, n_filters=64, pool='up', name='elastic_upblock64')
        outputs = self.concat(outputs, skip_1, name='elastic_upblock32')
        outputs = self.convolution(outputs, n_filters=32, pool='up', name='elastic_upblock32')
        outputs = self.concat(outputs, skip_0, name='elastic_upblock16')
        outputs = self.convolution(outputs, n_filters=16, pool='up', name='elastic_upblock16')

        # apply transform
        transform = Conv3D(3, 1, use_bias=False, padding='same', name='E')(outputs)
        wrapped = SpatialTransformer(name='wrapE')([moving, transform])

        return transform, wrapped

    def build( self, name ):

        fixed = Input(shape=self.input_shape, name='input_fixed')
        moving = Input(shape=self.input_shape, name='input_moving')
        transform_0, wrapped_0 = self.affine(fixed, moving)
        transform_1, wrapped_1 = self.deformable(fixed, wrapped_0)

        return Model([fixed, moving], [transform_0, wrapped_0, transform_1, wrapped_1], name=name)


class Loss():

    def __init__( self, input_shape, smooth=1, window=9, eps=1e-5 ):

        self.input_shape = input_shape
        self.smooth = smooth
        self.win = [window, ] * 3
        self.eps = eps

    def cc( self, y_true, y_pred ):

        y_true = tf.nn.avg_pool3d(y_true, [1, self.smooth, self.smooth, self.smooth, 1], [1, ]*5, 'SAME')
        y_pred = tf.nn.avg_pool3d(y_pred, [1, self.smooth, self.smooth, self.smooth, 1], [1, ]*5, 'SAME')

        sizes = np.prod(self.input_shape)

        flatten1 = tf.reshape(y_true, [-1, sizes])
        flatten2 = tf.reshape(y_pred, [-1, sizes])

        mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
        mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])

        var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
        var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
        cov12 = tf.reduce_mean((flatten1 - mean1) * (flatten2 - mean2), axis=-1)

        pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        return 1 - pearson_r


class Penalty():

    def __init__(self, input_shape, batch_size):

        self.input_shape = input_shape
        self.batch_size = batch_size

    def affine( self, y_true, y_pred ):

        A = tf.linalg.eye(3, 3, batch_shape=[self.batch_size]) + y_pred[:, :3, :3]

        # determinant should be close to one
        determinant_loss = tf.nn.l2_loss(tf.linalg.det(A) - 1.)

        # should be orthogonal, A'A eigenvalues close to 1
        covariance = tf.matmul(A, A, True) + 1e-5
        eigvalsh = tf.linalg.eigvalsh(covariance)
        ortho_loss = (eigvalsh + 1e-5) + ((1 + 1e-5) ** 2) / (eigvalsh + 1e-5)
        ortho_loss = - 6. + tf.reduce_sum(ortho_loss, axis=1)

        return ortho_loss + determinant_loss

    def elastic( self, y_true, y_pred ):

        size = np.prod(self.input_shape[1:-1]) * 3
        d1 = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        d2 = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        d3 = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
        loss = (tf.reduce_sum(d1*d1, axis=(1, 2, 3, 4)))
        loss = loss + (tf.reduce_sum(d2*d2, axis=(1, 2, 3, 4)))
        loss = loss + (tf.reduce_sum(d3*d3, axis=(1, 2, 3, 4)))
        loss = loss / size

        return loss
