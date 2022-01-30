
import numpy as np
import keras.backend as K

from keras.models import Model, load_model

from keras.layers import Input, Multiply
from layers import GroupNormalization
from libs.neuron.neuron.layers import SpatialTransformer

# from sklearn.linear_model import RANSACRegressor
from collections import OrderedDict
from scipy.stats import kurtosis, skew


class Embeddings(object):

    def __init__(self, model_path):

        self.model_path = model_path
        self._build_encoder()
        self._make_bidirectional()

    def _build_encoder( self ):

        custom_objects = {'GroupNormalization': GroupNormalization,'SpatialTransformer': SpatialTransformer}
        net = load_model(self.model_path, custom_objects=custom_objects, compile=False)

        self.encoder = Model(inputs=net.input, outputs=net.get_layer('elastic_block96_relu').output, name='encoder')

    def _make_bidirectional( self ):

        inflow_1 = Input(K.int_shape(self.encoder.input[0])[1:], name='input_1')
        inflow_2 = Input(K.int_shape(self.encoder.input[0])[1:], name='input_2')

        outflow_1 = self.encoder([inflow_1, inflow_2])
        outflow_2 = self.encoder([inflow_2, inflow_1])

        outflow = Multiply(name='multiply')([outflow_1, outflow_2])

        self.encoder = Model([inflow_1, inflow_2], outflow, name='encoder')

    def __call__(self, image_1, image_2 ):

        f = self.encoder.predict([image_1[None, ..., None], image_2[None, ..., None]])

        # compute stats
        f_distr = f.reshape(
            np.prod(self.encoder.output_shape[1:-1]), self.encoder.output_shape[-1])

        f_stats = np.concatenate([
            np.mean(f_distr, axis=0), 
            np.std(f_distr, axis=0), 
            skew(f_distr, axis=0), 
            kurtosis(f_distr, axis=0)
        ], axis=-1)

        # package in a dict
        features = OrderedDict()
        for i, f in enumerate(f_stats):
            features['feature_' + str(i)] = f

        return features


class Registration(object):

    def __init__( self, model_path ):
        self.model_path = model_path
        self.network = self._build()

    def _build( self ):
        custom_objects = {'GroupNormalization': GroupNormalization, 'SpatialTransformer': SpatialTransformer}
        return load_model(self.model_path, custom_objects=custom_objects, compile=False)

    def __call__( self, image_1, image_2 ):
        tA, A, tD, D = self.network.predict([image_1[None, ..., None], image_2[None, ..., None]])
        return (tA, tD), (A, D)
