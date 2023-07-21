"""
    This python file is working with the localizer-CT model from tensorflow
"""
import os
os['CUDA_VISIBLE_DEVICES'] = '0'

import numpy                    as     np

from   tensorflow.keras.models  import load_model
from   sklearn.linear_model     import RANSACRegressor
from   SimpleITK                import GetArrayFromImage, Crop


# import local libraries 
import sys
sys.path.append('../../')

from   libs.frida.transforms    import Transform


class SmartCrop(Transform):
    def __init__( self, margin=0. ):
        self.margin = margin
        self.__load_localizer()
        super(SmartCrop, self).__init__()


    def __load_localizer( self ):
        self.localizer = load_model(r'localizer.h5', compile=False)


    def __plot( self, image_arr ):
        import matplotlib.pyplot as plt
        # get anatomical coordinates
        image_arr              = np.clip(image_arr, -120, 300)
        anatomical_coordinates = self.localizer.predict(image_arr[..., None])
        plt.plot(anatomical_coordinates)


    def __get_coordinates( self, image_arr, query_lo, query_hi ):
        # get anatomical coordinates
        image_arr              = np.clip(image_arr, -120, 300)
        scan_coordinates       = np.arange(image_arr.shape[0])
        anatomical_coordinates = self.localizer.predict(image_arr[..., None], batch_size=1)

        # linear fit between anatomical and scan coords
        m = RANSACRegressor(random_state=42)
        m.fit(scan_coordinates[..., None], anatomical_coordinates.squeeze())
        coef, intercept = m.estimator_.coef_[0], m.estimator_.intercept_

        # process query
        #safe = lambda x: min(max(x, 0), image_arr.shape[0])
        inverse_fn      = lambda x: (x - intercept) / coef

        margin_fn       = lambda x: max(x, 0) if inverse_fn(query_lo + self.margin) > 0 else -1.
        query_result_lo = margin_fn((query_lo - intercept) / coef)

        top             = image_arr.shape[0] - 1
        margin_fn       = lambda x: min(x, top) if inverse_fn(query_hi - self.margin) < top else -1.
        query_result_hi = margin_fn((query_hi - intercept) / coef)

        return query_result_lo, query_result_hi


class CropThorax(SmartCrop):
    def __init__( self, margin=0. ):
        super(CropThorax, self).__init__(margin)

    def __call__(self, image):
        outputs   = None
        image_arr = GetArrayFromImage(image)
        pelvis, diaphram = self._SmartCrop__get_coordinates(image_arr, 10, 80) # Initially: (image_arr, 20, 70)

        if (pelvis >= 0) and (diaphram <= image_arr.shape[0]) and (pelvis < diaphram):
            outputs = Crop(image, [0, 0, int(pelvis)], [0, 0, int(image_arr.shape[0] - diaphram)])

        return outputs

class CropAbdomen(SmartCrop):
    def __init__( self, margin=0. ):
        super(CropAbdomen, self).__init__(margin)

    def __call__(self, image):
        outputs          = None
        image_arr        = GetArrayFromImage(image)
        pelvis, diaphram = self._SmartCrop__get_coordinates(image_arr, -70, 25)

        if (pelvis >= 0) and (diaphram <= image_arr.shape[0]) and (pelvis < diaphram):
            outputs      = Crop(image, [0, 0, int(pelvis)], [0, 0, int(image_arr.shape[0] - diaphram)])

        return outputs

