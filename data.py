import numpy as np
import pandas as pd

from keras.utils import Sequence
from keras.models import load_model

from libs.frida.io import ImageLoader, ReadVolume
from libs.frida.transforms import Transform, PadAndCropTo, ZeroOneScaling, TransformFromITKFilter, Resample, ToNumpyArray

from SimpleITK import GetArrayFromImage, Crop, ClampImageFilter

from sklearn.linear_model import RANSACRegressor


class SmartCrop(Transform):

    def __init__( self, margin=0. ):
        self.margin = margin
        self.__load_localizer()
        super(SmartCrop, self).__init__()

    def __load_localizer( self ):
        self.localizer = load_model(r'models/localizer.h5', compile=False)

    def __plot( self, image_arr ):
        import matplotlib.pyplot as plt
        # get anatomical coordinates
        image_arr = np.clip(image_arr, -120, 300)
        anatomical_coordinates = self.localizer.predict(image_arr[..., None])
        plt.plot(anatomical_coordinates)

    def __get_coordinates( self, image_arr, query_lo, query_hi ):

        # get anatomical coordinates
        image_arr = np.clip(image_arr, -120, 300)
        scan_coordinates = np.arange(image_arr.shape[0])
        anatomical_coordinates = self.localizer.predict(image_arr[..., None])

        # linear fit between anatomical and scan coords
        m = RANSACRegressor(random_state=42)
        m.fit(scan_coordinates[..., None], anatomical_coordinates.squeeze())
        coef, intercept = m.estimator_.coef_[0], m.estimator_.intercept_

        # process query
        #safe = lambda x: min(max(x, 0), image_arr.shape[0])
        inverse_fn = lambda x: (x - intercept) / coef

        margin_fn = lambda x: max(x, 0) if inverse_fn(query_lo + self.margin) > 0 else -1.
        query_result_lo = margin_fn((query_lo - intercept) / coef)

        top = image_arr.shape[0] - 1
        margin_fn = lambda x: min(x, top) if inverse_fn(query_hi - self.margin) < top else -1.
        query_result_hi = margin_fn((query_hi - intercept) / coef)

        return query_result_lo, query_result_hi


class CropThorax(SmartCrop):

    def __init__( self, margin=0. ):
        super(CropThorax, self).__init__(margin)

    def __call__(self, image):
        outputs = None
        image_arr = GetArrayFromImage(image)
        pelvis, diaphram = self._SmartCrop__get_coordinates(image_arr, 20, 70)
        if (pelvis >= 0) and (diaphram <= image_arr.shape[0]) and (pelvis < diaphram):
            outputs = Crop(image, [0, 0, int(pelvis)], [0, 0, int(image_arr.shape[0] - diaphram)])
        return outputs


class CropAbdomen(SmartCrop):

    def __init__( self, margin=0. ):
        super(CropAbdomen, self).__init__(margin)

    def __call__(self, image):
        outputs = None
        image_arr = GetArrayFromImage(image)
        pelvis, diaphram = self._SmartCrop__get_coordinates(image_arr, -70, 25)
        if (pelvis >= 0) and (diaphram <= image_arr.shape[0]) and (pelvis < diaphram):
            outputs = Crop(image, [0, 0, int(pelvis)], [0, 0, int(image_arr.shape[0] - diaphram)])
        return outputs


if __name__ == '__main__':

    from SimpleITK import ClampImageFilter
    from SimpleITK import GetImageFromArray
    from SimpleITK import WriteImage

    clamp = ClampImageFilter()
    clamp.SetUpperBound(300)
    clamp.SetLowerBound(-120)

    loader = ImageLoader(
    	ReadVolume(), # or ReadDICOM()
    	CropThorax(),
        Resample(2), 
        PadAndCropTo((192, 192, 160), cval=-1000), 
        TransformFromITKFilter(clamp),
        ZeroOneScaling(),
        ToNumpyArray(add_batch_dim=False, add_singleton_dim=False)
    )

    processed_ct_scan = loader(r'data/LIDIC-IDRI-0001.nii.gz')
    processed_ct_scan = GetImageFromArray(processed_ct_scan)
    WriteImage(processed_ct_scan, r'data/LIDIC-IDRI-0001_processed.nii.gz')