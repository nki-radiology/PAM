import numpy                    as np

from sklearn.linear_model       import RANSACRegressor
from SimpleITK                  import GetArrayFromImage
from SimpleITK                  import Crop

from frida.transforms           import Transform

# this models works with tensorflow, you can find the environment in the yml file
from tensorflow.keras.models    import load_model 


class Localizer():

    NECK_BASE       = 80
    DIAPHRAGM       = -10
    PELVIC_FLOOR    = -70

    def __init__(self) -> None:
        self.is_fitted  = False
        self.localizer  = load_model(r'preprocessing/localizer.h5', compile=False)


    def fit(self, image):
        image_arr = GetArrayFromImage(image)

        # get anatomical coordinates
        image_arr              = np.clip(image_arr, -120, 300)
        scan_coordinates       = np.arange(image_arr.shape[0])
        anatomical_coordinates = self.localizer.predict(image_arr[..., None], batch_size=1)

        # linear fit between anatomical and scan coords
        m = RANSACRegressor(random_state=42)
        m.fit(scan_coordinates[..., None], anatomical_coordinates.squeeze())
        coef, intercept = m.estimator_.coef_[0], m.estimator_.intercept_

        # set params
        self.__anatomical_coords    = anatomical_coordinates
        self.__scan_coords          = scan_coordinates
        self.coef                   = coef
        self.intercept              = intercept
        self.max_coord_scan         = np.max(scan_coordinates)

        self.is_fitted              = True


    def anatomy2scan( self, query ):
        return (query - self.intercept) / self.coef
    

    def scan2anatomy( self, query ):
        return query * self.coef + self.intercept
    

    def get_anatomical_region( self, up, lo, tolerance=0):
        # BUGFIX make it so that it includes as much as possible
        up = self.anatomy2scan(up)
        if up > self.max_coord_scan:
            if (self.scan2anatomy(self.max_coord_scan) - up) < tolerance:
                up = self.max_coord_scan
            else:
                up = -1

        lo = self.anatomy2scan(lo)
        if lo < 0:
            if (lo - self.scan2anatomy(0)) < tolerance:
                lo = 0
            else:
                lo = -1

        return up, lo


class SmartCrop(Transform):
    def __init__( self, up, lo, tolerance=0.):
        self.tolerance              = tolerance
        self.up                     = up
        self.lo                     = lo

        self.localizer              = Localizer()
        super(SmartCrop, self).__init__()


    def __call__(self, image):
        outputs             = None
        self.localizer.fit(image) 

        image_arr           = GetArrayFromImage(image)
        lower, upper        = self.localizer.get_anatomical_region(self.up, self.lo, self.tolerance)

        if (lower >= 0) and (lower < upper):
            outputs         = Crop(image, [0, 0, int(lower)], [0, 0, int(image_arr.shape[0] - upper)])

        return outputs
    
    
class ZombieCrop(Transform):
    def __init__(self, smart_crop):
        self.localizer = smart_crop.localizer
        self.tolerance = smart_crop.tolerance
        self.up        = smart_crop.up
        self.lo        = smart_crop.lo
        super().__init__()
        

    def __call__(self, image):
        outputs             = None

        image_arr           = GetArrayFromImage(image)
        lower, upper        = self.localizer.get_anatomical_region(self.up, self.lo, self.tolerance)

        if (lower >= 0) and (lower < upper):
            outputs         = Crop(image, [0, 0, int(lower)], [0, 0, int(image_arr.shape[0] - upper)])

        return outputs
    

class CropThoraxAbdomen(SmartCrop):
    def __init__(self,  up=Localizer.NECK_BASE, lo=Localizer.PELVIC_FLOOR, tolerance=0):
        super().__init__(up, lo, tolerance)


class CropThorax(SmartCrop):
    def __init__(self, up=Localizer.NECK_BASE, lo=Localizer.DIAPHRAGM, tolerance=0):
        super().__init__(up, lo, tolerance)
    

class CropAdbomen(SmartCrop):
    def __init__(self, up=Localizer.DIAPHRAGM, lo=-Localizer.PELVIC_FLOOR, tolerance=0):
        super().__init__(up, lo, tolerance)


