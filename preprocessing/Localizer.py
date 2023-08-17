import numpy                    as np

from sklearn.linear_model       import RANSACRegressor
from SimpleITK                  import GetArrayFromImage
from SimpleITK                  import Crop

from frida.transforms           import Transform

# this models works with tensorflow, you can find the environment in the yml file
from tensorflow.keras.models    import load_model 


class LocalizerFactory():
    def __init__(self):
        self.__load()

    def __load( self ):
        self.localizer = load_model(r'localizer.h5', compile=False)

    def init_localizer( self, image ):
        image_arr = GetArrayFromImage(image)

        # get anatomical coordinates
        image_arr              = np.clip(image_arr, -120, 300)
        scan_coordinates       = np.arange(image_arr.shape[0])
        anatomical_coordinates = self.localizer.predict(image_arr[..., None], batch_size=1)

        # linear fit between anatomical and scan coords
        m = RANSACRegressor(random_state=42)
        m.fit(scan_coordinates[..., None], anatomical_coordinates.squeeze())
        coef, intercept = m.estimator_.coef_[0], m.estimator_.intercept_

        return Localizer(coef, intercept, anatomical_coordinates, scan_coordinates)


class Localizer():
    def __init__( self, coef, intercept, anatomical_coords, scan_coords ):
        self.coef                   = coef
        self.intercept              = intercept
        self.anatomical_coords      = anatomical_coords
        self.scan_coords            = scan_coords

    def anatomy2scan( self, query ):
        return (query - self.intercept) / self.coef
    
    def scan2anatomy( self, query ):
        return query * self.coef + self.intercept
    
    def get_anatomical_region( self, upper_anatomical_coord, lower_anatomical_coord, tolerance=0):
        max_scan_coord = np.max(self.scan_coords)

        up = self.anatomy2scan(upper_anatomical_coord)
        if up > max_scan_coord:
            if (self.scan2anatomy(max_scan_coord) - upper_anatomical_coord) < tolerance:
                up = max_scan_coord
            else:
                up = -1

        lo = self.anatomy2scan(lower_anatomical_coord)
        if lo < 0:
            if (lower_anatomical_coord - self.scan2anatomy(0)) < tolerance:
                lo = 0
            else:
                lo = -1

        return up, lo


class SmartCrop(Transform):
    def __init__( self, tolerance=0. ):
        self.tolerance = tolerance
        self.localizer_factory = LocalizerFactory()
        super(SmartCrop, self).__init__()


class CropThoraxAbdomen(SmartCrop):
    def __init__( self, tolerance=0. ):
        super(CropThorax, self).__init__(tolerance)

    def __call__(self, image):
        outputs             = None
        image_arr           = GetArrayFromImage(image)

        localizer           = self.localizer_factory.init_localizer(image_arr)
        pelvis, neck        = localizer.get_anatomical_region(80, -70, self.tolerance)

        if (pelvis >= 0) and (pelvis < neck):
            outputs         = Crop(image, [0, 0, int(pelvis)], [0, 0, int(image_arr.shape[0] - neck)])

        return outputs
    

class CropThorax(SmartCrop):
    def __init__( self, tolerance=0. ):
        super(CropThorax, self).__init__(tolerance)

    def __call__(self, image):
        outputs             = None

        localizer           = self.localizer_factory.init_localizer(image)
        diaphram, neck      = localizer.get_anatomical_region(80, 10, self.tolerance)

        if (diaphram >= 0) and (diaphram < neck):
            outputs         = Crop(image, [0, 0, int(diaphram)], [0, 0, int(image.GetSize()[2] - neck)])

        return outputs
    

class CropAbdomen(SmartCrop):
    def __init__( self, tolerance=0. ):
        super(CropAbdomen, self).__init__(tolerance)

    def __call__(self, image):
        outputs             = None
        image_arr           = GetArrayFromImage(image)

        localizer           = self.localizer_factory.init_localizer(image_arr)
        pelvis, diaphram    = localizer.get_anatomical_region(25, -70, self.tolerance)

        if (pelvis >= 0) and (pelvis < diaphram):
            outputs         = Crop(image, [0, 0, int(pelvis)], [0, 0, int(image_arr.shape[0] - diaphram)])

        return outputs



