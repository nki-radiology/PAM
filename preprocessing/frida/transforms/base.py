import SimpleITK    as     sitk
from   numpy        import floor, ceil
from   numpy.random import uniform


class Transform(object):

    def __init__( self ):
        pass

    def __call__( self, image ):
        return image


class FirstToSucced(Transform):

    def __init__(self, *steps):
        self.steps = steps
        super().__init__()

    def __call__(self, image):
        result = None
        for s in self.steps:
            result = s(image)
            
            if result is not None:
                break

        return image


class RandomizedTransform(Transform):

    def __init__( self, probability, transform ):
        self.probability = probability
        self.transform   = transform
        super(RandomizedTransform, self).__init__()

    def __call__( self, image ):
        if self.probability < uniform(0., 1., 1):
            image = self.transform(image)
        return image


class TransformFromITKFilter(Transform):

    def __init__( self, itk_filter ):
        self.flt = itk_filter
        super(TransformFromITKFilter, self).__init__()

    def __call__( self, image ):
        return self.flt.Execute(image)


class TransformFromNumpyFunction(Transform):

    def __init__( self, np_function ):
        self.flt = np_function
        super(TransformFromNumpyFunction, self).__init__()

    def __call__( self, image ):
        arr = sitk.GetArrayFromImage(image)
        arr = self.flt(arr)
        return sitk.GetImageFromArray(arr)
    

class PadTo(Transform):

    def __init__( self, target_shape, cval=0. ):
        self.target_shape = target_shape
        self.cval         = cval
        super(PadAndCropTo, self).__init__()

    def __call__( self, image ):
        # padding
        shape        = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        pad          = [max(s - t, 0) for t, s in zip(shape, target_shape)]
        lo_bound     = [int(floor(p / 2)) for p in pad]
        up_bound     = [int(ceil(p / 2)) for p in pad]
        image        = sitk.ConstantPad(image, lo_bound, up_bound, self.cval)

        return image
    

class PadAndCropTo(Transform):

    def __init__( self, target_shape, cval=0. ):
        self.target_shape = target_shape
        self.cval         = cval
        super(PadAndCropTo, self).__init__()

    def __call__( self, image ):

        # padding
        shape        = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        pad          = [max(s - t, 0) for t, s in zip(shape, target_shape)]
        lo_bound     = [int(floor(p / 2)) for p in pad]
        up_bound     = [int(ceil(p / 2)) for p in pad]
        image        = sitk.ConstantPad(image, lo_bound, up_bound, self.cval)

        # cropping
        shape        = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        crop         = [max(t - s, 0) for t, s in zip(shape, target_shape)]
        lo_bound     = [int(floor(c / 2)) for c in crop]
        up_bound     = [int(ceil(c / 2)) for c in crop]
        image        = sitk.Crop(image, lo_bound, up_bound)

        return image


class Resample(Transform):

    def __init__( self, spacing=1., interpolator=sitk.sitkLinear ):
        self.spacing      = spacing
        self.interpolator = interpolator
        self.flt          = sitk.ResampleImageFilter()
        super(Resample, self).__init__()

    def __call__( self, image ):
        spacing     = self.spacing
        if not isinstance(spacing, list):
            spacing = [spacing, ] * 3
        self.flt.SetReferenceImage(image)
        self.flt.SetOutputSpacing(spacing)
        self.flt.SetInterpolator(self.interpolator)
        s0 = int(round((image.GetSize()[0] * image.GetSpacing()[0]) / spacing[0], 0))
        s1 = int(round((image.GetSize()[1] * image.GetSpacing()[1]) / spacing[1], 0))
        s2 = int(round((image.GetSize()[2] * image.GetSpacing()[2]) / spacing[2], 0))
        self.flt.SetSize([s0, s1, s2])
        return self.flt.Execute(image)


class ZeroOneScaling(Transform):

    def __init__(self):
        self.minmax_flt = sitk.MinimumMaximumImageFilter()
        self.cast_flt   = sitk.CastImageFilter()
        self.cast_flt.SetOutputPixelType(sitk.sitkFloat32)
        super(ZeroOneScaling, self).__init__()

    def __call__(self, image):
        image   = self.cast_flt.Execute(image)
        # get min and max
        self.minmax_flt.Execute(image)
        minimum = self.minmax_flt.GetMinimum()
        maximum = self.minmax_flt.GetMaximum()
        image   = (image - minimum)/(maximum - minimum)
        return image


class ToNumpyArray(Transform):

    def __init__(self, add_batch_dim=False, add_singleton_dim=False, channel_second=False):
        self.add_batch_dim     = add_batch_dim
        self.add_singleton_dim = add_singleton_dim
        self.channel_second    = channel_second
        super(ToNumpyArray, self).__init__()

    def __call__(self, image):
        image     = sitk.GetArrayFromImage(image)

        if self.add_batch_dim:
            image = image[None, ...]
        if self.add_singleton_dim:
            if self.channel_second:
                image = image[:, None, ...]
            else:
                image = image[..., None]
        return image

