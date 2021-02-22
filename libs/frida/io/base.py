import SimpleITK as sitk


class ImageLoader(object):

    def __init__( self, *steps ):

        self.steps = steps
        super(ImageLoader, self).__init__()

    def __call__( self, inputs ):

        for s in self.steps:
            inputs = s(inputs)

        return inputs


class Read(object):

    def __call__( self, filename ):
        pass


class ReadVolume(Read):

    def __call__( self, filename ):

        try:
            image = sitk.ReadImage(filename)
            return image
        except:
            return None


class ReadDICOM(Read):

    def __call__( self, filename ):

        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filename)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            return image
        except:
            return None