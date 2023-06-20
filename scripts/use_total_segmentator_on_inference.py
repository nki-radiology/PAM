from os import listdir, system
from os.path import join

from SimpleITK import ReadImage, WriteImage

root = r'/data/groups/beets-tan/l.estacio/data_tcia/train/'
filelist = listdir(root)

for i, filename in enumerate(filelist):
    # read and convert to nrrd
    fullpath = join(root, filename)
    im = ReadImage(fullpath)
    WriteImage(im, 'input.nii')

    print('processing', str(i), 'out of', len(filelist), '\t', filename)

    # run total segmentor
    cmd = 'TotalSegmentator --ml -i input.nii -o output'
    try:
        system(cmd)
    except:
        print('something went wrong with ' + filename)
        continue

    # compress output
    output_filename = filename.replace('.nrrd', '_seg.nii.gz')
    im = ReadImage('output.nii')
    fullpath = join('segmentations', output_filename)
    WriteImage(im, fullpath)