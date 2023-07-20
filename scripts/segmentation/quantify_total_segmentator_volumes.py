import SimpleITK as sitk
import os 
import pandas as pd

def load_segmentation(segmentation_path):
    im = sitk.ReadImage(segmentation_path)
    return im


def compute_volumes(segmentation):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(segmentation)
    volumes = dict()
    for label in stats.GetLabels():
        volumes['Volume_' + str(label)] = stats.GetPhysicalSize(label)
    return volumes


def is_label_on_bondary(segmentation):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(segmentation)
    result = dict()
    for label in stats.GetLabels():
        result['IsOnBorder_' + str(label)] = stats.GetPerimeterOnBorder(label) != 0
    return result


def get_bounding_box(segmentation):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(segmentation)
    result = dict()
    for label in stats.GetLabels():
        result['BoundingBox_' + str(label)] = stats.GetBoundingBox(label)
    return result

df = []

for i, filename in enumerate(os.listdir('segmentations')):
    print(i, end='.')
    segmentation_path = os.path.join('segmentations', filename)
    segmentation = load_segmentation(segmentation_path)

    entry = dict()
    entry['filename'] = filename
    entry.update(compute_volumes(segmentation))
    entry.update(is_label_on_bondary(segmentation))
    entry.update(get_bounding_box(segmentation))
    
    df.append(entry)

pd.DataFrame(df).to_csv('segmentation_volumes.csv', index=False)
