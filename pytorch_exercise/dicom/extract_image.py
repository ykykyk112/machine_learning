import pydicom
import numpy as np

'''
    - Implementation : Function that extract CT image from DICOM file, by reference Window Center and Window Width
    - Input : Dicom file, Window Center value, Window Width value
    - Output : Image file of numpy array
'''

def get_ct_image(dicom_file, window_center, window_width):
    image = dicom_file.pixel_array
    
    # Linear Transformation
    slope = dicom_file.RescaleSlope
    bias = dicom_file.RescaleIntercept
    image = slope * image + bias
    
    # Windowing using Window Center, Window Width and Rescaling in range zero to one
    min_value = window_center - (window_width/2.)
    max_value = window_center + (window_width/2.)
    image_w = np.clip(image, min_value, max_value)
    image_w = (image_w - min_value) / (max_value-min_value)
    
    return image_w*(-1)