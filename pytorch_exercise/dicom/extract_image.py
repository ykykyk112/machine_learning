import pydicom
import numpy as np

'''
      Function Implementation
        
        

      About Input

        - dicom_file : File contains ct image array, imformation of ct scan and patient and tomography equipment specification. (type : pydicom.dataset.FileDataset)

        - window_center : Median value of housefield unit range we want to reference on ct image. (type : int)

        - window_width : Length of range of housefield unit we want to reference on ct image. (type : int)

      About Output

        - image_w : Extracted ct image in dicom_file. (type : numpy.ndarray)
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