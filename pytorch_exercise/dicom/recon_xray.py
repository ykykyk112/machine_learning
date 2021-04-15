import cv2
import numpy as np

'''
      Function Implementation
        
        

      About Input

        - dicom_files : Set of dicom_files we use for convert ct images to x-ray image. (type : list)

        - window_center : Median value of housefield unit range we want to reference on converted x-ray image. (type : int)

        - window_width : Length of range of housefield unit we want to reference on converted x-ray image. (type : int)

        - thickness : It is z-axis value on single ct image, so referenced by y-axis pixel value about one converted ct image slice on x-ray image

        - axis(optional) : Standard axis referenced with projection, default axis is 0. (type : int)

        - rotate_degree(optional) : When you set on this value between 0' ~ 360', ct images is rotated as much rotate_degree you set and then pixel in image will be projected. (type : int)

      About Output

        - x_ray : Converted x-ray image, layered set of slices of ct image. (type : numpy.ndarray)
'''

def recon_xray(dicom_files, window_center, window_width, thickness, axis = 0, rotate_degree = None):
    
    IMAGE_W, IMAGE_H = 512, 512
    
    count = 1
    dicom_images = np.empty((len(dicom_files), IMAGE_H, IMAGE_W))
    
    if rotate_degree == None :
        for idx, f in enumerate(dicom_files) :
            if f.InstanceNumber != count :
                print('Not sorted! /', f.InstanceNumber)
                return
            count += 1
            dicom_images[idx] = get_ct_image(f, window_center, window_width)
    else :
        cX, cY = (IMAGE_W/2)-1, (IMAGE_H/2)-1
        for idx, f in enumerate(dicom_files) :
            if f.InstanceNumber != count :
                print('Not sorted! /', f.InstanceNumber)
                return
            count += 1
            image = get_ct_image(f, window_center, window_width)
            M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
            rot_image = cv2.warpAffine(image, M, (w, h))
            dicom_images[idx] = rot_image
        
    x_ray = np.empty((len(dicom_files)*thickness, 512))
    for idx, i in enumerate(dicom_images) :
        x_ray[thickness*idx:thickness*idx+thickness] = np.mean(i, axis = axis)
    return x_ray