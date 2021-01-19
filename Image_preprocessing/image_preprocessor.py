import cv2 as cv
import numpy as np

rose_shape = np.empty((100, 2))

for i in range(100) :
    path = 'Image_preprocessing\Dataset\\rose\\rose{:03d}.jpg'.format(i+1)
    rose = cv.imread(path, cv.IMREAD_GRAYSCALE)
    rose_shape[i] = rose.shape

width_mean = int(np.mean(rose_shape[:, :1]))
height_mean = int(np.mean(rose_shape[:, 1:]))

for i in range(100) :
    path = 'Image_preprocessing\Dataset\\rose\\rose{:03d}.jpg'.format(i+1)
    rose = cv.imread(path, cv.IMREAD_GRAYSCALE)
    rose_reshape = cv.resize(rose, (width_mean, height_mean))
    cv.imshow('original', rose)
    cv.imshow('resized', rose_reshape)
    key = cv.waitKey(0)
    cv.destroyAllWindows()
    if key == 27 :
        break

# path = 'Image_preprocessing\Dataset\\rose\\rose011.jpg'
# rose = cv.imread(path, cv.IMREAD_GRAYSCALE)

# rose_scaled = np.zeros((75, 75), dtype= rose.dtype)

# start_x = 0
# start_y = 0

# width_iter = rose.shape[0]//3
# height_iter = rose.shape[1]//3

# for i in range(width_iter) :
#     start_x = i*3
#     for j in range(height_iter) :
#         start_y = j*3
#         sub_image = rose[start_x:start_x+3, start_y:start_y+3]
#         rose_scaled[i][j] = np.mean(sub_image, dtype='uint8')

# rose_scaled = cv.resize(rose_scaled, (225, 225))
# cv.imshow('original', rose)
# cv.imshow('scaled', rose_scaled)
# cv.waitKey(0)
# cv.destroyAllWindows()
