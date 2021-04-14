'''
      Function Implementation
        
        - Function that convert orthogonal coordinates 2d-image to sinogram coordinates image

      About Input

        - image : 2d CT image, which is appied linear transformation and windowing. (type : numpy.ndarray)

        - theta : Range of degrees between x-axis and line, which projected 2d CT image. (type : numpy.ndarray)

        - transpose : Option that decide to operate transpose on return sinogram. (type : bool)

      About Output

        - sinogram : Sinogram converted from input argument image. (type : numpy.ndarray)
'''

def to_sinogram(image, theta, transpose = True):
    sinogram = radon(image, theta)
    if transpose :
        return sinogram.T
    else :
        return sinogram


'''
      Function Implementation
        
        Function that operates masking on sinogram image, which is input argument. 
        With reference to row_pos, col_pos which is center position of region we want to mask and degrees used in radon transformation, mask area on sinogram image.
        Argument radius is value that means radius of area and padding is value that means pixel value to expand region of masking on sinogram. 

      About Input

        - image : 2d CT image, which is appied linear transformation and windowing. (type : numpy.ndarray)

        - row_pos : Pixel index of y_axis from center position of region of interest to masking. (type : int)

        - col_pos : Pixel index of x_axis from center position of region of interest to masking. (type : int)

        - degrees : Group of degree we use on radon transformation. Also mean quantized value of y_axis in sinogram image. (type : numpy.ndarray)

        - radius : Radius of region of we interest to masking. (type : int)

        - padding : Value to expand extra pixel on masked sinogram image. (type : int)

      About Output

        - sinogram : Masked sinogram image. (type : numpy.ndarray)
'''

def mask_sinogram(image, row_pos, col_pos, degrees, radius, padding):
    
    # Compute all parameters using get position on sinogram coordinate system
    row_origin = (image.shape[1]/2)-1
    col_origin = (image.shape[0]/2)-1
    x, y = (col_pos-col_origin), (row_origin-row_pos)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    # Protect divided by zero in arccosine function
    if r == 0. or r == 0: r = 1e-9
    theta = np.degrees(np.arccos(x/r))
    if y < 0 : theta = theta * (-1)
    
    # Get sinogram image using CT image and degrees, which y-axis value in sinogram coordinate system
    sinogram = to_sinogram(image, degrees, True)
    
    # Get sinogram coordinate system's x-axis value 
    sinogram_x = np.empty((degrees.shape))
    for idx, d in enumerate(degrees) :
        t = theta-d
        t_rad = np.radians(t)
        sinogram_x[idx] = r * np.cos(t_rad)
    sinogram_x = np.round(sinogram_x) + row_origin

    # Masking target area using sinogram x-axis value we computed above
    for idx, s_x in enumerate(sinogram_x) :
        sinogram[idx][0:int(s_x)-radius-padding] = 0.
        sinogram[idx][int(s_x)+radius+padding:image.shape[0]-1] = 0.
    
    return sinogram