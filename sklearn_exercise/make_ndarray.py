import cv2
import numpy as np

# google colab 상에서 동작하도록 작성된 함수입니다.

fruit_name = ['Avocado', 'Banana', 'Blueberry', 'Chestnut', 'Corn', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Peach', 'Pear', 'Strawberry']
path = '/content/drive/MyDrive/Colab_Notebooks/image/Test_set/'
nonetype = cv2.imread('to/detect/NoneType/class')

def make_ndarray(path, fruit_name) :
    first = True
    for name in fruit_name :
        folder = path + '{}/'.format(name)
        for i in range(300) :
            file_name = folder+'{:03d}.jpg'.format(i+1)
            if first :
                data_set = cv2.imread(file_name, cv2.IMREAD_COLOR)
                data_set = data_set/255
                data_set = data_set.reshape(1, 10000, 3)
                first = False
                if type(data_set) is nonetype :
                    print('return -1')
                    return
            else :
                image = cv2.imread(file_name, cv2.IMREAD_COLOR)
                if type(image) is nonetype :
                    print('{0} 종료'.format(name))
                    break
                image_scaled = image/255
                image_reshaped = image_scaled.reshape(1, 10000, 3)
                data_set = np.append(data_set, image_reshaped, axis = 0)
        print('for {0} - {1} complete'.format(name, i))
        print("data set's shape : {}".format(data_set.shape))
    return data_set

test_data = make_ndarray(path, fruit_name)