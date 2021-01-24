from PIL import Image
import os

# 현재 파일이 존재하는 폴더임 -> 이름을 바꿀 데이터셋의 폴더로 이동
now_path = os.path.dirname(os.path.realpath(__file__))
download_path = 'C:\\Users\ykyky\python_code\ML\sklearn_exercise\Training_set_rename'

fruit_name = ['Avocado', 'Banana', 'Blueberry', 'Chestnut', 'Corn', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Peach', 'Pear', 'Strawberry']

# make fruit folder
# for name in fruit_name :
#     os.mkdir(download_path+'\{}'.format(name))

# fruit_name = ['Avocado']
#[143 166 154 153 150 156 164 166 160 164 164 164]
#[427 490 462 450 450 466 492 490 479 492 492 492]

for name in fruit_name :
    i = 1
    fruit_folder = now_path + '\{}\\'.format(name)
    download_folder = download_path + '\{}\\'.format(name)

    for index in range(500) :
        try :
            file_path = fruit_folder + '{0}_100.jpg'.format(index)
            temp = Image.open(file_path)
            new_name = download_folder + '{:03d}.jpg'.format(i)
            temp.save(new_name)
            i += 1
        except :
            pass

    for index_2 in range(500) :
        try :
            file_path = fruit_folder+'r_{0}_100.jpg'.format(index_2)
            temp = Image.open(file_path)
            new_name = download_folder + '{:03d}.jpg'.format(i)
            temp.save(new_name)
            i += 1
        except :
            pass

    for index_3 in range(500) :
        try :
            file_path = fruit_folder+'r2_{0}_100.jpg'.format(index_3)
            temp = Image.open(file_path)
            new_name = download_folder + '{:03d}.jpg'.format(i)
            temp.save(new_name)
            i += 1
        except :
            pass

    print('total download - {0}, {1} file'.format(name, i))