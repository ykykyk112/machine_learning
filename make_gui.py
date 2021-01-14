from tkinter import *
from tkinter import filedialog
import tkinter.font
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from urllib.request import urlretrieve
import time
import os

root = Tk()
root.title('Web Crawler')
root.geometry('480x320')
root.resizable(False, False)
URL = ''
className = ''
keyword = '장미'
download_path = ''
folder_selected = ''

def btn_download() :
    global keyword
    global URL
    global className
    folder_dataset = folder_selected + '/Dataset'
    folder_keyword = folder_dataset + '/{}'.format(keyword)
    download_path = folder_selected + '/' + folder_keyword
    if not os.path.isdir(folder_dataset) :
        os.mkdir(folder_dataset)
    if not os.path.isdir(folder_keyword) :
        os.mkdir(folder_keyword)
    status_message.insert(0, '[폴더 생성] ' + download_path)

    URL = getURL(website_value.get()).format(keyword)
    className = getClassName(website_value.get())
    # Option 설정
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('headless')

    # Chrome Driver 설정
    drivers = webdriver.Chrome('chromedriver_win32\chromedriver.exe', options=options)
    drivers.implicitly_wait(1)

    drivers.get(url=URL)
    drivers.maximize_window()

    time.sleep(3)
    drivers.quit()

def getURL(browser_name) :
    return {
        'naver' : 'https://search.naver.com/search.naver?where=image&query={}',
        'google' : 'https://www.google.com/search?q={}&tbm=isch',
        'daum' : 'https://search.daum.net/search?w=img&nil_search=btn&enc=utf8&q={}'
    }.get(browser_name, 'https://search.naver.com/search.naver?where=image&query={}')

def getClassName(browser_name) :
    return {
        'naver' : 'img._image',
        'google' : 'img.rg_i',
        'daum' : 'img.thumb_img'
    }.get(browser_name, 'img._image')

def set_download_path() :
    global folder_selected
    folder_selected = filedialog.askdirectory()
    if folder_selected is None :
         return
    txt_path.delete(0, END)
    txt_path.insert(0, folder_selected)


# image label 생성
icon_naver = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\\naver.jpg').subsample(26)
icon_google = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\google.jpg').subsample(25)
icon_daum = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\daum.jpg').subsample(18)
icon_download = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\download.jpg').subsample(12)


# text label 생성
website_font = tkinter.font.Font(family = "맑은 고딕", size = 15)


# status frame 생성
status_frame = LabelFrame(root, text = '상태 창')
status_frame.pack(fill = 'both', side = 'top', padx = 2, pady = 2)

scrollbar_yaxis = Scrollbar(status_frame)
scrollbar_xaxis = Scrollbar(status_frame, orient = 'horizontal')
scrollbar_yaxis.pack(side = 'right', fill = 'y')
scrollbar_xaxis.pack(side = 'bottom', fill = 'x')

status_message = Listbox(status_frame, selectmode = 'extended', yscrollcommand = scrollbar_yaxis.set, xscrollcommand = scrollbar_xaxis.set)
status_message.pack(side = 'left', fill= 'both', expand = True)

scrollbar_yaxis.config(command = status_message.yview)
scrollbar_xaxis.config(command = status_message.xview)

# download path frame 생성
path_frame = LabelFrame(root, text = '다운로드 경로')
path_frame.pack(fill = 'both', padx = 2, pady = 2)

txt_path = Entry(path_frame)
txt_path.pack(side = 'left', fill = 'x', expand = True, ipady = 2, padx = 3)

btn_path = Button(path_frame, text = '찾아보기', command = set_download_path)
btn_path.pack(side = 'right')

# website select frame 생성
website_select_frame = LabelFrame(root, text = '웹사이트', height = 10)
website_select_frame.pack(side = 'left', fill = 'both', expand = True, padx = 2, pady = 2)

website_value = StringVar()
website_naver = Radiobutton(website_select_frame, image = icon_naver, val = 'naver', variable = website_value)
website_google = Radiobutton(website_select_frame, image = icon_google, val = 'google', variable = website_value)
website_daum = Radiobutton(website_select_frame, image = icon_daum, val = 'daum', variable = website_value)

website_naver.pack(side = 'left')
website_google.pack(side = 'left')
website_daum.pack(side = 'left')
website_naver.select()

# download frame 생성
download_frame = LabelFrame(root, text = '다운로드')
download_frame.pack(side = 'right', fill = 'both', expand = True, padx = 2, pady = 2)

download = Button(download_frame, image = icon_download, command = btn_download)
download.pack()



root.mainloop()