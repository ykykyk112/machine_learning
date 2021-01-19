from tkinter import *
from tkinter import filedialog
import tkinter.font
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from urllib.request import urlretrieve
import time
import os



def btn_download() :

    # ListBox 초기화
    status_message.delete(0, status_message.size())

    # Option 설정
    status_message.insert(END, '웹 브라우저 설정...')
    root.update()
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('headless')

    # Chrome Driver 설정
    drivers = webdriver.Chrome('C:\\Users\ykyky\python_code\ML\chromedriver_win32\chromedriver.exe', options=options)

    keyword = download_keyword.get()
    website = website_value.get()
    URL = getURL(website).format(keyword)
    className = getClassName(website)

    folder_dataset = folder_selected + '/Dataset'
    download_path = folder_dataset + '/{}'.format(keyword)

    if not os.path.isdir(folder_dataset) :
        os.mkdir(folder_dataset)
    if not os.path.isdir(download_path) :
        os.mkdir(download_path)
    
    status_message.insert(END, '[폴더 생성] ' + download_path)
    time.sleep(0.2)
    root.update()
    status_message.insert(END, '[검색어] ' + keyword)
    root.update()
    status_message.insert(END, '[검색 엔진] ' + website)
    root.update()

    
    drivers.get(URL)

    # 전체화면이 아닐 경우에 elements가 찾아지지 않는 이벤트 발생
    drivers.maximize_window()

    status_message.insert(END, '다운로드 목록 생성중...')
    root.update()

    # 웹 브라우저에서 Page Down Key를 눌러서 크롤링 할 데이터 개수 설정
    body = drivers.find_element_by_css_selector('body')
    for i in range(13) :
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    elements = drivers.find_elements_by_css_selector(className)

    links = []

    for e in elements :
        link = e.get_attribute('src')
        if 'http' or 'data' in link and 'png' or 'PNG' not in link :
            links.append(link)

    total_links_len = len(links)
    num_download_image = 0
    for index, link in enumerate(links) :
        filename = '{0}/{1}{2:03d}{3}'.format(download_path, keyword, index+1, '.jpg')
        try : 
            urlretrieve(url=link, filename=filename)
            num_download_image += 1
        except :
            pass
        if index == 0 :
            status_message.insert(END, '[' + '{:.1f}%'.format((index+1)/total_links_len*100) + ']' + ' 다운로드 진행 중')
            root.update()
            time.sleep(0.05)
        else :
            status_message.delete(END, END)
            status_message.insert(END, '[' + '{:.1f}%'.format((index+1)/total_links_len*100) + ']' + ' 다운로드 진행 중')
            root.update()
            time.sleep(0.05)

    
    status_message.insert(END, '[모든 이미지 다운로드 완료] {}장'.format(num_download_image))
    root.update()
    time.sleep(0.2)
    status_message.insert(END, '웹 브라우저 종료...')
    root.update()
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

root = Tk()
root.title('Web Crawler')
root.geometry('480x320')
root.resizable(False, False)
URL = ''
className = ''
folder_selected = ''

# image label 생성
icon_naver = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\\naver.jpg').subsample(26)
icon_google = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\google.jpg').subsample(25)
icon_daum = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\daum.jpg').subsample(18)
icon_download = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\download.jpg').subsample(12)

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

# download button frame 생성
download_frame = LabelFrame(root, text = '다운로드')
download_frame.pack(side = 'right', fill = 'both', expand = True, padx = 2, pady = 2)

download_keyword = Entry(download_frame)
download_keyword.pack(side = 'left', fill = 'x', expand = True, ipady = 2, padx = 10)

download = Button(download_frame, image = icon_download, command = btn_download)
download.pack(side = 'right', expand = True)


root.mainloop()