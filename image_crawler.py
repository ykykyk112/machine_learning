from tkinter import *
from tkinter import filedialog
import tkinter.font
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from urllib.request import urlretrieve
import time
import os

# Selenium을 이용한 이미지 크롤러 코드 클래스화 한다.
# GUI에서 다운로드를 재실행 할 경우에 소멸자를 이용해 인스턴스를 release 시켜주기 위함이다.
# 코드 가독성을 향상시키기 위함이다.

chrome_path = 'C:\\Users\ykyky\python_code\ML\chromedriver_win32\chromedriver.exe'

class Crawler_GUI() :
    # Constructor -> tkinter 객체 생성
    def __init__(self) :
        self.root = tkinter.Tk()
        self.root.title('Web Crawler')
        self.root.geometry('480x320+600+200')
        self.root.resizable(False, False)
        self.crawler = Crawler(self)

        # Image label 생성
        icon_naver = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\\naver.jpg').subsample(26)
        icon_google = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\google.jpg').subsample(25)
        icon_daum = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\daum.jpg').subsample(18)
        icon_download = PhotoImage(file = 'C:\\Users\ykyky\python_code\ML\img_src\download.jpg').subsample(12)

        # status frame 생성
        self.status_frame = LabelFrame(self.root, text = '상태 창')
        self.status_frame.pack(fill = 'both', side = 'top', padx = 2, pady = 2)

        self.scrollbar_yaxis = Scrollbar(self.status_frame)
        self.scrollbar_xaxis = Scrollbar(self.status_frame, orient = 'horizontal')
        self.scrollbar_yaxis.pack(side = 'right', fill = 'y')
        self.scrollbar_xaxis.pack(side = 'bottom', fill = 'x')

        self.status_message = Listbox(self.status_frame, selectmode = 'extended', yscrollcommand = self.scrollbar_yaxis.set, xscrollcommand = self.scrollbar_xaxis.set)
        self.status_message.pack(side = 'left', fill= 'both', expand = True)

        self.scrollbar_yaxis.config(command = self.status_message.yview)
        self.scrollbar_xaxis.config(command = self.status_message.xview)

        # download path frame 생성
        self.folder_selected = None
        self.path_frame = LabelFrame(self.root, text = '다운로드 경로')
        self.path_frame.pack(fill = 'both', padx = 2, pady = 2)

        self.txt_path = Entry(self.path_frame)
        self.txt_path.pack(side = 'left', fill = 'x', expand = True, ipady = 2, padx = 3)

        self.btn_path = Button(self.path_frame, text = '찾아보기', command = self.set_download_path)
        self.btn_path.pack(side = 'right')
        
        # website select frame 생성
        self.website_select_frame = LabelFrame(self.root, text = '웹사이트', height = 10)
        self.website_select_frame.pack(side = 'left', fill = 'both', expand = True, padx = 2, pady = 2)

        self.website_selected = StringVar()
        self.website_naver = Radiobutton(self.website_select_frame, image = icon_naver, val = 'naver', variable = self.website_selected)
        self.website_google = Radiobutton(self.website_select_frame, image = icon_google, val = 'google', variable = self.website_selected)
        self.website_daum = Radiobutton(self.website_select_frame, image = icon_daum, val = 'daum', variable = self.website_selected)

        self.website_naver.pack(side = 'left')
        self.website_google.pack(side = 'left')
        self.website_daum.pack(side = 'left')
        self.website_naver.select()

        # download button frame 생성
        self.download_frame = LabelFrame(self.root, text = '다운로드')
        self.download_frame.pack(side = 'right', fill = 'both', expand = True, padx = 2, pady = 2)

        self.download_keyword = Entry(self.download_frame)
        self.download_keyword.pack(side = 'left', fill = 'x', expand = True, ipady = 2, padx = 10)

        self.download = Button(self.download_frame, image = icon_download, command = self.btn_download)
        self.download.pack(side = 'right', expand = True)

        # GUI Program 유지
        self.root.mainloop()

    def __del__(self) :
        pass

    # GUI에서 download_path 지정을 위한 멤버 함수 지정
    def set_download_path(self) :
        # download_path 지정
        self.folder_selected = filedialog.askdirectory()
        if self.folder_selected is None :
            return
        
        # path_frame에 download_path 출력
        self.txt_path.delete(0, END)
        self.txt_path.insert(0, self.folder_selected)

    def btn_download(self) :
        # 폴더 선택 및 검색 키워드에 대한 예외 처리 구문
        # 폴더가 선택되지 않았거나 키워드가 입력되지 않았으면 해당 행동을 지시하는 이벤트 창 출력 이후 return
        self.keyword = self.download_keyword.get()
        if self.folder_selected is None :
            pass
        if self.keyword == '' :
            pass
        self.crawler.downloadImage(keyword = self.keyword, website = self.website_selected.get(), path = self.folder_selected)


class Crawler() :
    def __init__(self, gui) :
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option('excludeSwitches', ['enable-logging'])
        self.options.add_argument('headless')
        print("Initialize Complete")
        self.gui = gui

    def __del__(self) :
        print("Destruction Called")

    def downloadImage(self, keyword, website, path) :
        # make webdriver
        self.drivers = webdriver.Chrome(chrome_path, options=self.options)
        self.website = website
        self.keyword = keyword
        self.URL = self.getURL(website)
        self.className = self.getClassName(website)
        self.folder_dataset = path + '/Dataset'
        self.download_path = self.folder_dataset + '/{}'.format(self.keyword)

        # ListBox 초기화
        self.gui.status_message.delete(0, self.gui.status_message.size())

        self.gui.status_message.insert(END, '웹 브라우저 설정...')

        if not os.path.isdir(self.folder_dataset) :
            os.mkdir(self.folder_dataset)
        if not os.path.isdir(self.download_path) :
            os.mkdir(self.download_path)
        
        self.gui.status_message.insert(END, '[폴더 생성] ' + self.download_path)
        time.sleep(0.2)
        self.gui.root.update()
        self.gui.status_message.insert(END, '[검색어] ' + self.keyword)
        self.gui.root.update()
        self.gui.status_message.insert(END, '[검색 엔진] ' + self.website)
        self.gui.root.update()

        self.drivers.get(self.URL)
        # 전체화면이 아닐 경우에 elements가 찾아지지 않는 이벤트 발생
        self.drivers.maximize_window()

        self.gui.status_message.insert(END, '다운로드 목록 생성중...')
        self.gui.root.update()
        
        # 웹 브라우저에서 Page Down Key를 눌러서 크롤링 할 데이터 개수 설정
        body = self.drivers.find_element_by_css_selector('body')
        for i in range(10) :
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.4)

        elements = self.drivers.find_elements_by_css_selector(self.className)

        self.image_links = []

        for e in elements :
            link = e.get_attribute('src')
            if 'http' or 'data' in link and 'png' or 'PNG' not in link :
                self.image_links.append(link)

        total_links_len = len(self.image_links)
        num_download_image = 0
        for index, link in enumerate(self.image_links) :
            filename = '{0}/{1}{2:03d}{3}'.format(self.download_path, self.keyword, index+1, '.jpg')
            try : 
                urlretrieve(url=link, filename=filename)
                num_download_image += 1
            except :
                pass
            if index == 0 :
                self.gui.status_message.insert(END, '[' + '{:.1f}%'.format((index+1)/total_links_len*100) + ']' + ' 다운로드 진행 중')
                self.gui.root.update()
                time.sleep(0.05)
            else :
                self.gui.status_message.delete(END, END)
                self.gui.status_message.insert(END, '[' + '{:.1f}%'.format((index+1)/total_links_len*100) + ']' + ' 다운로드 진행 중')
                self.gui.root.update()
                time.sleep(0.05)

        
        self.gui.status_message.insert(END, '[모든 이미지 다운로드 완료] {}장'.format(num_download_image))
        self.gui.root.update()
        time.sleep(0.2)
        self.gui.status_message.insert(END, '웹 브라우저 종료...')
        self.gui.root.update()
        self.drivers.quit()

    def getURL(self, website) :
        return {
            'naver' : 'https://search.naver.com/search.naver?where=image&query={}'.format(self.keyword),
            'google' : 'https://www.google.com/search?q={}&tbm=isch'.format(self.keyword),
            'daum' : 'https://search.daum.net/search?w=img&nil_search=btn&enc=utf8&q={}'.format(self.keyword)
        }.get(website, 'https://search.naver.com/search.naver?where=image&query={}'.format(self.keyword))

    def getClassName(self, website) :
        return {
            'naver' : 'img._image',
            'google' : 'img.rg_i',
            'daum' : 'img.thumb_img'
        }.get(website, 'img._image')


root = Crawler_GUI()
