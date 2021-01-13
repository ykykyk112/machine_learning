from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from urllib.request import urlretrieve
import time
import os

keyword = input("검색어를 입력하세요 : ")

URL = 'https://search.naver.com/search.naver?where=image&query={}'.format(keyword)

# Option 설정
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])

# Chrome Driver 설정
drivers = webdriver.Chrome('chromedriver_win32\chromedriver.exe', options=options)
drivers.implicitly_wait(3)

drivers.get(url=URL)
drivers.maximize_window()

# 웹 브라우저에서 Page Down Key를 눌러서 크롤링 할 데이터 개수 설정
body = drivers.find_element_by_css_selector('body')
for i in range(5) :
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

elements = drivers.find_elements_by_css_selector('img._image')

print('[폴더 생성]')
if not os.path.isdir('Dataset') :
    os.mkdir('Dataset')
if not os.path.isdir('Dataset\{}'.format(keyword)) :
    os.mkdir('Dataset\{}'.format(keyword))

links = []
for e in elements :
    link = e.get_attribute('src')
    if 'http' in link :
        links.append(link)

print('[다운로드 시작]')
for index, link in tqdm(enumerate(links), total= len(links)) :
    filename = './Dataset/{0}/{0}{1:03d}{2}'.format(keyword, index, '.jpg')
    urlretrieve(url=link, filename=filename)

print('[다운로드 종료], total {} image'.format(len(links)))

time.sleep(5)
drivers.quit()