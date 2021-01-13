from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import time

keyword = input("데이터셋을 입력하세요 : ")

URL = "https://search.naver.com/search.naver?where=image&sm=tab_jum&query={}".format(keyword)

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])

driver = webdriver.Chrome('chromedriver_win32\chromedriver.exe', options=options)
driver.implicitly_wait(3)
driver.get(url=URL)
driver.maximize_window()

body = driver.find_element_by_css_selector('body')
for i in range(5) :
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.5)

elements = driver.find_elements_by_css_selector('img._image')

lists = []
for elem in elements :
    e = elem.get_attribute('src')
    if 'http' in e :
        lists.append(e)

if not os.path.isdir('./{}'.format(keyword)) :
    os.mkdir('./{}'.format(keyword))

print('[폴더생성]')

print('[다운로드 시작]')

for index, l in tqdm(enumerate(lists), total = len(lists)) :
    filetype = '.jpg'
    filename = "./{0}/{0}{1:03d}{2}".format(keyword, index, filetype)
    urlretrieve(l, filename = filename)

print('[다운로드 완료]')

time.sleep(3)
driver.quit()