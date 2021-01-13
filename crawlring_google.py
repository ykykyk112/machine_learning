import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

keyword = list(input().split())
query = ""
for index, key in enumerate(keyword) :
    if index != len(keyword)-1 :
        query += key
        query += "+"
    else :
        query += key
print(query)
URL = 'https://www.google.com/search?q={0}&tbm=isch&oq={1}'.format(query, query)
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(r'C:\Users\ykyky\python_code\chromedriver_win32\chromedriver', options=options)
driver.implicitly_wait(3)
driver.maximize_window()

driver.get(url=URL)

driver.find_element_by_xpath('//*[@id="gb"]/div/div[1]/a').click()
driver.find_element_by_name('identifier').send_keys('ykykyk112@naver.com')
driver.find_element_by_xpath('//*[@id="identifierNext"]/div/button/div[2]').click()

imgs = driver.find_elements_by_css_selector('img.rg_i')

links = []
for img in imgs :
     link = img.get_attribute('src')
     if 'data' in link :
         links.append(link)

print(len(links))




time.sleep(60)
driver.quit()