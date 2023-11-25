# 한국어 수어 사전 데이터 구축

import re
import urllib
from urllib.request import urlopen, Request

import requests
import pandas as pd
from bs4 import BeautifulSoup
import collections
from tqdm import tqdm

url = "https://sldict.korean.go.kr/front/sign/signList.do"

catetory_code = ['CTE00'+str(i) for i in range(1, 17)]

#html = requests.get(url).text
#excparam =

# 1 page url
# https://sldict.korean.go.kr/front/sign/signList.do?current_pos_index=&origin_no=0&searchWay=&top_category=&category=CTE001&detailCategory=&searchKeyword=&pageIndex=1&pageJumpIndex=
# 2 page url
# https://sldict.korean.go.kr/front/sign/signList.do?current_pos_index=&origin_no=0&searchWay=&top_category=&category=CTE001&detailCategory=&searchKeyword=&pageIndex=2&pageJumpIndex=


print()
url = 'https://sldict.korean.go.kr/front/sign/signList.do?current_pos_index=&origin_no=0&searchWay=&top_category=&category={}&detailCategory=&searchKeyword=&pageIndex={}&pageJumpIndex='

ksl_cat_dict = collections.defaultdict()

for cat_cd in tqdm(catetory_code):
    # category별로 ex.일상
    ksl_words = []  # ksl 담을 리스트
    for page_num in tqdm(range(1, 34)):
        # category 안에서 페이지별로
        response = requests.get(url.format(cat_cd, page_num))
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        li_words = soup.select(".tit")

        for words in li_words:
            ksl_words.append(re.sub(r"[^ㄱ-ㅣ가-힣,]", '', words.text))

        print('category가 ', cat_cd, '일 때, 수어 단어 리스트들 :' , ksl_words)

        ksl_cat_dict[cat_cd] = ksl_words


