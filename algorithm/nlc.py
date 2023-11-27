from os import name
import sys
import urllib.request
import requests
import pprint
import xml.etree.ElementTree as et
import pandas as pd
import bs4
from lxml import html
from urllib.parse import urlencode, quote_plus, unquote
import datetime
from random import randint
import random

import requests
from selenium import webdriver   # selenium에서 사용할 웹 브라우더 인터페이스
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By              # Web elements에 접근하기 위함.
from selenium.webdriver.common.keys import Keys          # 키입력 이벤트 처리에 필요.

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager # 크롬 드라이버 자동 업데이트을 위한 모듈
import time

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')


# 사서추천도서 카테고리별 파싱 및 분할 저장 - 국립중앙도서관
def Recommended_books():
    url = 'https://nl.go.kr/NL/search/openApi/saseoApi.do'
    authKey = '32a7bd872ba05a0698093446565ea0df96c15701f1ff564645b4338559c5fefb'

    # datetime으로 최신 날짜로 수정해야 함
    params = {'key': authKey, 'startRowNumApi': 1, 'endRowNumApi': 1305, 'start_date': 20101201, 'end_date': 20231125}

    response = requests.get(url, params=params, verify=False)
    content = response.text

    xml_obj = bs4.BeautifulSoup(content, 'lxml-xml')
    item_list = xml_obj.findAll('item')
    drCode_list = xml_obj.findAll('drCode')

    # 각 카테고리별 데이터 리스트 만들기
    literature_list = []  # 분류번호 11: 문학
    HumanitiesSciences_list = []  # 분류번호 6: 인문과학
    socialScience_list = []  # 분류번호 5: 사회과학
    naturalScience_list = []  # 분류번호 4: 자연과학
    etc_list = []  # 그 외

    # 카테고리별로 나누기
    for i in range(0, len(item_list)):
        line = item_list[i].find_all()

        if drCode_list[i].text == '11':
            literature_list.append(line)
        elif drCode_list[i].text == '6':
            HumanitiesSciences_list.append(line)
        elif drCode_list[i].text == '5':
            socialScience_list.append(line)
        elif drCode_list[i].text == '4':
            naturalScience_list.append(line)
        else:
            etc_list.append(line)

    # 각 카테고리별 데이터 수집
    all_list = [literature_list, HumanitiesSciences_list, socialScience_list, naturalScience_list, etc_list]

    literature_df = pd.DataFrame()
    HumanitiesSciences_df = pd.DataFrame()
    socialScience_df = pd.DataFrame()
    naturalScience_df = pd.DataFrame()
    etc_df = pd.DataFrame()

    for i in range(0, len(all_list)):

        for j in range(0, len(all_list[i])):
            line = all_list[i][j]
            row_dict = {}

            # 데이터 수집
            for element in line:
                row_dict[element.name] = element.text

            df = pd.DataFrame([row_dict])
            if i == 0:
                literature_df = pd.concat([literature_df, df], ignore_index=True)
            elif i == 1:
                HumanitiesSciences_df = pd.concat([HumanitiesSciences_df, df], ignore_index=True)
            elif i == 2:
                socialScience_df = pd.concat([socialScience_df, df], ignore_index=True)
            elif i == 3:
                naturalScience_df = pd.concat([naturalScience_df, df], ignore_index=True)
            elif i == 4:
                etc_df = pd.concat([etc_df, df], ignore_index=True)

    return literature_df, HumanitiesSciences_df, socialScience_df, naturalScience_df, etc_df


# 사서추천도서의 카테고리에서 기준이 되는 사서추천도서와 동일 카테고리의 또 다른 도서 선정
def pick_randomBook(literature_df, HumanitiesSciences_df, socialScience_df, naturalScience_df, etc_df):
    drCode = 0  # 사서추천도서 카테고리 분류 번호

    # 분류번호 11: 문학
    # 분류번호 6: 인문과학
    # 분류번호 5: 사회과학
    # 분류번호 4: 자연과학

    # 각 사서추천도서 카테고리별 데이터 프레임에서 모든 isbn 번호를 리스트로 추출
    literature_isbn_list = literature_df['recomisbn'].to_list()
    HumanitiesSciences_isbn_list = HumanitiesSciences_df['recomisbn'].to_list()
    socialScience_isbn_list = socialScience_df['recomisbn'].to_list()
    naturalScience_isbn_list = naturalScience_df['recomisbn'].to_list()
    etc_isbn_list = etc_df['recomisbn'].to_list()

    # 모든 리스트를 하나의 리스트로 묶기
    all = [literature_isbn_list, HumanitiesSciences_isbn_list, socialScience_isbn_list, naturalScience_isbn_list,
           etc_isbn_list]

    # 사서추천도서를 추출한 임의의 카테고리 선정
    list_num = random.randint(0, 4)

    list = all[list_num]

    # 특정 카테고리에서 6개의 사서추천도서를 추출하여 하나는 도서 추천의 기준이 되는 사서추천도서로, 나머지는 추천할 도서가 없을 경우 동일 카테고리에서 추천하는 다른 도서로 사용
    while True:
        randomBooks = random.sample(list, 6)
        cnt = 0

        for isbn in randomBooks:
            if len(isbn) == 13:
                cnt += 1

        if cnt == 6:
            return randomBooks



# 사용자가 읽은 사서추천도서에 대한 키워드와 그에 해당하는 도서 검색 결과인 ISBN 리스트를 키워드 가중치가 높은 순으로 정렬하여 데이터 프레임 생성 (도서 검색 결과가 0건인 키워드는 제외) - 정보나루
# 데이터 프레임의 구조 : 키워드, 가중치, 키워드에 대한 도서 검색 결과인 ISBN 리스트
def keywordWeightDataRrame(isbn):
    # 하나의 사서추천도서에 대한 키워드 목록 조회 API - 정보나루
    url_1 = 'http://data4library.kr/api/keywordList'
    authKey = '8dc468abe0634fa4277278de1fc152cadff4d1e175afcda39925e3f61d853af5'
    params_1 = {'authKey': authKey, 'isbn13': isbn, 'additionalYN': 'N'}

    response = requests.get(url_1, params=params_1)
    content = response.text

    xml_obj = bs4.BeautifulSoup(content, 'lxml-xml')
    item_list = xml_obj.findAll('item')
    print(item_list)

    # 키워드와 그에 따른 가중치값 데이터프레임
    keywordWeight_df = pd.DataFrame()

    for i in range(0, len(item_list)):
        line = item_list[i]
        row_dict = {}

        # 데이터 수집
        for element in line:
            row_dict[element.name] = element.text

        print(row_dict)
        df = pd.DataFrame([row_dict])

        keywordWeight_df = pd.concat([keywordWeight_df, df], ignore_index=True)

    print("keywordWeight_df.columns:", keywordWeight_df.columns)
    print('')
    print(keywordWeight_df)
    keyword_list = keywordWeight_df['word'].to_list()

    # 50개의 키워드 각각 도서 목록 조회 API - 정보나루
    url_2 = 'http://data4library.kr/api/srchBooks'

    keywordWeight_df['isbn_list'] = None  # 데이터 프레임에 isbn_list 열 추가 및 초기화
    isbn_list = []  # 하나의 키워드에 대해 검색되는 도서의 ISBN 리스트 : 데이터 프레임의 각 알맞는 키워드가 위치한 행에 추가
    dIndex_list = []  # 삭제할 행의 인덱스 번호 리스트

    for i in range(0, len(keyword_list)):
        params_2 = {'authKey': authKey, 'pageNo': 1, 'pageSize': 8000, 'keyword': keyword_list[i]}

        try:
            response = requests.get(url_2, params=params_2, timeout=5)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 처리
            content = response.text

            xml_obj = bs4.BeautifulSoup(content, 'lxml-xml')
            isbn13_list = xml_obj.findAll('isbn13')

            for j in range(0, len(isbn13_list)):
                line = isbn13_list[j]
                isbn_list.append(line.text)

            # 특정 키워드의 도서 검색 결과가 0건이라면 데이터 프레임에서 해당 키워드 행 삭제
            if len(isbn_list) <= 0:
                dIndex_list.append(i)

            # 특정 키워드의 도서 검색 결과가 한 건이라도 있을 경우 데이터 프레임에 추가
            else:
                keywordWeight_df.at[i, 'isbn_list'] = isbn_list

            # 초기화
            isbn_list = []

        except requests.exceptions.RequestException as e:

            print(f"Error during API request: {e}")

    # 0건의 도서 검색 결과를 가진 키워드를 데이터 프레임에서 삭제
    keywordWeight_df.drop(dIndex_list, axis=0, inplace=True)

    print(keywordWeight_df)

    return keywordWeight_df


# 특정 사서추천도서의 상위 5개 키워드에 모두 해당하는 도서 찾기. 해당 도서의 ISBN을 리스트로 추출
def find_common_books(lists):
    # 첫 번째 리스트를 기준으로 설정
    base_list = set(lists[0])

    # 모든 리스트에서의 교집합을 찾음
    common_elements = [element for element in base_list if all(element in lst for lst in lists[1:])]

    return common_elements


# 연속된 5개의 키워드에 공통으로 해당되는 도서 추출 - 정보나루
def continuedKword_books(df, randomBooks):
    # 데이터 프레임에서 isbn_list 열을 하나의 리스트로 생성
    allIsbn_list = df['isbn_list'].to_list()

    # 데이터 프레임에서 recomisbn 열을 하나의 리스트로 생성
    word_list = df['word'].to_list()

    user_isbn = randomBooks[0]
    print('기준이 되는 사서추천도서:', user_isbn)

    index = 4  # allIsbn_list의 인덱스 번호

    # 공통 도서 추출 함수에 넘겨줄 리스트 집합
    transfer_isbn_list = [allIsbn_list[0], allIsbn_list[1], allIsbn_list[2], allIsbn_list[3], allIsbn_list[4]]

    # 5개의 키워드에 해당하는 도서의 ISBN 번호 추출 함수 호출
    related_books_isbn = find_common_books(transfer_isbn_list)
    print('처음 추출된 공통도서:', related_books_isbn)

    # 공통되는 도서 ISBN 목록에 그 기준이 되는 사서추천도서가 있다면 삭제하기
    if len(related_books_isbn) > 0:
        for i in range(0, len(related_books_isbn)):
            if (related_books_isbn[i] == user_isbn):
                del related_books_isbn[i]

        print('삭제 과정을 마친 공통도서:', related_books_isbn)


    # 삭제 과정이 끝나고 최종으로 추출이 마친 공통 도서의 개수가 1개라도 존재한다면 해당 정보를 반환
    if len(related_books_isbn) > 0:
        print('상위 5개 키워드에 기반한 공통도서 추출 완료 :', related_books_isbn)
        print('뽑힌 키워드:', word_list[0], word_list[1], word_list[2], word_list[3], word_list[4])
        return related_books_isbn, 1  # 공통 도서의 isbn 리스트와 공통 도서임을 알리는 1 반환


    # 삭제 과정이 끝나고 최종으로 추출이 마친 공통 도서의 개수가 0개라면 isbn_list를 하나씩 증가하면서 공통 도서 추출
    else:
        while True:
            index += 1

            del transfer_isbn_list[0]
            transfer_isbn_list.append(allIsbn_list[index])

            # 공통도서 추출 함수 호출
            related_books_isbn = find_common_books(transfer_isbn_list)
            print('처음 추출된 공통도서:', related_books_isbn)

            # 공통되는 도서 ISBN 목록에 그 기준이 되는 사서추천도서가 있다면 삭제하기
            if len(related_books_isbn) > 0:
                for i in range(0, len(related_books_isbn)):
                    if (related_books_isbn[i] == user_isbn):
                        del related_books_isbn[i]

                print('삭제 과정을 마친 공통도서:', related_books_isbn)


            # 삭제 과정이 끝나고 최종으로 추출이 마친 공통 도서의 개수가 1개라도 존재한다면 해당 정보를 출력하고 종료
            if len(related_books_isbn) > 0:
                print('키워드 리스트에서 하나씩 내려가면서 공통도서 추출 완료 :', related_books_isbn)
                print('<뽑힌 키워드>')
                for i in range((index-4), (index+1)):
                    print(i, '번', word_list[i])

                return related_books_isbn, 1  # 공통 도서의 isbn 리스트와 공통 도서임을 알리는 1 반환

            # 마지막 키워드까지 다다랐다면
            if index >= (len(allIsbn_list) - 1):
                cnt = 0  # 랜덤을 도는 횟수

                # 키워드를 랜덤으로 선정하여 공통 도서 추출
                while True:
                    cnt += 1
                    transfer_isbn_list = []  # 리스트 초기화
                    index_list = random.sample(range(0, len(allIsbn_list)), 5)

                    # 랜덤으로 선정된 5개로 리스트 채우기
                    for i in index_list:
                        transfer_isbn_list.append(allIsbn_list[i])

                    # 공통도서 추출 함수 호출
                    related_books_isbn = find_common_books(transfer_isbn_list)
                    print('처음 추출된 공통도서:', related_books_isbn)

                    # 공통되는 도서 ISBN 목록에 그 기준이 되는 사서추천도서가 있다면 삭제하기
                    if len(related_books_isbn) > 0:
                        for i in range(0, len(related_books_isbn)):
                            if (related_books_isbn[i] == user_isbn):
                                del related_books_isbn[i]

                        print('삭제 과정을 마친 공통도서:', related_books_isbn)


                    # 삭제 과정이 끝나고 최종으로 추출이 마친 공통 도서의 개수가 1개라도 존재한다면 해당 정보를 출력하고 종료
                    if len(related_books_isbn) > 0:
                        print('키워드를 랜덤으로 선정하여 공통도서 추출 완료 :', related_books_isbn)
                        print('<뽑힌 키워드>')
                        for i in index_list:
                            print(i, '번', word_list[i])

                        return related_books_isbn, 1  # 공통 도서의 isbn 리스트와 공통 도서임을 알리는 1 반환

                    if cnt > 5000:
                        print('해당 사서추천도서와 관련된 추천도서가 없습니다')
                        ISBN = [randomBooks[1], randomBooks[2], randomBooks[3], randomBooks[4], randomBooks[5]]
                        print('해당 사서추천도서와 동일한 카테고리의 다른 도서를 추천합니다:', ISBN)
                        return ISBN, 0  # 동일 카테고리의 또 다른 도서의 isbn과 다른 도서임을 의미하는 0 반환


def crawl_book_contents_with_selenium(isbn):
    options = Options()
    options.add_experimental_option("detach", True)  # 브라우저 꺼짐 방지 옵션, 브라우저 종료하지않고 유지
    service = Service(ChromeDriverManager().install())  # 크롬 드라이버 최신 버전 설정
    driver = webdriver.Chrome(service=service, options=options)  # 사용할 웹 브라우저 선택 및 설정

    url = f'https://www.nl.go.kr/NL/contents/search.do?pageNum=1&pageSize=30&srchTarget=total&kwd={isbn}#'
    driver.maximize_window()

    driver.get(url)   # 해당 URL을 선택된 브라우저로 오픈...

    time.sleep(1)
    try:
        driver.find_element(By.CSS_SELECTOR, '#sub_content > div.content_wrap > div > div.integSearch_wrap > div.search_cont_wrap > div > div > div.search_right_section > div.section_cont_wrap > div.section_cont.ompatibility_link_wrap > div.cont_list.list_type > div.row > div.row_info_wrap > a').click()
        time.sleep(1)
        driver.find_element(By.CSS_SELECTOR,'#sub_content > div.content_wrap > div > div.integSearch_wrap > div.search_cont_wrap > div > div > div.search_right_section > div.section_cont_wrap > div.section_cont.ompatibility_link_wrap > div.cont_list.list_type > div.row > div.row_info_wrap > div > a').click()

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        target_element = soup.select_one("#sub_content > div.content_wrap > div > div.integSearch_wrap > div.search_cont_wrap > div > div > div.search_right_section > div.section_cont_wrap > div.section_cont.ompatibility_link_wrap > div.cont_list.list_type > div.row > div.row_info_wrap > div > div")

        text_content = target_element.text
        print("추출된 텍스트:", text_content)
        return text_content

    except Exception as e:
            print("요소를 찾을 수 없습니다.")
            print(e)
            return None


def get_cosine_similarity(book1, book2):
    vectorizer = CountVectorizer().fit_transform([book1, book2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def main():
    # 사서추천도서를 카테고리별로 분류 및 데이터프레임 생성
    literature_df, HumanitiesSciences_df, socialScience_df, naturalScience_df, etc_df = Recommended_books()

    # 사용자가 읽은 사서추천도서와 동일 카테고리의 또 다른 도서 선정
    randomBooks = pick_randomBook(literature_df, HumanitiesSciences_df, socialScience_df, naturalScience_df, etc_df)
    print(randomBooks)

    # 사용자가 읽은 사서추천도서의 모든 키워드와 가중치, 해당 키워드에 해당하는 도서 검색 결과인 isbn 리스트를 데이터 프레임으로 생성
    keywordWeight_df = keywordWeightDataRrame(randomBooks[0])

    # 사용자가 읽은 사서추천도서에 대한 5가지 키워드에 대한 공통 도서 추출 (공통 도서가 없다면 동일 카테고리의 또 다른 도서 추출)
    isbn_list = []
    isbn_list, opt = continuedKword_books(keywordWeight_df, randomBooks)

    # 기준이 되는 사서추천도서와 관련된 책이 추출된 경우
    if opt == 1:
        isbn_list.insert(0, randomBooks[0])
        contents_list = []

        for i in range(len(isbn_list)):
            isbn = isbn_list[i]
            text_content = crawl_book_contents_with_selenium(isbn)

            if text_content is not None:
                contents_list.append(text_content)
            else:
                contents_list.append("목차를 찾을 수 없습니다.")
            print('----------')

        for i in range(len(contents_list)):
            print(contents_list[i])

        # 책과 유사도를 저장할 딕셔너리 생성
        similarity_dict = {}

        # 기준이 되는 책과의 목차 유사도 계산 및 저장
        base_index = 0
        base_contents = contents_list[base_index]
        for i in range(len(contents_list)):
            if i != base_index:
                similarity = get_cosine_similarity(base_contents, contents_list[i])
                print(f"책 {isbn_list[base_index]}과 책 {isbn_list[i]}의 목차 유사도: {similarity}")

                # 결과를 딕셔너리에 저장
                similarity_dict[isbn_list[i]] = similarity

        print("-----정렬한 결과----")
        # 딕셔너리를 유사도를 기준으로 내림차순 정렬
        sorted_similarity = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

        # 정렬된 결과 출력
        for isbn, similarity in sorted_similarity:
            print(f"책 {isbn_list[base_index]}과 책 {isbn}의 목차 유사도: {similarity}")

    # 기준이 되는 사서추천도서와 동일 카테고리의 다른 책들이 추출된 경우
    elif opt == 0:
        for isbn in isbn_list:
            print("동일 카테고리 내 다른 도서:", isbn)





if __name__ == '__main__':
    main()