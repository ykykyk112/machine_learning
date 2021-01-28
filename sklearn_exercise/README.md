# SVM을 이용한 과일 이미지 분류기
### 'Avocado', 'Banana', 'Blueberry', 'Chestnut', 'Corn', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Peach', 'Pear', 'Strawberry' 등 12종류의 과일 이미지를 분류하는 분류기 입니다.

## 개발환경
- Python
- Scikit-learn (ML Library)

## 데이터셋
- 총 12개 Class의 컬러 이미지 데이터셋 이용
- 학습용 데이터셋 1904장, 테스트용 데이터셋 5682장
- 이미지 출처 : https://www.kaggle.com/moltean/fruits

## 성능
- 2021.01.28 기준 약 94%
- ![모델 성능](https://user-images.githubusercontent.com/59644868/106130635-efc30b80-61a4-11eb-8c06-2814c03b8297.JPG)

## 개선사항
- 외부 이미지 전처리 과정 추가
- kernel SVC에 대한 Gridsearch 수행으로 최적의 C와 gamma값 찾기

### Contributors
- Seokjun Lee (ykykyk112)<br/>
<br/>
Soongil Univ.
