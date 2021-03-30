# Daily Logs

매일 매일의 개발 과정을 담는다. 최근 날짜를 위로, 날짜 내 사건은 가장 오래된 것부터

```
2021.03.30
Accident 1
- ...
Accident 2
- ...

2021.03.29
...
...
```



#### 2021.03.29

김상훈님의 조언에 따라 베이스라인을 먼저 구축한 뒤, 이에 뒤따라 EDA를 진행해볼 예정

##### Model

###### `VanillaResNet`

- ResNet을 활용한 기본적인 분류기

###### `AttetionNet(가칭)`

- Faster-RCNN과  GoogLeNet의 구조로부터 ideation
- 성별, 마스크 상태, 연령대를 각기 다른 task로 나누어 예측
- Pretrained ImageNet을 활용해 이미지로부터 feature를 추출한 뒤, 추출한 feature로부터 3가지의 task를 수행

