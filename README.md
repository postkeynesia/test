---
title: '미국 보스턴 집값 예측'
author: "문영훈"
date: '2021 12 20 '
output: html_document
editor_options: 
  markdown: 
    wrap: 72
---

```{r, echo=FALSE}
knitr::opts_chunk$set(error = TRUE)
```

# **연구요약**

-   미국 보스턴 지역의 주택가격에 관한 집값 예측을 위하여,선형회귀모형,
    회귀나무모형, 인공신경망 모형, 랜덤포레스트 모형을 사용함

-   따라서, 평균오차값(MSE) 비교를 통하여 모형성능의 우수성을 평가한
    결과, 1) 랜덤포레스트 모형, 2) 신경망 모형, 3) 회귀나무모형, 4)
    선형회귀모형 순으로 나타남

# **1. 연구배경**

-   지난 10여 년간 주택공급이 꾸준히 늘어나면서 양적인 부족은 전반적으로
    완화되었으나, 주택매매 가격 이 크게 상승하며 실수요자의 내집 마련이
    어려운 상황이 됐음

-   2019년 기준 전국 주택보급율은 104.8%로 지난 2015년 102.3%에서 2.5%p
    상승하였음

**전국 및 지역별 주택보급률**

![전국 및 지역별
주택보급률](C:/Users/yahoo/Desktop/%EC%A0%84%EA%B5%AD%20%EB%B0%8F%20%EC%A7%80%EC%97%AD%EB%B3%84%20%EC%A3%BC%ED%83%9D%EB%B3%B4%EA%B8%89%EB%A5%A0.jpg)

주 : 주택보급률은 일반가구수에 대한 주택수의
백분율(주택수/일반가구수×100)로 정의됨

자료 : e-나라지표(<https://www.index.go.kr/potal>).

-   특히 다주택자들의 주택 구매 등과 함께 고소득층의 자가점유율은
    늘어나는 반면 저소득층의 내집 마련은 더욱 힘들어지고 있음

**소득별 자가보유율**

![소득별
자가보유율](C:/Users/yahoo/Desktop/%EC%86%8C%EB%93%9D%EB%B3%84%20%EC%9E%90%EA%B0%80%EB%B3%B4%EC%9C%A0%EC%9C%A8.jpg)

주 : 하위소득은 소득 1\~4분위, 중위소득은 5\~8순위, 상위소득은
9\~10분위를 의미함

자료 : 국토교통부(2020), 2020년도 주거실태조사

-   특히 우리나라는 자산 중 부동산 비율이 매우 높아 부동산 가격변화에
    따른 국민의 관심도가 매우 높은 상황임

-   우리나라의 경우, 자산 중 비금융자산(부동산) 비율이 64.4%이며,
    금융자산 비율은 35.6%로 구성됨

**주요국 가계자산 구성 비교(2019)**

+----+----+----+----+----+----+
| 구 | 한 | 미 | 일 | 영 | 호 |
| 분 | 국 | 국 | 본 | 국 | 주 |
+:==:+:==:+:==:+:==:+:==:+:==:+
| 비 | 6  | 2  | 3  | 4  | 5  |
| 금 | 4  | 8  | 7  | 5  | 7  |
| 융 | .4 | .1 | .9 | .2 | .0 |
| 자 |    |    |    |    |    |
| 산 |    |    |    |    |    |
| (  |    |    |    |    |    |
| 부 |    |    |    |    |    |
| 동 |    |    |    |    |    |
| 산 |    |    |    |    |    |
| )  |    |    |    |    |    |
+----+----+----+----+----+----+
| 금 | 3  | 7  | 5  | 5  | 4  |
| 융 | 5  | 1  | 2  | 4  | 3  |
| 자 | .6 | .9 | .1 | .8 | .0 |
| 산 |    |    |    |    |    |
+----+----+----+----+----+----+

자료 : 금융투자협회(2021). 『2021 주요국 가계 금융자산 비교』.

-   이로 인해 현재 부동산 가격의 측정뿐만 아니라 미래 부동산 가격의
    상승·하락 여부는 구민들의 중요한 관심사항임

-   이에 따라 본 연구는 부동산 가격에 미치는 영향요인을 머신러닝을
    통해서 분석하고자 함

# 2. 선행연구

-   배성완(2019)은 머신러닝 기법을 통하여 공동주택 및 주택가격지수의
    가격을 에측하는 연구를 수행히였으며, 서포트벡터머신(SVM),
    랜덤포레스트(RF), 딥러닝(DNN), 인공신경망(장단기메모리 모형) 등을
    이용하여 공동주택 가격 및 주택가격지수를 예측하였으며, 이중
    인공신경망(장단기메모리 모형)이 가장 예측력이 높다고 분석함

-   박성훈(2020)은 다양한 부동산 지수를 대상으로 이론적 고찰과
    머신러닝(인공신경망(LSTM), XGBoost, 랜덤포레스트)을 활용한 예측을
    진행하였으며, 머신러닝 모델의 예측 정확도를 비교분석을 통하여
    인공신경망 모델 중 장단기메모리 모형이 가장 부동산 지수를 예측한다고
    분석함

-   이창훈(2021)은 베이지안 머신러닝 기법을 이용한 아파트 실거래가를
    예측하였으며, 단기예측보다는 중기와 장기예측에서 전통적 모형보다
    베이지안 머신러닝 기법이 상당이 향상된 예측 정확도를 보인다고 분석함

-   본 연구에서는 선행연구를 바탕으로 주택가격 예측을 위하여, ⅰ)
    선형회귀분석, ⅱ) 회귀트리분석, ⅲ) 인공신경망분석, ⅳ)
    랜덤포레스트분석을 실시자고자 함

# 3. 데이터 소개

-   본 연구는 1978년 미국 보스턴 지역의 ⅰ) 주택가격변수, ⅱ) 주택가격에
    영향을 미치는 13가지 변수로 이루어진 데이터셋을 분석하고자 함

-   종속변수는 본인소유의 주택가격(MEDV)로 설정하고, 설명변수는 주택가격
    변수를 제외한 나머지 변수로 설정하고자 함

```{r 1}
Boston <- read.csv('https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv', fileEncoding =  "utf-8")
```

## 1) 주요변수 설명

+---------------+------------------------------------------+
| 변수          | 변수내용                                 |
+:=============:+:========================================:+
| CRIM          | 자치시(town) 별 1인당 범죄율             |
+---------------+------------------------------------------+
| ZN            | 25,000 평방피트를 초과하는 거주지역의    |
|               | 비율                                     |
+---------------+------------------------------------------+
| INDUS         | 비소매상업지역이 점유하고 있는 토지의    |
|               | 비율                                     |
+---------------+------------------------------------------+
| CHAS          | 찰스강에 대한 더미변수(강의 경계에       |
|               | 위치한 경우는 1, 아니면 0)               |
+---------------+------------------------------------------+
| NOX           | 10ppm 당 농축 일산화질소                 |
+---------------+------------------------------------------+
| RM            | 주택 1가구당 평균 방의 개수              |
+---------------+------------------------------------------+
| AGE           | 1940년 이전에 건축된 소유주택의 비율     |
+---------------+------------------------------------------+
| DIS           | 5개의 보스턴 직업센터까지의 접근성 지수  |
+---------------+------------------------------------------+
| RAD           | 방사형 도로까지의 접근성 지수            |
+---------------+------------------------------------------+
| TAX           | 10,000 달러 당 재산세율                  |
+---------------+------------------------------------------+
| PTRATIO       | 자치시(town)별 학생/교사 비율            |
+---------------+------------------------------------------+
| Black         | 1000(Bk-0.63)\^2, 여기서 Bk는 자치시별   |
|               | 흑인의 비율                              |
+---------------+------------------------------------------+
| LSTAT         | 모집단의 하위계층의 비율(%)              |
+---------------+------------------------------------------+
| MEDV          | 본인 소유의 주택가격(중앙값) (단위:      |
|               | \$1,000)                                 |
+---------------+------------------------------------------+

```{r 2}

Boston <- Boston[,-15]
str(Boston)
```

## 2) 변수간 상관관계 분석

-   주택 1가구당 평균 방의 개수가 증가하면 본인소유의 주택가격은
    증가하지만, 하위계층의 비율이 증가할수록 본인소유의 주택가격은
    감소함

```{r 3}
if(!require(corrplot)) install.packages('corrplot')
library(corrplot)

corrplot(cor(Boston), method = "number", type = "upper", diag = FALSE)
```

## 3) 종속변수(본인 소유의 주택가격)와 설명변수 간의 관계

-   종속변수와 양의관계인 설명변수는 black, chas, dis, rm, zn 변수임 □
    종속변수와 음의 관계인 설명변수는 age, crim, indus, lstat, nox,
    ptratio, rad, tax 변수임

```{r 4}
if(!require(ggplot2)) install.packages('ggplot2')
library(ggplot2)
if(!require(tidyr)) install.packages('tidyr')
library(tidyr)


Boston %>%
  gather(key, val, -medv) %>%
  ggplot(aes(x = val, y = medv)) +
  geom_point() +
  stat_smooth(method = "lm", se = TRUE, col = "blue") +
  facet_wrap(~key, scales = "free") +
  theme_gray() +
  ggtitle("Scatter plot of dependent variables vs Median Value (medv)") 
```

# 4.예측모형

-   본연구는 선형회귀모형, 회귀나무모형, 인공신경망 모형, 랜덤포레스트
    모형을 이용하여 분석함

## 1) 선형회귀모형

-   선형회귀모형의 경우, 원자료 중 70%는 트레이닝 데이터, 30%는 테스트
    데이터로 사용하였음

```{r 5}
i <- sample(1:nrow(Boston), round(nrow(Boston)*0.7))
Boston.train <- Boston[i,] #70% for training data
Boston.test <- Boston[-i,] #30% for test data
fit.reg <- lm(medv~., data = Boston.train) # 선형회귀모형
summary(fit.reg) # 선형회귀모형 요약
```

-   단계적 선택법을 사용하여, 종속변수(부동산 가격)에 미치는 영향이 작은
    설명변수를 제거하였음

```{r 6}

fit.step.reg <- step(fit.reg, direction = 'both', trace = FALSE) # 단계적 선택법
summary(fit.step.reg) # 조정된 모델 요약

fit.step.reg$anova # 제거된 변수 보기
```

-   따라서 예측모형의 평균제곱오차(MSE)는 다음과 같이 약 31.51임

```{r 7}

yhat.reg <- predict(fit.step.reg, newdata = Boston.test, type = 'response') # 데이터 예측, 목표값 예측시 type = 'response'
mean((Boston.test$medv-yhat.reg)^2) # 예측모형의 평균 제곱 오차 mse
```

## 2) 회귀나무모형

-   회귀나무모형의 경우, 회귀모형과 동일하게 70%는 트레이닝 데이터,
    30%는 테스트 데이터로 사용하였음

```{r 8}
if(!require(rpart)) install.packages('rpart')
library(rpart)
set.seed(1234)
i <- sample(1:nrow(Boston), round(nrow(Boston)*0.7))
Boston.train <- Boston[i,] #70% for training data
Boston.test <- Boston[-i,] #30% for test data

my.control <- rpart.control(xval = 10, cp = 0,
                            minsplit = nrow(Boston.train)*0.05)
```

-   종속변수가 연속형변수이므로, 회귀나무 모형을 통하여 분석하였음

```{r 9}
fit.tree <- rpart(medv~., data = Boston.train,
                  method = 'anova', control = my.control) 

# method = anova 선택 시 회귀나무, class 선택시 분류나무
which.min(fit.tree$cp[,4])
fit.tree$cp[17,]
fit.tree$cp[17,1]
ii <- which.min(fit.tree$cp[,4])
fit.prune.tree <- prune(fit.tree, cp = fit.tree$cp[ii,1])
```

-   회귀나무 모형 분석을 통하여, 설명변수 중 가장 영향력이 강한 것을
    나타내는 뿌리노드의 경우, 지역 내 하위계층의 비율 나타내는것을 알수
    있음
-   다음으로, 1) 주택 1가구당 평균 방의 개수, 2) 자치시(town)별
    학생/교사 비율 등 순으로 영향을 미치는 것으로 분석할 수 있음

```{r 10}
plot(fit.prune.tree, uniform = T, margin = 0.1)
text(fit.prune.tree, col = 'blue', cex = 0.7)

yhat.tree <- predict(fit.prune.tree, newdata = Boston.test, type='vector')
```

-   따라서 예측모형의 평균제곱오차(MSE)는 다음과 같이 약 10.28임

```{r 11}
mean((Boston.test$medv - yhat.tree)^2)
```

## 3) 신경망모형

-   신경망모형의 경우, 회귀모형과 동일하게 70%는 트레이닝 데이터, 30%는
    테스트 데이터로 사용하였음

```{r 12}
if(!require(neuralnet)) install.packages('neuralnet')
library(neuralnet)

Boston2 <- Boston
for(i in
    1:ncol(Boston2)) if (!is.numeric(Boston2[,i])) Boston2[,i]=as.numeric(Boston[,i])
max1 <- apply(Boston2,2,max)
min1 <- apply(Boston2,2,min)

sdat <- scale(Boston2, center = min1, scale = max1-min1)
sdat <- as.data.frame(sdat)

set.seed(1234)
i <- sample(1:nrow(Boston2), round(nrow(Boston2)*0.7))
Boston2.train <- sdat[i,]
Boston2.test<- sdat[-i,]

vname <- names(Boston2.train)
form <- as.formula(paste('medv~', paste(vname[!vname%in%
                                                "medv"], collapse = '+')))

form
fit.nn <- neuralnet(form, data = Boston2.train, hidden = c(3,2), linear.output = T)
plot(fit.nn)

str(Boston.test)
pred <- compute(fit.nn, Boston2.test[,1:13])
summary(pred)
yhat.nn <- pred$net.result*(max(Boston2$medv)-
  min(Boston2$medv))+min(Boston2$medv) # 역변환
head(cbind(Boston2[i,14], yhat.nn), 50)

Boston2.test$medv<- Boston2.test$medv*(max(Boston2$medv)-min(Boston2$medv))+min(Boston2$medv)
```

-   따라서 예측모형의 평균제곱오차(MSE)는 다음과 같이 약 10.67임

```{r 13}
mean((Boston2.test$medv-yhat.nn)^2)
```

## 4) 랜덤포레스트

-   랜덤포레스트 모형의 경우, 회귀모형과 동일하게 70%는 트레이닝 데이터,
    30%는 테스트 데이터로 사용하였음
-   또한 모형을 적합하는데 ntree = 100, 중간 노드마다 랜덤하게 선택하는
    횟수는 5로하고, 변수의 중요도도 계산하고, 결측치는 삭제가 되도록
    하여 모형을 적합하였음

```{r 14}
if(!require(randomForest)) install.packages('randomForest')
library(randomForest)

set.seed(1234)
i <- sample(1:nrow(Boston), round(nrow(Boston)*0.7))
Boston.train <- Boston[i,]
Boston.test <- Boston[-i,]
fit.rf <- randomForest(medv~., data = Boston.train, ntree = 100, mtry = 5,
                       importance = T, na.action = na.omit)
```

-   랜덤포레스트 분석결과, 회귀 잔차는 12.18로 나왔으며 설명도 또한 약
    85%로 이 모형은 성능이 우수하다고 할 수 있다.

-   importance 함수를 통해 입력변수의 중요도를 확인한 결과, rm 변수가
    가장 높게 나왔으며 그 다음으로 lstat변수가 높게 나왔음

```{r 15}
fit.rf
summary(fit.rf)
importance(fit.rf, type=1)

```

-   따라서 예측모형의 평균제곱오차(MSE)는 다음과 같이 약 9.67임

```{r 16}
yhat.tree <- predict(fit.rf, newdata = Boston.test)
mean((Boston.test$medv - yhat.tree)^2)
```

# 5.결론

-   선형회귀모형, 회귀나무모형, 인공신경망 모형, 랜덤포레스트 모형의
    평균오차값(MSE) 비교를 통하여 모형성능의 우수성을 평가한 결과, 1)
    랜덤포레스트 모형, 2) 신경망 모형, 3) 회귀나무모형, 4) 선형회귀모형
    순으로 나타남

+---------------+------------------------------------------+
| 모형          | 평균오차 값(MSE)                         |
+:=============:+:========================================:+
| 선형회귀      | 31.52                                    |
+---------------+------------------------------------------+
| 회귀나무      | 14.28                                    |
+---------------+------------------------------------------+
| 신경망        | 10.7                                     |
+---------------+------------------------------------------+
| 랜덤포레스트  | 9.68                                     |
+---------------+------------------------------------------+
