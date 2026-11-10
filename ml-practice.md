# 머신러닝 기초 실습

> 목표: 여러 모델을 직접 돌려보고, **과적합 / 과소적합**을 눈으로 확인한다.

---

## 0. 준비

```bash
uv add scikit-learn matplotlib pandas
```

실행은 이렇게 한다.

```bash
uv run python 파일명.py
```

### 데이터 가져오는 방식 3가지

| 방식 | 예시 | 특징 |
|---|---|---|
| **내장 실제 데이터** | `load_iris()`, `load_breast_cancer()`, `load_diabetes()` | scikit-learn 안에 들어 있어서 바로 쓸 수 있음 |
| **인공 생성 데이터** | `make_moons()`, `make_classification()` | 노이즈와 샘플 수를 조절할 수 있어서 실험하기 좋음 |
| **외부 데이터** | Kaggle CSV → `pd.read_csv()` | 실전 연습용 (예: 타이타닉) |

---

## 실습 0. 데이터 먼저 들여다보기

`load_breast_cancer()`는 **실제 데이터**다.
- 출처: 미국 위스콘신 대학교 병원 유방암 검사 데이터
- 환자 569명, 특성 30개 (세포 크기, 둘레, 매끄러운 정도 등)
- 정답: `0` = 악성, `1` = 양성

```python
# ex0_look_data.py
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(df.shape)                     # (569, 31)
print(df.head())                    # 앞 5줄
print(df["target"].value_counts())  # 양성/악성 각각 몇 명인지
print(df.describe())                # 특성별 평균, 최소, 최대 등
```

**확인할 것**
- [ ] 행과 열은 각각 몇 개인가?
- [ ] 양성과 악성 비율은 어떻게 되는가? 한쪽으로 쏠려 있는가?
- [ ] 특성마다 값의 범위(스케일)가 얼마나 다른가?

---

## 실습 1. 여러 모델 비교하기

기본 흐름: **데이터 → 학습/테스트 분리 → 학습(`fit`) → 평가(`score`)**

```python
# ex1_compare_models.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1) 데이터
X, y = load_breast_cancer(return_X_y=True)

# 2) 학습 80% / 테스트 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) 모델 준비
models = {
    "로지스틱 회귀": LogisticRegression(max_iter=5000),
    "결정 트리": DecisionTreeClassifier(random_state=42),
    "랜덤 포레스트": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
}

# 4) 학습 + 평가
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"{name:10s} | 학습: {train_acc:.3f} | 테스트: {test_acc:.3f}")
```

**확인할 것**
- [ ] 학습 정확도가 1.000인 모델은 무엇인가? 테스트 정확도와 차이는 얼마인가? → **과적합 신호**
- [ ] 테스트 정확도가 가장 높은 모델은 무엇인가?

**바꿔보기**
- [ ] `random_state`를 0, 1, 7로 바꾸면 결과가 얼마나 달라지는가?
- [ ] `test_size`를 0.5로 바꾸면 어떻게 되는가?
- [ ] KNN의 `n_neighbors`를 1, 5, 50으로 바꿔보기 (1이면 어떻게 될까?)

---

## 실습 2. 결정 트리로 과적합 / 과소적합 보기

트리 깊이(`max_depth`) = 모델 복잡도. 깊이를 늘려가며 점수 변화를 그래프로 그린다.

```python
# ex2_overfitting_tree.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

depths = range(1, 16)
train_scores, test_scores = [], []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.plot(depths, train_scores, "o-", label="train")
plt.plot(depths, test_scores, "o-", label="test")
plt.xlabel("max_depth (model complexity)")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

**그래프 읽는 법**

| 구간 | 상태 | 특징 |
|---|---|---|
| 깊이가 얕을 때 | **과소적합** | train과 test 점수가 둘 다 낮음 |
| 중간 | **적당함** | test 점수가 가장 높음 |
| 깊이가 깊을 때 | **과적합** | train은 1.0인데 test는 떨어지거나 제자리 |

**확인할 것**
- [ ] test 점수가 가장 높은 `max_depth`는 몇인가?
- [ ] train 점수가 1.0이 되는 깊이는 몇부터인가?

**바꿔보기: 노이즈가 많은 인공 데이터**

데이터 부분을 아래처럼 바꾸면 과적합이 더 뚜렷하게 보인다.

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
```

- [ ] `noise`를 0.1과 0.5로 바꾸면 그래프가 어떻게 달라지는가?
- [ ] `n_samples`를 3000으로 늘리면 과적합이 줄어드는가?

---

## 실습 3. 회귀로 과적합 눈으로 보기

다항식 차수 = 모델 복잡도. 차수 1, 4, 15로 곡선을 그려 비교한다.

```python
# ex3_overfitting_poly.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 진짜 정답 함수: 코사인 곡선. 여기에 노이즈를 섞어서 데이터를 만든다
rng = np.random.RandomState(0)
X = np.sort(rng.rand(30))
y = np.cos(1.5 * np.pi * X) + rng.randn(30) * 0.1
X = X.reshape(-1, 1)

# 학습에 쓰지 않은 테스트 데이터
X_test = np.sort(rng.rand(100)).reshape(-1, 1)
y_test = np.cos(1.5 * np.pi * X_test).ravel() + rng.randn(100) * 0.1

X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
degrees = [1, 4, 15]

plt.figure(figsize=(15, 4))
for i, deg in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X, y)

    train_mse = mean_squared_error(y, model.predict(X))
    test_mse = mean_squared_error(y_test, model.predict(X_test))

    plt.subplot(1, 3, i + 1)
    plt.scatter(X, y, s=20, label="train data")
    plt.plot(X_plot, model.predict(X_plot), "r-", label="model")
    plt.plot(X_plot, np.cos(1.5 * np.pi * X_plot), "g--", label="true")
    plt.ylim(-2, 2)
    plt.title(f"degree={deg}\ntrain MSE={train_mse:.3f} / test MSE={test_mse:.3f}")
    plt.legend()

plt.tight_layout()
plt.show()
```

> MSE(평균제곱오차)는 **작을수록 좋다**. 정확도와 반대 방향이다.

**확인할 것**
- [ ] 차수 1: 곡선을 제대로 따라가는가? → 과소적합
- [ ] 차수 15: train MSE와 test MSE를 비교하면? → 과적합
- [ ] 세 개 중에서 test MSE가 가장 작은 차수는 무엇인가?

**바꿔보기**
- [ ] 데이터 개수를 30에서 300으로 늘리면 차수 15도 괜찮아지는가?
- [ ] `degrees = [1, 2, 3, 4, 5, 8, 15]`로 바꿔서 test MSE가 가장 작은 차수를 찾아보기

---

## 정리: 과적합 / 과소적합 판단법

| | train 점수 | test 점수 | 해결 방향 |
|---|---|---|---|
| **과소적합** | 낮음 | 낮음 | 모델을 더 복잡하게, 특성 추가 |
| **적당함** | 높음 | 높음 (train과 비슷) | 👍 |
| **과적합** | 매우 높음 | 낮음 (차이가 큼) | 모델 단순화, 데이터 늘리기, 규제 |

---

## 다음 단계

1. **교차 검증** (`cross_val_score`): 데이터를 어떻게 나누느냐에 따라 점수가 흔들리는 문제 해결
2. **하이퍼파라미터 튜닝** (`GridSearchCV`): 최적의 `max_depth`를 자동으로 찾기
3. **전처리** (`StandardScaler`, `Pipeline`): KNN, 로지스틱 회귀는 스케일링하면 성능이 달라짐
4. **평가 지표**: 정밀도, 재현율, F1, 혼동행렬
5. **실전 데이터**: Kaggle 타이타닉으로 처음부터 끝까지 해보기

---

## 실습 메모

| 실습 | 결과 / 느낀 점 |
|---|---|
| 0 | |
| 1 | |
| 2 | |
| 3 | |
