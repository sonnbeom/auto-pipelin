# 데이터셋 불러오는 함수
from sklearn.datasets import load_breast_cancer
# 데이터를 학습용과 테스트 용으로 나눕니다.
from sklearn.model_selection import train_test_split
# 로지스틱회귀 모델 import
from sklearn.linear_model import LogisticRegression
# 결정 트리 import 
from sklearn.tree import DecisionTreeClassifier
# 랜덤포레스트 결정 트리 여러개를 만들어 다수결로 결정. 트리 하나보다 안정적
from sklearn.ensemble import RandomForestClassifier
# KNN 새 데이터와 가장 가까운 k개 이웃을 보고 다수결 (여기서 말하는 거리는 벡터를 의미함)
from sklearn.neighbors import KNeighborsClassifier

# 데이터셋 불러오기
X, Y = load_breast_cancer(return_X_y=True)

## test_size=0.2 : 20%를 테스트용으로, 80%를 학습용으로 나눔 random_state=42 : 랜덤하게 나누되, 같은 값으로 고정하여 재현성을 확보
# 학습용 테이터와 테스트용 데이터를 나누기
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 모델 생성
models ={
    "로지스틱 회귀": LogisticRegression(max_iter=10000),
    "결정 트리": DecisionTreeClassifier(random_state=42),
    "랜덤포레스트": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# 학습, 평가하기
for name, model in models.items():
    # 학습
    model.fit(X_train, Y_train)
    # 평가
    train_score = model.score(X_train, Y_train)  # 학습 데이터로 평가
    test_score = model.score(X_test, Y_test)     # 테스트 데이터로 평가
    print(f"{name} | 학습: {train_score:.4f} | 테스트: {test_score:.4f} | 차이: {train_score - test_score:.4f}")










































    