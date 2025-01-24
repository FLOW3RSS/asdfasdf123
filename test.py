import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
plt.rcParams['axes.unicode_minus'] = False    # 한글 폰트 사용 시 마이너스 기호 깨짐 방지

# 지수 함수 정의
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# 로그 함수 정의
def logarithmic_func(x, a, b):
    return a * np.log(x) + b

# 웹페이지 제목
st.title("머신러닝과 데이터 분석 웹 앱")

# 파일 업로드
uploaded_file = st.file_uploader("분석할 CSV 파일 업로드", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("데이터 정리:", data)

    # 목표 변수와 특징 변수 선택
    target_col = st.selectbox("목표 변수 선택", data.columns)
    feature_cols = st.multiselect("특징 변수 선택", [col for col in data.columns if col != target_col])

    if target_col and feature_cols:
        X = data[feature_cols]
        y = data[target_col]

        # 회귀 유형 선택
        regression_type = st.selectbox("회귀 유형 선택", ["단순 선형 회귀", "다중 선형 회귀", "다항 회귀", "로그 회귀", "지수 회귀"])

        def display_metrics(y_test, y_pred):
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.markdown(f"""
            **모델 정확도 평가**

            - $R^2$ (결정 계수):
            $$R^2 = 1 - \\frac{{\sum{{(y_i - \hat{{y}}_i)^2}}}}{{\sum{{(y_i - \\bar{{y}})^2}}}}$$  
            **결과**: {r2:.2f}

            - 평균 제곱 오차 (MSE):
            $$MSE = \\frac{{\sum{{(y_i - \hat{{y}}_i)^2}}}}{{n}}$$  
            **결과**: {mse:.2f}
            """, unsafe_allow_html=True)

            st.write("""
            여기서:
            - \(y_i\): 실제값
            - \(\hat{y}_i\): 예측값
            - \(\\bar{y}\): 실제값의 평균
            - \(n\): 데이터 개수
            """)

        if regression_type == "단순 선형 회귀":
            if len(feature_cols) != 1:
                st.warning("단순 선형 회귀는 정확히 하나의 특징 변수가 필요합니다.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 회귀식 출력
                coef = model.coef_[0]
                intercept = model.intercept_
                st.latex(f"y = {coef:.2f}x + {intercept:.2f}")

                # 모델 정확도 평가
                display_metrics(y_test, y_pred)

                # 시각화
                plt.figure(figsize=(8, 6))
                plt.scatter(X[feature_cols[0]], y, color='blue', label='실제값')
                plt.plot(X[feature_cols[0]], model.predict(X), color='red', label='회귀선')
                plt.xlabel(feature_cols[0])
                plt.ylabel(target_col)
                plt.title("단순 선형 회귀")
                plt.legend()
                st.pyplot(plt)

                # 예측 모델
                st.write("### 예측 모델:")
                input_value = st.number_input(f"{feature_cols[0]} 값을 입력하세요:", value=0.0, step=0.1, key="single_linear_input")
                if st.button("단순 선형 예측 실행", key="single_linear_predict"):
                    prediction = model.predict([[input_value]])
                    st.write(f"예측 결과: {prediction[0]:.2f}")

        elif regression_type == "다중 선형 회귀":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 회귀식 출력
            coefficients = model.coef_
            intercept = model.intercept_
            equation = f"y = {intercept:.2f} + " + " + ".join([f"({coef:.2f} * {feature})" for coef, feature in zip(coefficients, feature_cols)])
            st.latex(equation)

            # 모델 정확도 평가
            display_metrics(y_test, y_pred)

            # 시각화
            if len(feature_cols) == 2:
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[feature_cols[0]], X[feature_cols[1]], y, color='blue', label='실제값')
                ax.set_xlabel(feature_cols[0])
                ax.set_ylabel(feature_cols[1])
                ax.set_zlabel(target_col)
                ax.plot_trisurf(X[feature_cols[0]], X[feature_cols[1]], model.predict(X), color='red', alpha=0.5)
                plt.title("다중 선형 회귀 3D 시각화")
                st.pyplot(plt)
            elif len(feature_cols) > 2:
                st.warning("3개 이상의 독립 변수는 시각화가 제공되지 않습니다.")

            # 예측 모델
            st.write("### 예측 모델:")
            input_values = [st.number_input(f"{col} 값을 입력하세요:", value=0.0, step=0.1, key=f"multi_input_{col}") for col in feature_cols]
            if st.button("다중 선형 예측 실행", key="multi_linear_predict"):
                prediction = model.predict([input_values])
                st.write(f"예측 결과: {prediction[0]:.2f}")

        elif regression_type == "다항 회귀":
            degree = st.slider("다항식 차수 선택", 2, 5, 2)
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 회귀식 출력
            coefs = model.coef_  # 다항식 계수
            intercept = model.intercept_
            equation = " + ".join([f"{coefs[i]:.2f}x^{i}" for i in range(1, len(coefs))])
            st.latex(f"y = {intercept:.2f} + {equation}")

            # 모델 정확도 평가
            display_metrics(y_test, y_pred)

            # 시각화
            if len(feature_cols) == 1:
                sorted_X = np.sort(X[feature_cols[0]].values)
                spline = UnivariateSpline(sorted_X, model.predict(poly.fit_transform(sorted_X.reshape(-1, 1))), s=1)
                plt.figure(figsize=(8, 6))
                plt.scatter(X[feature_cols[0]], y, color='blue', label='실제값')
                plt.plot(sorted_X, spline(sorted_X), color='red', label='회귀선')
                plt.xlabel(feature_cols[0])
                plt.ylabel(target_col)
                plt.title(f"다항 회귀 (차수: {degree})")
                plt.legend()
                st.pyplot(plt)
            else:
                st.warning("2개 이상의 특징 변수가 있는 경우 다항 회귀 시각화는 제공되지 않습니다.")

            # 예측 모델
            st.write("### 예측 모델:")
            input_values = [st.number_input(f"{col} 값을 입력하세요:", value=0.0, step=0.1, key=f"poly_input_{col}") for col in feature_cols]
            if st.button("다항 회귀 예측 실행", key="poly_predict"):
                input_poly = poly.transform([input_values])
                prediction = model.predict(input_poly)
                st.write(f"예측 결과: {prediction[0]:.2f}")

        elif regression_type == "로그 회귀":
            if len(feature_cols) != 1:
                st.warning("로그 회귀는 정확히 하나의 특징 변수가 필요합니다.")
            else:
                X_log = X[feature_cols[0]].replace(0, np.nan).dropna()
                y_filtered = y.loc[X_log.index]
                popt, _ = curve_fit(logarithmic_func, X_log, y_filtered)
                a, b = popt
                st.latex(f"y = {a:.2f} \log(x) + {b:.2f}")

                # 모델 정확도 평가
                y_pred = logarithmic_func(X_log, *popt)
                display_metrics(y_filtered, y_pred)

                # 시각화
                plt.figure(figsize=(8, 6))
                sorted_X = np.sort(X_log)
                plt.scatter(X_log, y_filtered, color='blue', label='실제값')
                plt.plot(sorted_X, logarithmic_func(sorted_X, *popt), color='red', label='회귀선')
                plt.xlabel(feature_cols[0])
                plt.ylabel(target_col)
                plt.title("로그 회귀")
                plt.legend()
                st.pyplot(plt)

                # 예측 모델
                st.write("### 예측 모델:")
                input_value = st.number_input(f"{feature_cols[0]} 값을 입력하세요:", value=1.0, min_value=0.1, step=0.1, key="log_input")
                if st.button("로그 회귀 예측 실행", key="log_predict"):
                    prediction = logarithmic_func(input_value, *popt)
                    st.write(f"예측 결과: {prediction:.2f}")

        elif regression_type == "지수 회귀":
            if len(feature_cols) != 1:
                st.warning("지수 회귀는 정확히 하나의 특징 변수가 필요합니다.")
            else:
                popt, _ = curve_fit(exponential_func, X[feature_cols[0]], y, maxfev=10000)
                a, b, c = popt
                st.latex(f"y = {a:.2f}e^{{{b:.2f}x}} + {c:.2f}")

                # 모델 정확도 평가
                y_pred = exponential_func(X[feature_cols[0]], *popt)
                display_metrics(y, y_pred)

                # 시각화
                plt.figure(figsize=(8, 6))
                sorted_X = np.sort(X[feature_cols[0]])
                plt.scatter(X[feature_cols[0]], y, color='blue', label='실제값')
                plt.plot(sorted_X, exponential_func(sorted_X, *popt), color='red', label='회귀선')
                plt.xlabel(feature_cols[0])
                plt.ylabel(target_col)
                plt.title("지수 회귀")
                plt.legend()
                st.pyplot(plt)

                # 예측 모델
                st.write("### 예측 모델:")
                input_value = st.number_input(f"{feature_cols[0]} 값을 입력하세요:", value=0.0, step=0.1, key="exp_input")
                if st.button("지수 회귀 예측 실행", key="exp_predict"):
                    prediction = exponential_func(input_value, *popt)
                    st.write(f"예측 결과: {prediction:.2f}")

        # 혼동 행렬 체크박스 추가
        if st.checkbox("혼동 행렬 보기"):
            y_class = (y > y.mean()).astype(int)
            y_pred_class = (y_pred > y.mean()).astype(int)
            cm = confusion_matrix(y_class, y_pred_class)
            st.write("### 혼동 행렬")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])            
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt)