import pandas as pd
import numpy as np
import streamlit as st 
from skimage import io
import matplotlib.pyplot as plt

st.title(':orange[Построение логистической регрессии]  :chart:')
st.subheader('***:violet[Допустим датасет с двумя фичами и одним таргетом — в последнем столбце]***')
file = st.file_uploader(':orange[Загрузите ваш файл csv]', type=['csv'])

if file is not None:
    file = pd.read_csv(file)
else:
    st.stop()
    
from sklearn.preprocessing import StandardScaler 
s_scaler = StandardScaler()

file.iloc[:,[0]] = s_scaler.fit_transform(file.iloc[:,[0]])
file.iloc[:,[1]] = s_scaler.fit_transform(file.iloc[:,[1]])
   
def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid

class LogReg:
    def __init__(self, learning_rate, n_iters,n_inputs = 2):
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.n_iters = n_iters
        self.coef_ = np.random.uniform(-7, 7, n_inputs)
        self.intercept_ = np.random.uniform(-7, 7, 1)
        
    def fit(self, X, y):
        n_inputs, n_features = X.shape

        for _ in range(self.n_iters):
            init_pred = np.dot(X, self.coef_) + self.intercept_
            sigm_pred = sigmoid(init_pred)
            grad_w = (-1/n_inputs) * np.dot(X.T, (y - sigm_pred))
            grad_w0 = (-1/n_inputs) * np.sum(y - sigm_pred)
            self.coef_  = self.coef_  - self.learning_rate * grad_w
            self.intercept_ = self.intercept_ - self.learning_rate * grad_w0
    
    def predict(self, X):
        init_pred = np.dot(X, self.coef_) + self.intercept_
        sigm_pred = sigmoid(init_pred)
        class_pred = [0 if g<=0.5 else 1 for g in sigm_pred]
        return class_pred
try:
    with st.form(key='my_form'):
        submit_button = st.form_submit_button(label='Отправить')
        learning_rate=float(st.text_input(label='Напишите желаемый learning rate)'))    
    with st.form(key='my_form2'):
        submit_button = st.form_submit_button(label='Отправить')
        n_iters=int(st.text_input(label='Напишите число желаемых операций)'))
except:
    st.stop()
   
log_r = LogReg(learning_rate, n_iters)
log_r.fit(file.iloc[:, [0, 1]], file.iloc[:,2])
#class_prediction = log_r.predict(file[['Income','CCAvg']])


st.subheader('ω0 (свободный коэффициент) =') 
st.subheader(log_r.intercept_)  
st.subheader('ω1,ω2 (коэффициенты первой и второй фичей) =')
st.subheader(log_r.coef_)   

x = file.iloc[:, 0]
y = file.iloc[:, 1]
z = file.iloc[:, 2]
colors = np.where(z == 0, 'blue', 'red')
fig, ax = plt.subplots()

ax.scatter(x, y, c=colors)
ax.set_xlabel('feat1')
ax.set_ylabel('feat2')
ax.set_title('Логистическая регрессия')

ω1 = log_r.coef_[1]
ω2 = log_r.coef_[0]
ω0 = log_r.intercept_

line_x = np.linspace(-2, 3, 1000)
line_y = (-ω1 * line_x - ω0) / ω2
ax.plot(line_x, line_y, color='green')
st.pyplot(fig)