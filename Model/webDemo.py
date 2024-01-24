import streamlit as st
from pyvi import ViTokenizer
import pandas as pd
import re
import joblib
from tensorflow.keras.models import load_model

# Load các model và bộ tiền xử lý
count_vectorizer = joblib.load('count_vectorizer.joblib')
onehot = joblib.load('one_hot_encoder.joblib')
model = load_model('model.h5')
st.title("Phân Loại Câu Hỏi Tuyển Sinh")

user_input = st.text_input("Câu hỏi là gì?....:")

def preprocessing(text):
    with open('stopword.txt','r', encoding='utf-8') as f:
        sw = f.readlines()
    sw = [i[:-1] for i in sw]
    text = ' '.join([word.lower() for word in text.split() if word.lower() not in sw]).strip()
    text = re.sub(r'[^a-zA-Z0123456789àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\s]', '', text.lower())
    return text
if user_input:
    # Tiền xử lý văn bản
    preprocessed_input = preprocessing(user_input)

    # Mã hóa văn bản
    input_encoded = count_vectorizer.transform([preprocessed_input]).toarray()

    # Thực hiện dự đoán
    prediction = model.predict([input_encoded])
    predicted_label = onehot.inverse_transform(prediction)

    st.write("Thể loại của câu hỏi:", predicted_label[0])
