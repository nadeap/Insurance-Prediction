import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('model.pkl', 'rb'))

def insureance_predict(input_data):
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = model.predict(id_reshaped)
    print(prediction)

def main():
    
    st.title('Percobaan Insurance Prediction')
    
    age = st.text_input('Age')
    sex = st.text_input('Sex | 0:Female & 1:Male |')
    bmi = st.text_input('BMI')
    children = st.text_input('Children')
    smoker = st.text_input('Smoker | 1:Yes & 0:No |')
    region = st.text_input('Region | 0:Northeast | 1:Northwest | 2:Southeast | 3:Southwest|')
    
    charger = ''
    
    if st.button('PREDICT'):
        charger = insureance_predict([age, sex, bmi, children, smoker, region])
        
    st.success(charger)
    
if __name__=='__main__':
    main()