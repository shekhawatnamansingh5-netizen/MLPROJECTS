import streamlit as st
import pandas as pd
import joblib
model=joblib.load("LG_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")
st.title("Heart Stroke Prediction from NAMAN")
st.markdown("provide the following detail")
age=st.slider("Age", 18, 100, 40)
sex=st.selectbox("SEX",["Male","Female"])
chest_pain=st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp=st.number_input("Restung Blood Pressure(mm Hg)",80,200,120)
cholestrol=st.number_input("Cholestrol(mg/dl)",100,600,200)
fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ecg=st.selectbox("Resting ECG Result",["Normal","ST","LVH"])
max_hr=st.slider("Maximum Heart Rate Achieved",60,220,150)
exercise_angina=st.selectbox("Exercise Induced Angina",['Y','N'])
oldpeak=st.number_input("Oldpeak(ST Depression)",0.0,6.0,0.5)
st_slope=st.selectbox("ST slope",["Up","Flat","Down"])
if st.button("Predict"):
    raw_input={
        'Age':age,
        'resting_bp':resting_bp,
        'Cholestrol':cholestrol,
        'FastingBS':fasting_bs,
        'MaxHR':max_hr,
        'Oldpeak':oldpeak,
        'Sex_'+ sex : 1,
        'ChestPainType_'+chest_pain: 1,
        'RestingECG_'+resting_ecg: 1,
        'ExerciseAngina_'+exercise_angina: 1,
        'ST_Slope_'+st_slope: 1


    }
    input_df=pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0
    input_df=input_df[expected_columns]
    scaler_input=scaler.transform(input_df)
    prediction=model.predict(scaler_input)[0]
    if prediction==1:
        st.error("⚠️⚠️⚠️The person is likely to have a heart stroke☠️☠️☠️")
    else:
        st.success("The person is unlikely to have a heart stroke😮‍💨")





