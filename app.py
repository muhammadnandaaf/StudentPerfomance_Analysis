import streamlit as st
import pandas as pd
import joblib
from preprocessing_func import *
from prediction_funct import prediction

col1, col2 = st.columns([1, 5])
with col1:
    st.image("images/student_logo.jpg", width=130)
with col2:
    st.header('Education Prediction App (Prototype)')

data = pd.DataFrame()

#============================================================================================================================
# Categorical Column
#============================================================================================================================
col1, col2, col3 = st.columns(3)
with col1:
    Marital_status = st.selectbox(label='Marital_status', options=encoder_Marital_status.classes_, index=1)
    data["Marital_status"] = [Marital_status]
 
with col2:
    Application_mode = st.selectbox(label='Application_mode', options=encoder_Application_mode.classes_, index=1)
    data["Application_mode"] = [Application_mode]
 
with col3:
    Course = st.selectbox(label='Course', options=encoder_Course.classes_, index=5)
    data["Course"] = Course

#============================================================================================================================
col1, col2, col3 = st.columns(3)
with col1:
    Daytime_evening_attendance = st.selectbox(label='Daytime_evening_attendance', options=encoder_Daytime_evening_attendance.classes_, index=1)
    data["Daytime_evening_attendance"] = [Daytime_evening_attendance]
 
with col2:
    Previous_qualification = st.selectbox(label='Previous_qualification', options=encoder_Previous_qualification.classes_, index=1)
    data["Previous_qualification"] = [Previous_qualification]
 
with col3:
    Nacionality = st.selectbox(label='Nacionality', options=encoder_Nacionality.classes_, index=5)
    data["Nacionality"] = Nacionality

#============================================================================================================================
col1, col2, col3 = st.columns(3)
with col1:
    Mothers_qualification = st.selectbox(label='Mothers_qualification', options=encoder_Mothers_qualification.classes_, index=1)
    data["Mothers_qualification"] = [Mothers_qualification]
 
with col2:
    Fathers_qualification = st.selectbox(label='Fathers_qualification', options=encoder_Fathers_qualification.classes_, index=1)
    data["Fathers_qualification"] = [Fathers_qualification]
 
with col3:
    Mothers_occupation = st.selectbox(label='Mothers_occupation', options=encoder_Mothers_occupation.classes_, index=5)
    data["Mothers_occupation"] = Mothers_occupation

#============================================================================================================================
col1, col2, col3 = st.columns(3)
with col1:
    Fathers_occupation = st.selectbox(label='Fathers_occupation', options=encoder_Fathers_occupation.classes_, index=1)
    data["Fathers_occupation"] = [Fathers_occupation]
 
with col2:
    Displaced = st.selectbox(label='Displaced', options=encoder_Displaced.classes_, index=2)
    data["Displaced"] = [Displaced]
 
with col3:
    Educational_special_needs = st.selectbox(label='Educational_special_needs', options=encoder_Educational_special_needs.classes_, index=1)
    data["Educational_special_needs"] = Educational_special_needs

#============================================================================================================================
col1, col2, col3 = st.columns(3)
with col1:
    Debtor = st.selectbox(label='Debtor', options=encoder_Debtor.classes_, index=1)
    data["Debtor"] = [Debtor]
 
with col2:
    Tuition_fees_up_to_date = st.selectbox(label='Tuition_fees_up_to_date', options=encoder_Tuition_fees_up_to_date.classes_, index=1)
    data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]
 
with col3:
    Gender = st.selectbox(label='Gender', options=encoder_Gender.classes_, index=1)
    data["Gender"] = Gender

#============================================================================================================================
col1, col2 = st.columns(2)
with col1:
    Scholarship_holder = st.selectbox(label='Scholarship_holder', options=encoder_Scholarship_holder.classes_, index=1)
    data["Scholarship_holder"] = [Scholarship_holder]
 
with col2:
    International = st.selectbox(label='International', options=encoder_International.classes_, index=1)
    data["International"] = [International]

#============================================================================================================================
# Numerical Column
#============================================================================================================================  
col1, col2, col3, col4 = st.columns(4)
with col1:
    Application_order = int(st.number_input(label='Application_order', value=5))
    data["Application_order"] = Application_order
 
with col2:
    Previous_qualification_grade = int(st.number_input(label='Previous_qualification_grade', value=122.0))
    data["Previous_qualification_grade"] = Previous_qualification_grade
 
with col3:
    Admission_grade = int(st.number_input(label='Admission_grade', value=124.8))
    data["Admission_grade"] = Admission_grade
 
with col4:
    Age_at_enrollment = float(st.number_input(label='Age_at_enrollment', value=19))
    data["Age_at_enrollment"] = Age_at_enrollment
 
#============================================================================================================================
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    Curricular_units_1st_sem_credited = int(st.number_input(label='Curricular_units_1st_sem_credited', value=0))
    data["Curricular_units_1st_sem_credited"] = Curricular_units_1st_sem_credited
 
with col2:
    Curricular_units_1st_sem_enrolled = int(st.number_input(label='Curricular_units_1st_sem_enrolled', value=6))
    data["Curricular_units_1st_sem_enrolled"] = Curricular_units_1st_sem_enrolled
 
with col3:
    Curricular_units_1st_sem_evaluations = int(st.number_input(label='Curricular_units_1st_sem_evaluations', value=6))
    data["Curricular_units_1st_sem_evaluations"] = Curricular_units_1st_sem_evaluations
 
with col4:
    Curricular_units_1st_sem_approved = float(st.number_input(label='Curricular_units_1st_sem_approved', value=6))
    data["Curricular_units_1st_sem_approved"] = Curricular_units_1st_sem_approved

with col5:
    Curricular_units_1st_sem_grade	 = float(st.number_input(label='Curricular_units_1st_sem_grade	', value=14))
    data["Curricular_units_1st_sem_grade	"] = Curricular_units_1st_sem_grade	

with col6:
    Curricular_units_1st_sem_without_evaluations = float(st.number_input(label='Curricular_units_1st_sem_without_evaluations', value=0))
    data["Curricular_units_1st_sem_without_evaluations"] = Curricular_units_1st_sem_without_evaluations

#============================================================================================================================
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    Curricular_units_2nd_sem_credited = int(st.number_input(label='Curricular_units_2nd_sem_credited', value=0))
    data["Curricular_units_2nd_sem_credited"] = Curricular_units_2nd_sem_credited
 
with col2:
    Curricular_units_2nd_sem_enrolled = int(st.number_input(label='Curricular_units_2nd_sem_enrolled', value=6))
    data["Curricular_units_2nd_sem_enrolled"] = Curricular_units_2nd_sem_enrolled
 
with col3:
    Curricular_units_2nd_sem_evaluations = int(st.number_input(label='Curricular_units_2nd_sem_evaluations', value=6))
    data["Curricular_units_2nd_sem_evaluations"] = Curricular_units_2nd_sem_evaluations
 
with col4:
    Curricular_units_2nd_sem_approved = float(st.number_input(label='Curricular_units_2nd_sem_approved', value=6))
    data["Curricular_units_2nd_sem_approved"] = Curricular_units_2nd_sem_approved

with col5:
    Curricular_units_2nd_sem_grade	 = float(st.number_input(label='Curricular_units_2nd_sem_grade	', value=13.66))
    data["Curricular_units_2nd_sem_grade	"] = Curricular_units_2nd_sem_grade	

with col6:
    Curricular_units_2nd_sem_without_evaluations = float(st.number_input(label='Curricular_units_2nd_sem_without_evaluations', value=0))
    data["Curricular_units_2nd_sem_without_evaluations"] = Curricular_units_2nd_sem_without_evaluations
 
#============================================================================================================================
col1, col2, col3 = st.columns(3)
with col1:
    Unemployment_rate = float(st.number_input(label='Unemployment_rate', value=13.9))
    data["Unemployment_rate"] = Unemployment_rate
 
with col2:
    Inflation_rate = float(st.number_input(label='Inflation_rate', value=-0.3))
    data["Inflation_rate"] = Inflation_rate
 
with col3:
    GDP = float(st.number_input(label='GDP', value=0.79))
    data["GDP"] = GDP

#============================================================================================================================
if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Education Status: {}".format(prediction(new_data)))