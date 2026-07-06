
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

states = ["AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"]

st.title("📊 Customer Churn Prediction")

with st.sidebar:
    st.header("Model")
    st.write("Decision Tree Pipeline")
    st.write("Enter customer details and click Predict.")

c1,c2=st.columns(2)

with c1:
    state=st.selectbox("State",states)
    account=st.number_input("Account length",1,500,100)
    intl_plan=st.selectbox("International plan",["No","Yes"])
    vm_plan=st.selectbox("Voice mail plan",["No","Yes"])
    day_min=st.number_input("Total day minutes",0.0,400.0,180.0)
    day_calls=st.number_input("Total day calls",0,300,100)
    eve_min=st.number_input("Total eve minutes",0.0,400.0,200.0)
    eve_calls=st.number_input("Total eve calls",0,300,100)

with c2:
    night_min=st.number_input("Total night minutes",0.0,400.0,200.0)
    night_calls=st.number_input("Total night calls",0,300,100)
    intl_min=st.number_input("Total intl minutes",0.0,30.0,10.0)
    intl_calls=st.number_input("Total intl calls",0,30,4)
    cs_calls=st.number_input("Customer service calls",0,20,1)

if st.button("Predict Churn", use_container_width=True):
    total_charges = day_min*0.17 + eve_min*0.085 + night_min*0.045 + intl_min*0.27
    total_usage = day_min + eve_min + night_min + intl_min
    service_stress = cs_calls/max(account,1)

    row = {
        "Account length": account,
        "International plan": 1 if intl_plan=="Yes" else 0,
        "Voice mail plan": 1 if vm_plan=="Yes" else 0,
        "Total day minutes": day_min,
        "Total day calls": day_calls,
        "Total eve minutes": eve_min,
        "Total eve calls": eve_calls,
        "Total night minutes": night_min,
        "Total night calls": night_calls,
        "Total intl minutes": intl_min,
        "Total intl calls": intl_calls,
        "Customer service calls": cs_calls,
        "Total Charges": total_charges,
        "Total_Usage": total_usage,
        "Service_Stress": service_stress,
    }

    for s in states:
        row[f"State_{s}"]=1 if s==state else 0

    row["Revenue_Segment_Medium"]=0
    row["Revenue_Segment_High"]=0
    if total_charges>=100:
        row["Revenue_Segment_High"]=1
    elif total_charges>=50:
        row["Revenue_Segment_Medium"]=1

    df=pd.DataFrame([row])

    # ensure exact order
    if hasattr(model,"feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col]=0
        df=df[list(model.feature_names_in_)]

    pred=model.predict(df)[0]
    prob=model.predict_proba(df)[0][1] if hasattr(model,"predict_proba") else None

    if pred==1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    if prob is not None:
        st.metric("Churn Probability", f"{prob*100:.2f}%")

    st.subheader("Engineered Features")
    st.write({
        "Total Charges": round(total_charges,2),
        "Total Usage": round(total_usage,2),
        "Service Stress": round(service_stress,3)
    })
