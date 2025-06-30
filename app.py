import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import io

# Load data and model only once
@st.cache_resource
def load_data_and_model():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    reduced_data = training.groupby(training['prognosis']).max()
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    clf = DecisionTreeClassifier().fit(x, y_enc)
    return clf, le, cols, reduced_data, training

clf, le, cols, reduced_data, training = load_data_and_model()

# Load consult data
@st.cache_data
def load_consult():
    df = pd.read_csv('doc_consult.csv', header=None, index_col=0)
    consult = df[1].to_dict()
    return consult
consult = load_consult()

# Greeting logic
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# Session state for conversation
if 'step' not in st.session_state:
    st.session_state.step = 'greet'
if 'symptom_idx' not in st.session_state:
    st.session_state.symptom_idx = 0
if 'symptoms_present' not in st.session_state:
    st.session_state.symptoms_present = []
if 'tree_path' not in st.session_state:
    st.session_state.tree_path = []
if 'disease' not in st.session_state:
    st.session_state.disease = None
if 'other_symptoms' not in st.session_state:
    st.session_state.other_symptoms = []

st.title("MTU Health Care AI Powered Chatbot")
st.write("I will answer your queries about your health related problem. If you want to exit, type Bye!")

# Helper to get feature names from tree
feature_names = list(cols)
tree_ = clf.tree_

def get_next_symptom():
    from sklearn.tree import _tree  # type: ignore
    node = 0
    for answer in st.session_state.tree_path:
        if answer == 1:
            node = tree_.children_right[node]
        else:
            node = tree_.children_left[node]
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        return feature_names[tree_.feature[node]], node
    else:
        return None, node

def get_disease_and_symptoms(node):
    from sklearn.tree import _tree  # type: ignore
    present_disease = le.inverse_transform(tree_.value[node][0].nonzero()[0])
    if len(present_disease) > 0:
        diss = present_disease[0]
    else:
        diss = "Unknown"
    red_cols = reduced_data.columns
    symptoms_given = red_cols[reduced_data.loc[[diss]].values[0].nonzero()]
    return diss, list(symptoms_given)

if st.session_state.step == 'greet':
    user_input = st.text_input("You:", key="greet_input")
    if user_input:
        if user_input.lower() in ["bye", "exit"]:
            st.write("Bot: Bye! take care..")
            st.session_state.step = 'done'
        elif greeting(user_input) is not None:
            st.write(f"Bot: {greeting(user_input)}")
            st.write("Bot: Please reply Yes or No for the following symptoms")
            st.session_state.step = 'symptoms'
        else:
            st.write("Bot: I am sorry! I don't understand you")

elif st.session_state.step == 'symptoms':
    symptom, node = get_next_symptom()
    if symptom:
        st.write(f"Bot: Do you have {symptom}?")
        col1, col2 = st.columns(2)
        if col1.button("Yes", key=f"yes_{node}"):
            st.session_state.tree_path.append(1)
            st.session_state.symptoms_present.append(symptom)
            st.experimental_rerun()
        if col2.button("No", key=f"no_{node}"):
            st.session_state.tree_path.append(0)
            st.experimental_rerun()
    else:
        # Leaf node reached
        diss, other_symptoms = get_disease_and_symptoms(node)
        st.session_state.disease = diss
        st.session_state.other_symptoms = other_symptoms
        st.session_state.step = 'result'
        st.experimental_rerun()

elif st.session_state.step == 'result':
    diss = st.session_state.disease
    other_symptoms = st.session_state.other_symptoms
    st.write(f"Bot: You may have **{diss}**")
    st.write("Bot: Symptoms present: " + ", ".join(st.session_state.symptoms_present))
    st.write("Bot: Other symptoms: " + ", ".join(other_symptoms))
    risk = consult.get(diss, 0)
    consult_doc = ['YES', 'NO']
    if risk > 50:
        st.write("Bot: You should consult a doctor as soon as possible")
        data1 = [risk, 0]
    else:
        st.write("Bot: You may consult a doctor")
        data1 = [0, risk]
    fig, ax = plt.subplots()
    ax.bar(consult_doc, data1, color='red', width=0.15)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Risk')
    ax.set_xlabel('Consult a doctor')
    st.pyplot(fig)
    if st.button("Restart"):
        for key in ["step", "symptom_idx", "symptoms_present", "tree_path", "disease", "other_symptoms"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

elif st.session_state.step == 'done':
    st.write("Session ended. Refresh to start again.") 