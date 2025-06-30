#using the concept of decision-tree
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import streamlit as st
import matplotlib.pyplot as plt

#for ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importing the dataset
training = pd.read_csv('Training.csv')
testing  = pd.read_csv('Testing.csv')
# saving the information of columns
cols     = training.columns
cols     = cols[:-1]
# Slicing and Dicing the dataset to separate features from predictions
x        = training[cols]
y        = training['prognosis']
y1       = y

# dimensionality Reduction for removing redundancies
reduced_data = training.groupby(training['prognosis']).max()

# encoding/mapping String values to integer constants
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting-the-dataset-into-training-set-and-test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# print(x_test)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)
#greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

#implement the Decision-Tree-Classifier
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
# checking the Important features
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return disease

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    
    from sklearn.tree import _tree  # Fix: import _tree

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]  #storing disease name from the file
            threshold = tree_.threshold[node]
#             print(threshold)
            print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: "+ name + " ?")
            ans = input()
            ans = ans.lower()
            if ans == 'yes':
                val = 1
            elif ans == 'no':
                val = 0
            else:
                print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: I am sorry! I don't understand you")
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            
            present_disease = print_disease(tree_.value[node])
            for di in present_disease:
                diss=di
            for i in symptoms_present:
                dis=i
            print( "MTU-HEALTH-CARE-AI-POWERED-CHATBOT: You may have " +diss)
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: symptoms present " + dis)
            data = pd.DataFrame({'OTHER SYMPTOMS': list(symptoms_given)})
            print(data)
            import csv
            with open('doc_consult.csv', 'r') as f:
                read = csv.reader(f)
            consult={}
            consult_doc=[' YES','NO']
            
            for row in read:
                consult[row[0]]=int(row[1]) #converting csv to dictionary
            if(consult[diss]>50):
                    print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: You should consult a doctor as soon as possible")
                    data1=[consult[diss],0]
                    plt.ylim([0,100])
                    plt.bar(consult_doc,data1,align='center',color='red',width=0.15)
                    plt.ylabel('Risk')
                    plt.xlabel('Consult a doctor')
                    plt.show()
                    
            else:
                print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: You may consult a doctor")
                data1=[0,consult[diss]]
                plt.ylim([0,100])
                plt.bar(consult_doc,data1,align='center',color='red',width=0.15)
                plt.ylabel('Risk')
                plt.xlabel('Consult a doctor')
                plt.show()
                
            
    recurse(0, 1)

flag=True
print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: My name is MTU-HEALTH-CARE-AI-POWERED-CHATBOT. I will answer your queries about your health related problem. If you want to exit, type Bye!")
while(flag==True):
    
    user_response=input()
    user_response=user_response.lower()
     
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: You are welcome..")
       
        else:
            if(greeting(user_response)!=None):
                print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: "+greeting(user_response))
                print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: Please reply Yes or No for the following symptoms")
                tree_to_code(clf,cols)
            else:
                print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: I am sorry! I don't understand you")
                flag=True
                
    else:
        flag=False
        print("MTU-HEALTH-CARE-AI-POWERED-CHATBOT: Bye! take care..")    

# --- Data/model loading ---
@st.cache_resource
def load_data_and_model():
    training = pd.read_csv('Training.csv')
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    reduced_data = training.groupby(training['prognosis']).max()
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    clf = DecisionTreeClassifier().fit(x, y_enc)
    return clf, le, cols, reduced_data

clf, le, cols, reduced_data = load_data_and_model()

@st.cache_data
def load_consult():
    df = pd.read_csv('doc_consult.csv', header=None, index_col=0)
    consult = df[1].to_dict()
    return consult
consult = load_consult()

# --- UI/UX setup ---
st.set_page_config(page_title="MTU Health Care Chatbot", page_icon="💬", layout="centered")
st.sidebar.title("About")
st.sidebar.info(
    "This is an AI-powered health chatbot using a decision tree. "
    "It will ask you about your symptoms and suggest possible conditions. "
    "All data is local and private."
)
st.sidebar.markdown("**Instructions:**\n- Greet the bot to start\n- Answer Yes/No to symptoms\n- Get your result and advice")

st.title("💬 MTU Health Care AI Powered Chatbot")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = "greet"
if "tree_path" not in st.session_state:
    st.session_state.tree_path = []
if "symptoms_present" not in st.session_state:
    st.session_state.symptoms_present = []
if "disease" not in st.session_state:
    st.session_state.disease = None
if "other_symptoms" not in st.session_state:
    st.session_state.other_symptoms = []

# --- Helper functions ---
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

# --- Chat UI ---
def display_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

display_messages()

if st.session_state.step == "greet":
    user_input = st.text_input("Type your greeting to start:", key="greet_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        if user_input.lower() in ["bye", "exit"]:
            st.session_state.messages.append({"role": "bot", "content": "Bye! take care.."})
            st.session_state.step = "done"
        elif greeting(user_input) is not None:
            bot_greet = greeting(user_input)
            st.session_state.messages.append({"role": "bot", "content": bot_greet})
            st.session_state.messages.append({"role": "bot", "content": "Please reply Yes or No for the following symptoms"})
            st.session_state.step = "symptoms"
        else:
            st.session_state.messages.append({"role": "bot", "content": "I am sorry! I don't understand you"})
        st.experimental_rerun()

elif st.session_state.step == "symptoms":
    symptom, node = get_next_symptom()
    if symptom:
        st.session_state.messages.append({"role": "bot", "content": f"Do you have {symptom}?"})
        col1, col2 = st.columns(2)
        if col1.button("Yes", key=f"yes_{node}"):
            st.session_state.tree_path.append(1)
            st.session_state.symptoms_present.append(symptom)
            st.experimental_rerun()
        if col2.button("No", key=f"no_{node}"):
            st.session_state.tree_path.append(0)
            st.experimental_rerun()
    else:
        diss, other_symptoms = get_disease_and_symptoms(node)
        st.session_state.disease = diss
        st.session_state.other_symptoms = other_symptoms
        st.session_state.step = "result"
        st.experimental_rerun()

elif st.session_state.step == "result":
    diss = st.session_state.disease
    other_symptoms = st.session_state.other_symptoms
    st.session_state.messages.append({"role": "bot", "content": f"You may have **{diss}**"})
    st.session_state.messages.append({"role": "bot", "content": "Symptoms present: " + ', '.join(st.session_state.symptoms_present)})
    st.session_state.messages.append({"role": "bot", "content": "Other symptoms: " + ', '.join(other_symptoms)})
    risk = consult.get(diss, 0)
    consult_doc = ['YES', 'NO']
    if risk > 50:
        st.session_state.messages.append({"role": "bot", "content": "You should consult a doctor as soon as possible"})
        data1 = [risk, 0]
    else:
        st.session_state.messages.append({"role": "bot", "content": "You may consult a doctor"})
        data1 = [0, risk]
    fig, ax = plt.subplots()
    ax.bar(consult_doc, data1, color='red', width=0.15)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Risk')
    ax.set_xlabel('Consult a doctor')
    st.pyplot(fig)
    if st.button("Restart"):
        for key in ["messages", "step", "tree_path", "symptoms_present", "disease", "other_symptoms"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

elif st.session_state.step == "done":
    st.session_state.messages.append({"role": "bot", "content": "Session ended. Refresh to start again."})
    display_messages()    