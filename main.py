import streamlit as st
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI
from streamlit import session_state as ss
import urllib.parse
from utils.helper import get_app_base_url, dump_data_to_db
from utils.openai_helper import score_reviews
import pandas as pd
import time

load_dotenv()

####-------------------- Data base and openai client initialization------------------------------
username = st.secrets['MONGODB_USERNAME']
password = st.secrets['MONGODB_PASSWORD']
cluster = st.secrets['CLUSTER']
ENV = st.secrets.get('ENV', None)
MODEL = st.secrets.get('MODEL', None)
escaped_username = urllib.parse.quote_plus(username)
escaped_password = urllib.parse.quote_plus(password)
if ENV == "prod":   
    DB_URL=f"mongodb+srv://{escaped_username}:{escaped_password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0"
    model = MODEL or "gpt-4-turbo-preview"
else:
    DB_URL="mongodb://localhost:27017"
    model = MODEL or "gpt-3.5-turbo"

ss['model_name'] = model

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
mongo_client = MongoClient(DB_URL)
db = mongo_client['reviews']

collection = None
if 'collection' not in ss:
    ss['collection'] = db['amazon_reviews']
    
   
_open_ai_model = ChatOpenAI(model=model, temperature=0.5, max_tokens=60)
openai_client = OpenAI(
    api_key=st.secrets['OPENAI_API_KEY'],
)
ss['open_ai_model'] = _open_ai_model    
ss['openai_client'] = openai_client    

### ---------------------------------------------------------------------------------------------

# Get url query params
query_params = dict(st.query_params)
print(f"Query Params: {query_params}")

data_file_id = query_params.get('data_file_id', None)

if not data_file_id:
    uploaded_file = st.file_uploader("Choose a Classifed labels CSV file", type="csv")

    st.write("Or u can enter data file id, if data is already classified.")

    data_file_id = st.text_input("**Enter Data File Id:**", "")
                            
data_rows = []
collection = ss['collection']
if data_file_id:
    ss["data_file_id"] = data_file_id
    data_rows = collection.find_one({"data_file_id" : data_file_id})
    if not data_rows:
        st.toast(f"No Data found for data_file_id: {data_file_id}")
        data_rows = []

data_exists = len(data_rows) > 0

ss["attributes"] = ["Build Quality", "Price", "Comfort", "Design", "Battery life", "Sound Quality"]
    
ss["data_exists"] = data_exists

if 'uploaded_file' not in ss:
    ss['uploaded_file'] = None

if data_exists:
    st.switch_page("pages/dashboard.py")
elif data_file_id and not data_exists:
    st.toast(f"No Data found for data_file_id: {data_file_id}")
  

if uploaded_file:
        
    attributes = ss["attributes"]       
    app_base_url = get_app_base_url()
    data_file_id = int(time.time())
                
    input_df = pd.read_csv(uploaded_file)
    
    # ss["attributes"] = attributes = get_attributes(input_df, _open_ai_model)
    
    # print(f"Attributes generated: {attributes}")    

    # To run classification need to un comment this 

    with st.spinner('Generating Classification Labels, please wait...'):
        scores_generated_df = score_reviews(input_df, _open_ai_model, attributes)
        
    # scores_generated_df.to_csv('scores_generated_new.csv', index=False)
    
    # This can be commented out to run classification
    # scores_generated_df = input_df  # = pd.read_csv('scores_generated.csv')
    
    data_rows = dump_data_to_db(scores_generated_df, data_file_id)
    
    url = f"{app_base_url}?data_file_id={data_file_id}"
    st.write(f"Data can be viewed with this url: {url}" )    
    ss["app_url"] = url
    ss["data_file_id"] = data_file_id    
    ss["data_exists"] = True
    ss['data_rows'] = data_rows 
    

    
    st.switch_page("pages/dashboard.py")


print("-----------------End of main.py-----------------")        
        
        
          
    
    
    
