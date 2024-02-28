import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pymongo import MongoClient

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI
from dotenv import load_dotenv
from openai_helper import score_reviews, generate_summary_input, get_summary_from_openai, get_attributes, get_aggregation_query_from_openai
import math
import json
load_dotenv()

_open_ai_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=60)
_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

db_url = os.environ.get("DB_URL")
mongo_client = MongoClient(db_url)
db = mongo_client['reviews']
collection = db['amazon_reviews']

collection.delete_many({})

st.set_page_config(layout="wide")

all_products_summary = {}
chart_data = {}
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

col1, col2, col3 = st.columns(3)

with col1:
    uploaded_file = st.file_uploader("Choose a Classifed labels CSV file", type="csv")
    
    
def preprocess_data(data_rows):    
    rows = []
    for row in data_rows:
        product = row['Product Name']
        scores = row['label_scores']
        for key, value in scores:
            rows.append([product, key, value])

    new_df = pd.DataFrame(rows, columns=['Product Name', 'Label', 'Score'])
    return new_df


def dump_data_to_db(df):
    
    rows = []
    for index, row in df.iterrows():
        scores = row['scores'].split('\n')
        label_scores_list = []        
        for score in scores:
            if score:
                parts = score.split(':') 
                if len(parts) == 2:
                    key, value = parts
                    key = key.strip()  
                    value = value.strip()
                    try:
                        value = math.floor(float(value))
                        label_scores_list.append({"label":key, "score":value})
                    except ValueError:
                        # Handle cases where value cannot be converted to float
                        pass
        row_dict = row.to_dict()
        row_dict['label_scores'] = label_scores_list
        del row_dict['scores']
        del row_dict['images']
        del row_dict['variant:']
        del row_dict['variant:size']
        rows.append(row_dict)

    collection.insert_many(rows)
    
    return rows



@st.cache_data( show_spinner=False)
def make_radar_chart(title, categories, values):
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8,10], ["2","4","6","8","10"], color="grey", size=7)
    plt.ylim(0,5)
    
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=title)
    
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(title, size=11, color='blue', y=1.1)
    
    return fig

if uploaded_file is not None:
    
    input_df = pd.read_csv(uploaded_file)
    
    # attributes = get_attributes(input_df, _open_ai_model)
    
    # print(f"Attributes generated: {attributes}")
    
    attributes = ["Build Quality", "Price", "Comfort", "Design", "Battery life", "Sound Quality"]

    # To run classification need to un comment this 

    with st.spinner('Generating Classification Labels, please wait...'):
        scores_generated_df = score_reviews(input_df, _open_ai_model, attributes)
        
    # scores_generated_df.to_csv('scores_generated.csv', index=False)
    
    # This can be commented out to run classification
    # scores_generated_df = input_df = pd.read_csv('scores_generated.csv')
    
    data_rows = dump_data_to_db(scores_generated_df)
    
    df = preprocess_data(data_rows)
        
    # df.to_csv('preprocessed_data.csv', index=False)
    
    product_list = df['Product Name'].unique()
    with col2:
        selected_product = st.selectbox('Select a product:', product_list)
        
        filtered_data = df[df['Product Name'] == selected_product]
            
        avg_scores = filtered_data.groupby('Label')['Score'].mean().reindex(attributes)
                
        with st.spinner('Generating Chart, please wait...'):
            if selected_product not in chart_data:
                chart_data[selected_product] = make_radar_chart(selected_product, list(avg_scores.index), list(avg_scores.values))            
            st.write(f"{selected_product} Radar Chart:")
            st.pyplot(chart_data[selected_product])
    with col3:
        with st.spinner('Generating summary, please wait...'):
            if selected_product not in all_products_summary: 
                summary_input = generate_summary_input(filtered_data)
                all_products_summary[selected_product] = get_summary_from_openai(summary_input, _llm)       
            
            st.write(f'{selected_product} Summary:')         
            st.write(all_products_summary[selected_product])
            
    
    with col1:
        st.subheader("Sample JSON Data")
        st.write(
            """<style>
            td { 
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            } 
            .stTextInput p {
                    font-size: 18px;
                    padding: 10px; 
                }
            </style>""",
            unsafe_allow_html=True
        )
        st.table(data_rows[:5])
        
        single_row_json = data_rows[0]
        
        user_query = st.text_input("Enter Query:", "")
        print(f"User Query: {user_query}")
        if st.button("Submit"):  
            max_attempts = 3
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                query = get_aggregation_query_from_openai(client, user_query, single_row_json)
                print(f"Query from OpenAI: {query}")
                try:
                    query = json.loads(query)
                    
                    # Attempt to fetch data from MongoDB
                    result = collection.aggregate(query)
                    st.subheader("Result from MongoDB:")
                    for doc in result:
                        st.write(doc)
                    break  # Exit the loop if successful
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON: {e}")

        
        
        
          
    
    
    
