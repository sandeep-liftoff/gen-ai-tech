import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import math
from streamlit import session_state as ss
import urllib.parse

@st.cache_data(ttl= None)
def get_app_base_url():
    # To get the app base url
    session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
    app_base_url = urllib.parse.urlunparse([session.client.request.protocol, session.client.request.host, "", "", "", ""])
    return app_base_url

@st.cache_data(ttl= None)
def preprocess_data(_data_rows):    
    rows = []
    for row in _data_rows:
        product = row['Product Name']
        scores = row['label_scores']
        for item in scores:
            key, value = item['label'], item['score']
            rows.append([product, key, value])
    new_df = pd.DataFrame(rows, columns=['Product Name', 'Label', 'Score'])
    return new_df

def dump_data_to_db(df, data_file_id):
    
    collection = ss["collection"]
    df = df.dropna(subset=['scores'])
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
        row_dict['data_file_id'] = str(data_file_id)
        del row_dict['scores']
        rows.append(row_dict)
    if rows:
        collection.insert_many(rows)
        print(f"{len(rows)} Rows inserted to DB successfully")
    return rows

@st.cache_data(ttl= None, show_spinner=False)
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


def reset_data():
    data_file_id = ss["data_file_id"]
    collection = ss["collection"]
    collection.delete_many({"data_file_id" : data_file_id})            
    ss["data_exists"] = False
    st.empty()
    st.rerun()

def show_table_data(data_rows):
    
    st.subheader("Sample JSON Data")
    st.write(
        """<style>
        td { 
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            width: 100px;
        } 
        th {
            width: 200px;
        }
        .stTextInput p {
                font-size: 18px;
                padding: 10px; 
            }
        </style>""",
        unsafe_allow_html=True
    )
    st.table(data_rows[:5])
    st.button("Delete Data", on_click=reset_data)
        

