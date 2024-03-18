import streamlit as st
from streamlit import session_state as ss
from utils.helper import preprocess_data, make_radar_chart, get_app_base_url
from utils.openai_helper import generate_summary_input, get_summary_from_openai, get_aggregation_query_from_openai, generate_summary_for_json_ouput
import json

fixed_questions = [
"What are the different products that are available",
"What is the best product for sound quality."
]
# Set page layout
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

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
data_exists = ss.get('data_exists', None)
collection = ss.get('collection', None)
_open_ai_model = ss.get('open_ai_model', None)
_openai_client = ss.get('openai_client', None)

if not data_exists:
    st.empty()    
    st.switch_page("main.py")    
    
col1, col2, col3 = st.columns(3)    

with col1:
    if st.button("Back", "primary"):
        st.cache_data.clear()
        st.switch_page("main.py")  

app_base_url = get_app_base_url()
data_file_id = ss["data_file_id"]
print(f"Data File ID: {data_file_id}")
data_rows = ss.get("data_rows", None)
if not data_rows:
    data_rows = collection.find({"data_file_id" : data_file_id})
    data_rows = list(data_rows)
    ss["app_url"] = app_url = f"{app_base_url}?data_file_id={data_file_id}"
    
app_url = ss.get('app_url', None)
st.write(f"Data can be viewed with this url: {app_url}" )    

print(f"---------Data rows Length ---{len(data_rows)}----")
# df.to_csv('preprocessed_data.csv', index=False)    
filtered_df = preprocess_data(data_rows)    

chart_data = {}
all_products_summary = {}
attributes = ss["attributes"]
collection = ss["collection"]
    
df = filtered_df
product_list = df['Product Name'].unique()
with col2:
    st.write(f"**Data File ID: {data_file_id}**")
    selected_product = st.selectbox('**Select a product:**', [f'{product}' for product in product_list])
    
    filtered_data = df[df['Product Name'] == selected_product]
    avg_scores = filtered_data.groupby('Label')['Score'].mean().reindex(attributes)
    with st.spinner('Generating Chart, please wait...'):
        chart_data[selected_product] = make_radar_chart(selected_product, list(avg_scores.index), list(avg_scores.values))            
        st.markdown(f"Radar Chart:")
        st.pyplot(chart_data[selected_product])
        
with col3:
    with st.spinner('Generating summary, please wait...'):
        summary_input = generate_summary_input(filtered_data)
        all_products_summary[selected_product] = get_summary_from_openai(_open_ai_model, summary_input, selected_product)       
        
    st.markdown(f"**{selected_product.upper()} Summary:**")         
    st.write(all_products_summary[selected_product])
        

with col1:
                                       
    single_row_json = collection.find_one()
    question_clicked = False
    user_query = ""
    st.write("**Select a question or enter a query:**")
    for question in fixed_questions:
        if st.button(question):
            user_query = question
            question_clicked = True
    user_query = st.text_input("**Enter Query:**", value=user_query)

    print(f"User Query: {user_query}")
    
    if st.button("Submit") or question_clicked and user_query:  
        with st.spinner('Getting data, please wait...'):
            
            max_attempts = 3
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                query = get_aggregation_query_from_openai(_openai_client, user_query, single_row_json)
                print(f"Query from OpenAI: {query}")
                try:
                    query = json.loads(query)
                    
                    # Attempt to fetch data from MongoDB
                    rows = collection.aggregate(query)
                    st.subheader("**Result from MongoDB:**")
                    
                    results = []
                    for doc in rows:
                        doc['_id'] = str(doc.get('_id', ''))
                        results.append(doc)
                    print(f"Results: {results}")
                    sum = generate_summary_for_json_ouput(_openai_client, results)  
                    st.write(sum)
                    break  # Exit the loop if successful
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    
            if attempts >= max_attempts:
                st.error("Failed to fetch data from MongoDB")
                st.write("")
                st.stop()
    

