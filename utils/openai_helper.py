import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed,
    wait_random
)
from streamlit import session_state as ss

import re
from collections import Counter

output_parser = StrOutputParser()

backoff_times = {
    "classification" :{
        "wait": wait_random_exponential(0.15, 0.35),
        "stop": stop_after_attempt(20)
    },
    "attribute_extraction" :{
        "wait": wait_random(0.15, 0.20),
        "stop": stop_after_attempt(5)
    }
}


@retry(wait=backoff_times["classification"]["wait"], stop=stop_after_attempt(5),reraise=True)
def run_classification_with_backoff(chain, **kwargs):
    return chain.batch(**kwargs)

@retry(wait=backoff_times["attribute_extraction"]["wait"], stop=stop_after_attempt(5),reraise=True)
def run_attribute_extraction_with_backoff(chain, **kwargs):
    return chain.batch(**kwargs)

@st.cache_data( show_spinner=False)
def score_reviews(reviews_df, attributes):
    model_name = ss['model_name']
    op_structure = "feature:score"
    prompt = ChatPromptTemplate.from_template(
                            """
                            Here is a product review: {review}
                            Determine how the reviewer rates this product in relation to these features:  
                            %s
                            Only provide a score (between 0 to 5). If the feature is not mentioned, provide a score of 0. Use the following format. No additional commentary.
                            %s
                            """ % (', '.join(attributes), op_structure)
                        )
    
    model = ChatOpenAI(model=model_name, temperature=0.5, max_tokens=45)

    chain = (
            {"review": RunnablePassthrough()} 
            | prompt
            | model
            | output_parser
    )
    review_texts = reviews_df['text'].to_list()
    review_texts = [str(text) for text in review_texts]
    # restrict all reviews to 12000 characters
    review_texts = [text[:12000] for text in review_texts]
    res = run_classification_with_backoff(chain, inputs = review_texts, config={"max_concurrency":10})
    for i, row in reviews_df.iterrows():
        reviews_df.at[i, 'scores'] = res[i]

    return reviews_df

def filter_special_characters(attribute):
    return re.sub(r'[^a-zA-Z\s]', '', attribute).strip()

@st.cache_data( show_spinner=False)
def get_attributes(reviews_df):
    model_name = ss['model_name']
    # pick only 500 reviews per asin
    reviews_df = reviews_df.groupby('asin').head(500)
    prompt = ChatPromptTemplate.from_template(
        """
        Here is a product review of a bluetooth speaker: {review}

        Extract the features that the reviewer mentions that describes the product.

        For example: 'sound quality', 'price'.

        Only provide product features that are mentioned in the review. Do not return adjectives or other descriptive verbs. Do not return entire sentences. Return only nouns.

        Do not return user emotions such as 'love it', 'hate it', 'very useful' etc.

        Make sure that each feature is at most 2 words long. Don't return features longer than this.

        Make sure that the features returned make sense as an attribute of a bluetooth speaker. For example, 'kitchen use' is not a valid feature. 'ease of connection' is not a valid feature. 'sound quality' is a valid feature. 'connectivity' is a valid feature.

        Do not return duplicates. For Eg: if the review mentions a feature in relation to multiple scenarios, return just the characteristic. If the speaker talks about connectivity with TV and phone,  do not return both of them. Just return "connectivity". 

        Disregard the other products in all reviews. 

        Return maximum 5 features per review. If there are more than 5 features, return the top 5 features.

        """
    )
    output_parser = StrOutputParser()
    model = ChatOpenAI(model=model_name, temperature=0, max_tokens=45, top_p=1)

    chain = (
            {"review": RunnablePassthrough()}
            | prompt
            | model
            | output_parser
    )
    review_texts = reviews_df['text'].to_list()
    review_texts = [str(text) for text in review_texts]
    # restrict all reviews to 12000 characters
    review_texts = [text[:12000] for text in review_texts]
    attributes = run_attribute_extraction_with_backoff(chain, inputs = review_texts, config={"max_concurrency":10})

    #  return unique attributes
    attributes = list(set(attributes))
    list_of_attr = []
    for attr in attributes:
        attr = filter_special_characters(attr)
        attrs = attr.split('\n')
        list_of_attr.extend([at.strip().lower() for at in attrs])

    attribute_counts = Counter(list_of_attr)
    top_5_attributes = [attr for attr, _ in attribute_counts.most_common(5)]
    print(f"Attributes generated: {list_of_attr}")
    print("\n\n")
    print(top_5_attributes)

    return top_5_attributes

def generate_summary_input(filtered_data):
    each_label_text = []
    for index, row in filtered_data.iterrows():
        each_label_text.append(f"- {row['Label']}: {row['Score']}/10\n")
    
    return each_label_text

@st.cache_data(ttl= None, show_spinner=False)
def get_summary_from_openai(data, _llm):
    prompt_template = """Write a concise summary of the following product reviews for the earbuds:
                    {text}
                    CONSCISE SUMMARY:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    docs = [Document(page_content=t) for t in data]
    chain = load_summarize_chain(_llm, chain_type='stuff', prompt=prompt)
    
    result = chain.invoke(docs)
    return result['output_text']

def generate_summary_for_json_ouput(llm, data):
    model_name = ss['model_name']
    response = llm.chat.completions.create(
        model=model_name,
        messages=[
        {
            "role": "system",
            "content": f"Please convert the following JSON output to a human-readable sentence"
        },
        {
            "role": "user",
            "content": f"{data}"
        }
        ],
        temperature=0.7,
        top_p=1
    )
    result = response.choices[0].message.content
    return result


def get_aggregation_query_from_openai(llm, query, single_row_data):
  # print(single_row_data)
  model_name = ss['model_name']
  response = llm.chat.completions.create(
    model=model_name,
    messages=[
      {
        "role": "system",
        "content": f"Given the following MongoDB document structure : \n {single_row_data} \n . Provide a MongoDB aggregation query in json format as only array. Just return the [] of aggregration pipeline. Make sure to enclose the property name in double quotes as it is pure JSON );"
      },
      {
        "role": "user",
        "content": f"{query}"
      }
    ],
    temperature=0.7,
    top_p=1
  )
  db_query = response.choices[0].message.content
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", "Given the following MongoDB document structure : \n {single_row_data} \n . Provide a MongoDB aggregation query in json format as only array. Just return the [] of aggregration pipeline. Make sure to enclose the property name in double quotes as it is pure JSON );"),
#       ("user", "{query}")
#   ])
#   chain = prompt | llm | output_parser
#   db_query = chain.invoke({"query": query, "single_row_data": single_row_data})
  return db_query