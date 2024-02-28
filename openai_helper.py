import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

@st.cache_data( show_spinner=False)
def score_reviews(reviews_df, _open_ai_model, attributes):
    op_structure = "feature:score"
    prompt = ChatPromptTemplate.from_template(
                            """
                            Here is a product review: {review}
                            Determine how the reviewer rates this product in relation to these features:  
                            %s
                            Only provide a score (between 0 to 3). If the feature is not mentioned, provide a score of 0. Use the following format. No additional commentary.
                            %s
                            """ % (', '.join(attributes), op_structure)
                        )
    output_parser = StrOutputParser()
    
    chain = (
            {"review": RunnablePassthrough()} 
            | prompt
            | _open_ai_model
            | output_parser
    )
    review_texts = reviews_df['text'].to_list()
    # restrict all reviews to 12000 characters
    review_texts = [text[:12000] for text in review_texts]
    res = chain.batch(review_texts)
    for i, row in reviews_df.iterrows():
        reviews_df.at[i, 'scores'] = res[i]

    return reviews_df

@st.cache_data( show_spinner=False)
def get_attributes(reviews_df, _open_ai_model):
    op_structure = "feature1, feature2"
    prompt = ChatPromptTemplate.from_template(
                            """
                            Here is a product review: {review}
                            Extract the features that the reviewer mentions in relation to the product. 
                            For example: sound quality, price.
                            Only provide features that are product attributes and are mentioned in the review. Do not return adjectives or other descriptive verbs. Use the following format. No additional commentary.
                            Make sure to return only features which are present in atleast 5 different reviews.
                            %s
                            """ % op_structure
                        )
    output_parser = StrOutputParser()
    chain = (
            {"review": RunnablePassthrough()} 
            | prompt
            | _open_ai_model
            | output_parser
    )
    review_texts = reviews_df['text'].to_list()
    # restrict all reviews to 12000 characters
    review_texts = [text[:12000] for text in review_texts]
    res = chain.batch(review_texts)
    #  return unique attributes
    res = list(set(res))
    return res

def generate_summary_input(filtered_data):
    each_label_text = []
    for index, row in filtered_data.iterrows():
        each_label_text.append(f"- {row['Label']}: {row['Score']}/10\n")
    
    return each_label_text

@st.cache_data( show_spinner=False)
def get_summary_from_openai(data, _llm):
    prompt_template = """Write a concise summary of the following product reviews for the earbuds:
                    {text}
                    CONSCISE SUMMARY:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    docs = [Document(page_content=t) for t in data]
    chain = load_summarize_chain(_llm, chain_type='stuff', prompt=prompt)
    
    return chain.run(docs)


def get_aggregation_query_from_openai(client, query, single_row_data):
  # print(single_row_data)
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
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
  query = response.choices[0].message.content
  return query