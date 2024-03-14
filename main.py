import argparse
import os

import requests
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.docstore.document import Document
from lxml import etree
import pandas as pd

os.environ["OPENAI_API_KEY"] = "<OPEN-AI API Key>"
os.environ[
    "ANTHROPIC_API_KEY"] = "<Anthropic-AI API Key>"
namespaces = {'dc': 'http://purl.org/dc/elements/1.1/', 'content': 'http://purl.org/rss/1.0/modules/content/'}


def get_feed_response(url):
    response = requests.get(url)
    resp_list = []
    if response.status_code == 200:
        # Parse the XML content
        tree = etree.fromstring(response.content)
        print('items are ', len(tree.findall('.//item')))
        for item in tree.findall('.//item'):
            # For each item, find the <title> element and print its text content
            title_element = item.find('title')
            title_text = title_element.text if title_element is not None else 'No title found'

            creator_element = item.find('dc:creator', namespaces)
            creator_text = creator_element.text if creator_element is not None else 'No creator found'

            article_link_element = item.find('link')
            article_link_text = article_link_element.text if article_link_element is not None else 'No link found'

            content_element = item.find('content:encoded', namespaces)
            content_text = content_element.text if content_element is not None else ''

            print(f'Data: {title_text}')
            resp_list.append({"title": title_text,
                              "creator": creator_text,
                              "content": content_text,
                              'medium_only': "Yes" if content_text == "" else "No",
                              'link': article_link_text})

    else:
        print("Failed to fetch the XML feed. Status code:", response.status_code)

    return resp_list


def persist_data_in_chroma(persist_directory, data):
    # various supported model options are available
    embedding_function = OpenAIEmbeddings()

    # Chroma instance takes in a directory location, embedding function and ids-> list of documentIds
    chroma_db = Chroma.from_documents(data,
                                      embedding_function, persist_directory)

    return chroma_db


def create_llm(model_name, **kwargs):
    # Map model names to their corresponding class constructors
    model_map = {
        "claude-3-sonnet-20240229": ChatAnthropic,
        "gpt-3.5-turbo": ChatOpenAI
        # Add more models here
    }
    # Get the constructor for the given model_name or raise a ValueError if not found
    model_constructor = model_map.get(model_name)
    if model_constructor:
        return model_constructor(model_name=model_name, **kwargs)
    else:
        raise ValueError("Unsupported model_name")


def summarize_data(db, llm):
    print("summarizing data with llm : ", llm)

    chatbot_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1})
    )

    template = """
    respond as clearly as possible {query}?
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    summary = chatbot_chain.run(
        prompt.format(query="summarize the article text in 100 words, ignore any html styling information?")
    )
    return summary


def update_excel_row(file_path, row_index, summary, model_name, response):
    # Load the Excel file
    df = pd.read_excel(file_path)
    print('length: ', df)

    # Check if row_index is valid
    if row_index > len(df):
        raise IndexError("The row_index is out of bounds of the Excel file")

    # Update the 'Title' and 'Description' columns for the specified row
    df.at[row_index, 'Title'] = response['title']
    df.at[row_index, 'Link'] = response['link']
    df.at[row_index, 'Creator'] = response['creator']
    df.at[row_index, 'Model'] = model_name
    df.at[row_index, 'Summary'] = summary
    df.at[row_index, 'Member Only'] = response['medium_only']

    # Write the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False)


def process_args():
    parser = argparse.ArgumentParser(description="Process model_name and file_path values")
    parser.add_argument('model_name', type=str, help='The name of the model to use')
    parser.add_argument('--file_path', type=str, default="MediumSummaries.xlsx",
                        help='The path to the Excel file with YouTube summaries')
    return parser


def process_response(resp, persist_directory='db'):
    content = resp['content']
    if content == "":
        return "No Summary Available"
    print('data value is: ', content)
    data = Document(page_content=content, metadata={"source": "local"})
    db = persist_data_in_chroma(persist_directory, [data])
    llm = create_llm(model_name, temperature=0, max_tokens=500)
    return summarize_data(db, llm)


if __name__ == '__main__':
    args = process_args().parse_args()
    file_path = args.file_path
    model_name = args.model_name

    response_list = get_feed_response("https://ai.plainenglish.io/feed")
    print("response list length", len(response_list))
    for index, response in enumerate(response_list):
        summary = process_response(response, 'db')
        update_excel_row(file_path, index, summary, model_name, response)
