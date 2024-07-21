from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import streamlit as st
from langchain_community.llms import Ollama

st.set_page_config(layout="wide")


page_bg_img = '''
<style>
.reportview-container {
        background: url("https://machinelearningmastery.com/wp-content/uploads/2014/11/How-to-become-a-data-scientist-994x1024-1.jpg")
    }
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


st.error("""Hi there! I'm Gal Hever ✌️ """, icon="📕")


st.markdown("""
        <div class="circle-image">
           <p>
                <a href="https://www.linkedin.com/in/gal-hever/">
                    <img src="data:image/jpeg;base64,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">
                </a>
            </p>
        </div>
        <style>
          .circle-image {
              width: 200px;
              height: 200px;
              border-radius: 50%;
              overflow: hidden;
              box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
          }

          .circle-image img {
              width: 100%;
              height: 100%;
              object-fit: cover;
          }
        </style>

        """, unsafe_allow_html=True)


with st.expander("Quick intro"):
    st.markdown("MSc in Data Science, with over a decade of accumulated expertise in Machine Learning & Data Analytics from 8200, academy, and industry. /n"
            " Deploying algorithms to production by applying data-driven Machine Learning & AI solutions end to end, starting from research to development and testing")

st.info(""" Down here I built my cool portfolio chatbot! 🤖 so, let's get to know each other a bit more!️ """)
st.info("""  I know that you were about to ask that so, yes behind the scene I'm using Ollama and RAG! """)


llm = Ollama(model="llama2")
embedding_model = OllamaEmbeddings(model="llama2")


def get_docs():
    data = [
        "https://medium.com/@galhever/about",
        "https://www.linkedin.com/in/gal-hever",
        "https://www.womenonstage.net/speakers/gal-hever",
    ]

    docs = []
    for doc in data:
        loader = WebBaseLoader(doc)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    return documents


def generate_response(input):
    documents = get_docs()
    vector_index = FAISS.from_documents(documents, embedding_model)
    retriever = vector_index.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a woman that is called Gal Hever. You like to dance and you have a husband called Avi and 3 children.
        You live in Tzoran.
        Answer questions as Gal Hever based on your internal knowledge and the provided context.
        If you are not sure then say you need to think about it.
        <context>
        {context}
        </context>

        Question: {input}

        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    res = retrieval_chain.invoke({"input": input})
    st.info(res["answer"])


with st.form('my_form'):
    text = st.text_area('Enter text:', 'Can you tell me about Gal Hever?')
    submitted = st.form_submit_button("submit")
    if submitted:
        generate_response(text)





st.sidebar.caption('Wish to connect?')
st.sidebar.write('📧: galhever@gmail.com')

pdfFileObj = open('files/Gal Hever - Resume.pdf', 'rb')
st.sidebar.download_button('download resume', pdfFileObj, file_name='Gal Hever - Resume.pdf', mime='pdf')

#
# # st.sidebar.write('LinkedIn | Github | Medium')
# st.sidebar.markdown("[![Foo](data:image/png;base64,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)](https://www.linkedin.com/in/gal-hever/)")
# st.sidebar.markdown("[![Foo](https://www.webfx.com/wp-content/uploads/2022/08/github-logo.png)](https://github.com/galhev)")
# st.sidebar.markdown("[![Foo](https://media.licdn.com/dms/image/D4E0BAQGLNpn-roUX0g/company-logo_200_200/0/1719258074326/datanights_logo?e=2147483647&v=beta&t=CAlWcS5k_x_tyJ1nZu9PMJFUypft-Q-CMq2Idcjevvs)](https://datanights-il.github.io/)")
# st.sidebar.markdown("[![Foo](https://media.licdn.com/dms/image/D4D0BAQFFNmFlKiblbg/company-logo_200_200/0/1688488446450/datahack_logo?e=2147483647&v=beta&t=krNwU4cf522tPtzbF-zxDc-7elz-VgxtZ8C7R4G6prY)](https://www.youtube.com/@DataHackIL)")
# st.sidebar.markdown("[![Foo](https://computercity.com/wp-content/uploads/1_jfdwtvU6V6g99q3G7gq7dQ1.png)](https://medium.com/me/stories/public)")
# st.sidebar.markdown('''[<img src='IMG_5452.jpg' width="250"/>](https://medium.com/me/stories/public)''')

