import streamlit as st

# st.header('My Lectures',  divider='grey')
st.title('My Lectures')

st.error(""" So, I'm also a public speaker and here I'll share with you my previous talks """, icon="🎤")



st.info("You can get more info by pressing on each of the expanders below")

st.subheader('Speech Recognition')
with st.expander('WER We Are?'):
    st.info("In the last decade voice technologies have become ubiquitous in everyday lives from our personal devices and household appliances all are voice activated. Speech revolution is already here and in the near future as those technologies become robust they will be incorporated into our everyday surroundings. Where did we start this decade and where are we now? What is the future of those technologies and are they really going to change our lives?")
    st.video("https://www.youtube.com/watch?v=zk7fLml4XYg&t=3s")

with st.expander('An Overview of Modern Speech Recognition'):
    st.info(""" Automatic speech recognition has been impacted by advances in related fields like image processing and natural language processing in recent years. One notable achievement in these areas has been the use of self-supervised learning to improve performance in computer vision and NLP tasks. This led to the development of the first self-supervised language model for speech representations, which has demonstrated impressive results in various NLP tasks. In this talk, we will review the key principles of automatic speech recognition and discuss the current progress, research, and challenges in the field. """)

with st.expander('The Art of Sound'):
    st.info(""" In this super cool workshop we are going to dirt our hands with the sound spectrum!
We'll be discovering the magic of sound and see how different signals can be visualized into different kind of acoustic features. We'll start from the basic waveform and continue with time-frequency representations such as the spectogram and MFCC. In the end of this workshop we'll even compose our own sounds, can you guess how? """)

with st.expander('The Race Gap in Speech Recognition Technology'):
    st.info(""" The importance of spoken language in our daily lives highlights the need for speech recognition systems that can effectively handle a wide range of speaking styles, including variations due to speaker demographics, first vs. second language usage, and ability level. However, the current state of these systems often fails to adequately serve non-native speakers or those with distinct accents. This talk will explore the impact of these shortcomings and encourage the development of inclusive speech recognition technologies that can accurately transcribe and understand a diverse range of languages and speaking styles. """)


st.subheader('LLMs')
with st.expander('Hands-On Session: Build your Minime with LangChain, LLM & Streamlit'):
    st.info(""" Imagine having a virtual minime that not only knows all about you but can also recommend new things that you might like to do in your free time!

With just a few simple steps we will build your Portfolio Chatbot with Streamlit and LangChain. You will be able to deploy your Potfolio Chatbot to Streamlit Community Cloud and share it with friends all around the globe! """)


st.subheader('Natural Language Processing')
with st.expander('Introduction to Natural Language Processing'):
    st.info("Have you ever wondered if computers can understand human languages? Since 1950, people have been trying to crack the secret of understanding the human language. In his famous Turing test, Alan Turing defined conversation between a human being and a computer as a criterion for intelligence. The complexity of understanding the human language can prove his argument. In this lecture we will explain how to deal with textual data, cover the main architectures for analyzing text and show how things work in practice.")

with st.expander('Traditional Sequence Models - An Overview'):
    st.info("What is a sequence model? How it works and which problem it can solve in NLP domain? This lecture will overview sequence models such as RNNs LSTM and GRU. We will go over the challenges of processing texts, capabilities and limitations of each model and why now they are less common than the new players that entered into the industry.")

with st.expander('The secret behind the Transformers'):
    st.info("Have you ever wondered what is the secret behind the attention mechanism? And what are the magic of the Transformers? The last trends of the NLP world enable in solving different problems from text generation to machine translation and more. In this talk we will discuss the concept of attention and how the Transformer uses this mechanism. We will go deep inside and understand the magic behind the the transformer and how it actually works with real data.")


st.subheader('Python Frameworks')
with st.expander('Getting Started with PyTorch'):
    st.info("Want to get to grips with PyTorch? This hands-on workshop will take you all the way from basics to advanced concepts with one of the most popular machine learning libraries for deep learning even if you’re starting from scratch. We will start with a short introduction to deep learning frameworks in general and then go deep into PyTorch syntax and understand how does it work behind the scenes. You will be learning how to define a network architecture and train a model and even solve a real-world problem.")

with st.expander('Hands-On Session: Build Your First Visualization Tool'):
    st.info(""" In this workshop, we will learn how to build an interactive web-based data visualization tool in Python which includes a three-hour-long Plotly Streamlit and Dash technical session. The workshops will go over a few topics such as tools overview, possible usages, and coding in groups. """)

with st.expander('Hands-On Session: Faster way to Build and Share Data Apps'):
    st.info(""" One-hour-long zero to hero technical session, in which we will learn how to build an interactive web-based data visualization tool in Python! We will work with each one of the frameworks Plotly, Streamlit and Dash and in the end of this workshop you’ll be a data apps ninja! """)


st.subheader('Soft Skills')
with st.expander('Four Types of Employees'):
    st.info('')
