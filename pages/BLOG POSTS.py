import pandas as pd
import streamlit as st
import os
import streamlit.components.v1 as components

st.title('My Blogs')
st.error(""" I'm writing a professional blog on [Medium](https://medium.com/me/stories/public) and here you can have a list of my previous blog-posts divided into sub-themes """, icon="📕")

path = os.getcwd()+'/files/my_blogs.csv'
df = pd.read_csv(path)
df['category1'] = df.apply(lambda x:x['category1'].split('|'),axis=1)
df = df.explode('category1')
grouped = df.groupby(['category1']).agg(list)
grouped['total'] = grouped['url'].transform(len)
grouped.at['Other technical','total'] = 0
grouped.at['Other non technical','total'] = 0
grouped = grouped.sort_values(by='total',ascending=False)
for x,y in grouped.iterrows():
    with st.expander(x.upper()):
        blog = {a:b for a,b in zip(y['title'],y['url'])}
        for a,b in blog.items():
            st.markdown("""<a href={}><b><u>{}</b></u></a>""".format(b,a),unsafe_allow_html=True)