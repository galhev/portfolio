import streamlit as st
from config import *
import pandas as pd
from PIL import Image
from streamlit_timeline import timeline
import plotly.express as px
import requests
import re

st.set_page_config(layout="wide")

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

image_style = """
        <div class="circle-image">
            <p>
                <a href="https://www.linkedin.com/in/gal-hever/">
                    <img src="https://miro.medium.com/v2/resize:fit:2400/1*8Cwf7DO4Tq4g5yydXezF2g.jpeg"/>
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

        """


st.title('About me')
about = """
\n
\n
\n
I have over a decade of accumulated expertise in Machine Learning & Data Analytics from the academy (MSc in Data Science), and industry. 
I have a diverse background working on a variety of projects with different data types. I believe that parallel fields in this domain can benefit each other. 
It was proven in the last couple of years, when NLP papers were inspired by Computer Vision world and the ASR domain was inspired by NLP works.  
During my studies at Ben-Gurion University, I co-founded the Tech7Juniors association, to promote youth in the periphery in tech domains. Besides that,
I am running DataNights, a non-profit program aimed at increasing the presence of under-represented populations in the AI world and the high-tech industry.
"""

col1, mid, col2 = st.columns([3,1,20])
with col1:
    st.markdown(image_style, unsafe_allow_html=True)
with col2:
    st.markdown(about)


st.subheader('Career snapshot')

with st.spinner(text="Building line"):
    with open('timeline.json', "r") as f:
        data = f.read()
        timeline(data, height=500)

st.subheader('Skills & Tools ⚒️')


def skill_tab():
    rows, cols = len(info['skills']) // skill_col_size, skill_col_size
    skills = iter(info['skills'])
    if len(info['skills']) % skill_col_size != 0:
        rows += 1
    for x in range(rows):
        columns = st.columns(skill_col_size)
        for index_ in range(skill_col_size):
            try:
                columns[index_].button(next(skills))
            except:
                break


with st.spinner(text="Loading section..."):
    skill_tab()

st.subheader('Education 📖')

st.write(pd.DataFrame({
    'Degree': ['B.Sc', 'M.Sc'],
    'Field of study': ['Industrail Engineering', 'Data Science'],
    'University': ['Ben-Gurion University of the Negev', 'Ben-Gurion University of the Negev'],
    'Start date': ["2012", "2015"],
    'End date': ["2016", "2017"],
    'Grade': [85, 93],
    'Description': ["""Student in Ben Gurion University, department of Industrial Engineering & Management, faculty of Engineering Sciences,  
Studying the 5th year in the accelerated B.Sc and M.Sc program (10 semesters) for excelling students.

Experience with reporting and Business Intelligence tools (e.g., Qlik Sense, Tableau).
Advanced working SQL knowledge and experience working with relational databases.
Experience in data analysis and visualization.

""", """ Studying the 5th year in the accelerated B.Sc and M.Sc program (10 semesters) for excelling students.
Experience as a Researcher with responsible for solving real problems from idea through prototype to deployment using data science techniques.
Coding experience with R such as RandomForest, Plotly, Shiny libraries.
Academic experience in machine-learning methods and statistical modeling techniques such as regression, clustering, classification, decision trees, neural networks.""" ],
}))

def plot_bar():
    st.info('Comparing Brute Force approach with the algorithms')
    temp1 = rapid_metrics.loc[['Brute-Force_Printed', 'printed'], :].reset_index().melt(id_vars=['category'],
                                                                                        value_vars=['precision',
                                                                                                    'recall',
                                                                                                    'f1_score'],
                                                                                        var_name='metrics',
                                                                                        value_name='%').reset_index()

    temp2 = rapid_metrics.loc[['Brute-Force_Handwritten', 'handwritten'], :].reset_index().melt(id_vars=['category'],
                                                                                                value_vars=['precision',
                                                                                                            'recall',
                                                                                                            'f1_score'],
                                                                                                var_name='metrics',
                                                                                                value_name='%').reset_index()

    cols = st.columns(2)

    fig = px.bar(temp1, x="metrics", y="%",
                 color="category", barmode='group')

    cols[0].plotly_chart(fig, use_container_width=True)

    fig = px.bar(temp2, x="metrics", y="%",
                 color="category", barmode='group')
    cols[1].plotly_chart(fig, use_container_width=True)


def image_and_status_loader(image_list, index=0):
    if index == 0:
        img = Image.open(image_list[0]['path'])
        st.image(img, caption=image_list[0]['caption'], width=image_list[0]['width'])

    else:
        st.success('C-Cube algorithm for printed prescriptions')
        rapid_metrics.loc[['Brute-Force_Printed', 'printed'], :].plot(kind='bar')
        cols = st.columns(3)
        for index_, items in enumerate(image_list[0]):
            cols[index_].image(items['path'], caption=items['caption'], use_column_width=True)

        st.success('3 step filtering algorithm for handwritten algorithms')
        cols = st.columns(3)
        for index_, items in enumerate(image_list[1]):
            cols[index_].image(items['path'], caption=items['caption'], use_column_width=True)

        plot_bar()


st.subheader('Achievements 🥇')
achievement_list = ''.join(['<li>' + item + '</li>' for item in info['achievements']])
st.markdown('<ul>' + achievement_list + '</ul>', unsafe_allow_html=True)

with st.expander('Get more info about DataNights'):
    st.markdown("""
    The [DataNights](https://www.linkedin.com/company/datanights/?originalSubdomain=il) program is an educational program designed for members of the Data Science community In Israel. This program runs as part of a collaboration between the [DataHack NGO](https://datahack.org.il/) - an amuta promoting data science and machine learning in Israel - and the Baot community. Each cohort of the program focuses on a specific topic in data science and is made up of about 4 to 8 teaching sessions. Each one of these sessions includes a theoretical lecture and practical work in groups.

The program began as a part of a collaboration between the "Baot" community, aimed at increasing the presence of women in the high-tech industry and to create a work environment in which they can share knowledge and practice in groups, and DataHack. Today there are also programs for men and women and the goal has expanded to share knowledge in the Israeli data science community in general, in addition to the original goal of promoting women in the industry. In this manner the programs are divided into 30% of mixed programs and roughly 70% programs for women only.
    """)

st.subheader('DataNights Youtube ▶️')
st.markdown("""<a href={}> access channel here</a>""".format("https://www.youtube.com/@DataHackIL"), unsafe_allow_html=True)



st.subheader('Medium Profile ✍️')
st.markdown("""<a href={}> access full profile here</a>""".format("https://galhever.medium.com/"), unsafe_allow_html=True)


try:
    page1, page2 = requests.get(info['Medium']), requests.get(info['publication_url'])

    followers = re.findall('(\d+\.\d+[kK]?) Followers', page1.text)[0]
    pub_followers = re.findall('Followers (?:\w+\s+){4}(\d+)', re.sub('\W+', ' ', page2.text))[0]

    cols = st.columns(2)
    cols[0].metric('Followers', followers)
    cols[1].metric('Publication followers', pub_followers)
except:
    pass


st.sidebar.caption('Wish to connect?')
st.sidebar.write('📧: galhever@gmail.com')

pdfFileObj = open('files/Gal Hever - Resume.pdf', 'rb')
st.sidebar.download_button('download resume', pdfFileObj, file_name='Gal Hever - Resume.pdf', mime='pdf')


# st.sidebar.write('LinkedIn | Github | Medium')
# st.sidebar.markdown("[![Foo](data:image/png;base64,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)](https://www.linkedin.com/in/gal-hever/)")
# st.sidebar.markdown("[![Foo](https://www.webfx.com/wp-content/uploads/2022/08/github-logo.png)](https://github.com/galhev)")
# st.sidebar.markdown("[![Foo](https://media.licdn.com/dms/image/D4E0BAQGLNpn-roUX0g/company-logo_200_200/0/1719258074326/datanights_logo?e=2147483647&v=beta&t=CAlWcS5k_x_tyJ1nZu9PMJFUypft-Q-CMq2Idcjevvs)](https://datanights-il.github.io/)")
# st.sidebar.markdown("[![Foo](https://media.licdn.com/dms/image/D4D0BAQFFNmFlKiblbg/company-logo_200_200/0/1688488446450/datahack_logo?e=2147483647&v=beta&t=krNwU4cf522tPtzbF-zxDc-7elz-VgxtZ8C7R4G6prY)](https://www.youtube.com/@DataHackIL)")
#
