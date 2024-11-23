import streamlit as st


# --- PAGE SETUP ---
home_page = st.Page(
    "views/home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)

crawling_page = st.Page(
    "views/crawling.py",
    title="Crawling",
    icon=":material/dataset:",
)

preprocessing_page = st.Page(
    "views/preprocessing.py",
    title="Preprocessing",
    icon=":material/bar_chart:",
)

classification_page = st.Page(
    "views/classification.py",
    title="Training",
    icon=":material/category:",
)

predict_page = st.Page(
    "views/predict.py",
    title="Classify",
    icon=":material/lightbulb:",
)


pg = st.navigation(pages=[home_page, crawling_page, preprocessing_page,classification_page,predict_page])


# --- SHARED ON ALL PAGES ---
# st.logo("assets/codingisfun_logo.png")
st.sidebar.markdown("by fraska")


# --- RUN NAVIGATION ---
pg.run()