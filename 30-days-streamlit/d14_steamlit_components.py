import streamlit as st
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


"""
https://streamlit.io/components
Components are third-party modules that extend what’s possible with Streamlit. They couldn't be simpler to use, just a pip-install away.
https://discuss.streamlit.io/t/streamlit-components-community-tracker/4634
Streamlit Components - Community Tracker
"""
st.header('`streamlit_pandas_profiling`')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

pr = df.profile_report()
st_profile_report(pr)