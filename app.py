import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.header("objectif de l'application")
st.latex(r"P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k")
#Quand on déplace le slider le script est exécuté plusieurs fois.

