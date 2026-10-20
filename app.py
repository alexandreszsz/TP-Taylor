import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.header("objectif de l'application")
st.latex(r"P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k")
#Quand on déplace le slider le script est exécuté plusieurs fois.

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """

    s=np.zeros(len(X))
    for i in range(len(derivatives)):
        s+=X**i *derivatives[i]/factorial(i)
    #Calcul de la somme approchée de taylor en x.
    return s

def taylor2(X, f, n):
    """
    X : le domaine
    f : la fonction (sur scalaire)
    n : le degré

    Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
    """
    deriv=[]
    func=f
    for i in range(n+1):
        deriv.append(func(0.0))
        func=grad(func)
    return taylor1(X,deriv)

def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2*np.sin(X)+np.cos(X/4)

FONCTIONS = { "sin" : np.sin,
             "cos" : np.cos,
             "exp" : np.exp,
             "ln" : lambda x : np.log(1+x),
             "custom" : custom}

with st.sidebar :
    f = st.selectbox("choisissez une fonction",FONCTIONS)
    ordre = st.slider("Choisissez un ordre",0,10)
    largeur = st.slider("Choisissez la largeur de l'intervalle",0, 10)
X=np.linspace(-largeur/2,largeur/2,400)
Y=FONCTIONS[f](X)
Y_taylor=taylor2(X,FONCTIONS[f],ordre)

fig, ax = plt.subplots()
ax.plot(Y)
ax.plot(Y_taylor)
st.pyplot(fig)

def taux_coincidence(Y, Y_taylor, seuil):
    """Renvoie la proportion de l'intervalle où |Y-Y_taylor| est inférieur au seuil"""
    