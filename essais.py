# À LA PLACE de l'habituel 'import numpy as np'
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
#Q1
X=np.linspace(0,2*np.pi,100)
#Q2
sin_der=(np.vectorize(grad(np.sin)))
#Q3
sin_der(X)==np.cos(X)

from math import factorial

#Version 1 dérivées fournies.

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

sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)
plt.plot(taylor1(X,sinus10))
plt.plot(np.sin(X))
plt.show()
#Version 2 dérivées non fournies

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

# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)
#Q1
plt.plot(Y)
plt.plot(Y2)
plt.show()

#Q2
diff=np.abs(Y-Y2)
#Calcul de l'écart entre Y et Y2
I_correct=X[diff<=0.001]
#On renvoie l'intervalle des X tels que |Y-Y2|<=10^(-3)

x=I_correct[-1]-I_correct[0] #On obtient l'intervalle approché : [-2.881761682240261 2.8817616822402616]
print(f"On a {100*(x/(4*np.pi))}% du domaine qui est proche de l'approximation à 10^-3 près")
#Q3
plt.plot(taylor2(X,np.cos,0))
plt.plot(np.cos(X))
plt.show()
#On obtient une droite horizontale y=1.

def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2*np.sin(X)+np.cos(X/4)
Y3 = custom(X)
Y4 = taylor2(X, custom, 14)
plt.plot(Y3)
plt.plot(Y4)
plt.show()
#Même méthode que précédement.
diff2=np.abs(Y3-Y4)
I_correct2=X[diff2<=0.001]
x2=I_correct2[-1]-I_correct2[0]
print(f"On a {100*(x2/(4*np.pi))}% du domaine qui est proche de l'approximation à 10^-3 près")
