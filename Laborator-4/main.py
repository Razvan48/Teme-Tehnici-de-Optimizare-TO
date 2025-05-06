import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def genereazaPuncteSeparabile(m, n):
    VALOARE_MAXIMA = 10.0
    w = (np.random.rand(n) * 2 - 1) * VALOARE_MAXIMA
    w = w / np.linalg.norm(w)
    b = (np.random.rand(1) * 2 - 1) * VALOARE_MAXIMA

    x = []
    etichete = []

    for _ in range(m):
        x_nou = (np.random.rand(n) * 2 - 1) * VALOARE_MAXIMA
        if np.dot(w, x_nou) + b == 0.0:
            continue
        elif np.dot(w, x_nou) + b > 0:
            etichete.append(1)
        else:
            etichete.append(-1)
        x.append(x_nou)

    return np.array(x), np.array(etichete), w, b


def afiseazaPuncte(x, etichete, w, b):
    plt.figure()
    plt.title('Punctele Generate Aleatoriu')
    plt.xlabel('x1')
    plt.ylabel('x2')

    for i in range(len(x)):
        if etichete[i] == 1:
            plt.scatter(x[i][0], x[i][1], color='blue')
        else:
            plt.scatter(x[i][0], x[i][1], color='red')

    xMinim = np.min(x[:, 0])
    xMaxim = np.max(x[:, 0])
    if w[1] != 0.0:
        yLinie0 = (-b - w[0] * xMinim) / w[1]
        yLinie1 = (-b - w[0] * xMaxim) / w[1]
    else:
        EPSILON = 1e-10
        yLinie0 = (-b - w[0] * xMinim) / EPSILON
        yLinie1 = (-b - w[0] * xMaxim) / EPSILON
    plt.plot([xMinim, xMaxim], [yLinie0, yLinie1], color='green')

    plt.show()


def comparaDrepteleSeparatoare(x, etichete, w, b, wOptim, bOptim, vectoriiSuport):
    plt.figure()
    plt.title('Dreapta Separatoare SVM Separabil CVXPY')
    plt.xlabel('x1')
    plt.ylabel('x2')

    for i in range(len(x)):
        if etichete[i] == 1:
            if vectoriiSuport[i]:
                plt.scatter(x[i][0], x[i][1], color='blue', marker='*', s=50)
            else:
                plt.scatter(x[i][0], x[i][1], color='blue')
        else:
            if vectoriiSuport[i]:
                plt.scatter(x[i][0], x[i][1], color='red', marker='*', s=50)
            else:
                plt.scatter(x[i][0], x[i][1], color='red')

    xMinim = np.min(x[:, 0])
    xMaxim = np.max(x[:, 0])

    EPSILON = 1e-10

    if w[1] != 0.0:
        yLinie0 = (-b - w[0] * xMinim) / w[1]
        yLinie1 = (-b - w[0] * xMaxim) / w[1]
    else:
        yLinie0 = (-b - w[0] * xMinim) / EPSILON
        yLinie1 = (-b - w[0] * xMaxim) / EPSILON
    plt.plot([xMinim, xMaxim], [yLinie0, yLinie1], color='green')

    if wOptim[1] != 0.0:
        yLinieOptim0 = (-bOptim - wOptim[0] * xMinim) / wOptim[1]
        yLinieOptim1 = (-bOptim - wOptim[0] * xMaxim) / wOptim[1]
    else:
        yLinieOptim0 = (-bOptim - wOptim[0] * xMinim) / EPSILON
        yLinieOptim1 = (-bOptim - wOptim[0] * xMaxim) / EPSILON
    plt.plot([xMinim, xMaxim], [yLinieOptim0, yLinieOptim1], color='orange')

    plt.show()


def solutieEx1(x, etichete, w, b):
    m = x.shape[0]
    n = x.shape[1]

    wVariabila = cp.Variable(n)
    bVariabila = cp.Variable(1)

    constrangeri = [etichete[i] * (cp.matmul(wVariabila, x[i]) + bVariabila) >= 1 for i in range(m)]
    obiectiv = cp.Minimize(cp.norm(wVariabila, 2))

    problema = cp.Problem(obiectiv, constrangeri)
    problema.solve()

    wOptim = wVariabila.value
    bOptim = bVariabila.value

    print('Exercitiul 1, w optim:', wOptim)
    print('Exercitiul 1, b optim:', bOptim)

    esteSolutieCorecta = True
    for i in range(m):
        if etichete[i] * (np.dot(wOptim, x[i]) + bOptim) < 1:
            esteSolutieCorecta = False
            break

    print('Verificarea Solutiei:', problema.status)
    print('Este Solutie Corecta:', 'DA' if esteSolutieCorecta else 'NU')

    EPSILON = 1e-10
    vectoriiSuport = [False for _ in range(m)]
    for i in range(m):
        if etichete[i] * (np.dot(wOptim, x[i]) + bOptim) - 1 < EPSILON:
            vectoriiSuport[i] = True

    comparaDrepteleSeparatoare(x, etichete, w, b, wOptim, bOptim, vectoriiSuport)


def ex1():
    np.random.seed(0)
    m = 100
    n = 2
    x, etichete, w, b = genereazaPuncteSeparabile(m, n)
    afiseazaPuncte(x, etichete, w, b)
    solutieEx1(x, etichete, w, b)


ex1()


def adaugareZgomot(etichete, probabilitateZgomot):
    for i in range(len(etichete)):
        if np.random.rand() < probabilitateZgomot:
            etichete[i] = -etichete[i]
    return etichete


def solutieEx2(x, etichete, w, b):
    pass


def ex2():
    np.random.seed(0)
    m = 100
    n = 2
    x, etichete, w, b = genereazaPuncteSeparabile(m, n)
    etichete = adaugareZgomot(etichete, 0.05)
    afiseazaPuncte(x, etichete, w, b)
    solutieEx2(x, etichete, w, b)


ex2()



