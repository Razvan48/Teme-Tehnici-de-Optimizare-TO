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


def functieObiectiv(lambdaCurent, QTQ, E, MIU, etichete):
    return 0.5 * lambdaCurent.T @ QTQ @ lambdaCurent - E.T @ lambdaCurent + MIU * etichete.T @ lambdaCurent


def afiseazaEvolutieFunctie(valoriFunctie):
    plt.figure()
    plt.title('Evolutia Functiei Obiectiv')
    plt.xlabel('Iteratia')
    plt.ylabel('Valoare Functie')

    plt.plot(valoriFunctie, color='red')

    plt.show()


def coborarePeGradientDualEx2(x, etichete, w, b, RHO, MIU):
    etichete = etichete.reshape((x.shape[0], 1))

    EPSILON_CRITERIU_OPRIRE = 1e-2
    LAMBDA_0 = np.full((x.shape[0], 1), 1.0)
    RATA_DE_INVATARE = 0.001
    NUMAR_ITERATII = 10000

    Q = np.array([(etichete[i] * x[i]).tolist() for i in range(x.shape[0])])
    QTQ = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            QTQ[i][j] = np.dot(Q[i], Q[j])
    E = np.ones((x.shape[0], 1))

    valoriFunctie = []

    lambdaCurent = LAMBDA_0
    valoareAnterioaraFunctie = np.inf
    valoareCurentaFunctie = functieObiectiv(lambdaCurent, QTQ, E, MIU, etichete)[0][0]
    valoriFunctie.append(valoareCurentaFunctie)

    iteratieCurenta = 0
    while np.abs(valoareCurentaFunctie - valoareAnterioaraFunctie) > EPSILON_CRITERIU_OPRIRE and iteratieCurenta < NUMAR_ITERATII:
        valoareAnterioaraFunctie = valoareCurentaFunctie
        gradient = QTQ @ lambdaCurent - E + MIU * etichete
        lambdaCurent = lambdaCurent - RATA_DE_INVATARE * gradient

        lambdaCurent = np.minimum(np.maximum(0.0, lambdaCurent), RHO)
        valoareCurentaFunctie = functieObiectiv(lambdaCurent, QTQ, E, MIU, etichete)[0][0]
        valoriFunctie.append(valoareCurentaFunctie)

        print('Iteratia:', len(valoriFunctie) - 1, 'Valoare Functie:', valoareCurentaFunctie)

        iteratieCurenta += 1

    wOptim = np.zeros((x.shape[1], 1))
    for i in range(x.shape[0]):
        wOptim += lambdaCurent[i] * etichete[i] * x[i].reshape((x.shape[1], 1))
    bOptim = np.mean(etichete - np.dot(x, wOptim))

    EPSILON = 1e-10
    vectoriiSuport = [False for _ in range(x.shape[0])]
    '''
    for i in range(x.shape[0]):
        if etichete[i][0] * (np.dot(wOptim.reshape(wOptim.shape[0]), x[i]) + bOptim) - 1 < EPSILON:
            vectoriiSuport[i] = True
    '''

    afiseazaEvolutieFunctie(valoriFunctie)
    comparaDrepteleSeparatoare(x, etichete, w, b, wOptim, bOptim, vectoriiSuport)


def ex2():
    np.random.seed(0)
    m = 100
    n = 2
    x, etichete, w, b = genereazaPuncteSeparabile(m, n)
    etichete = adaugareZgomot(etichete, 0.05)
    afiseazaPuncte(x, etichete, w, b)
    coborarePeGradientDualEx2(x, etichete, w, b, 10000.0, 10000.0)


ex2()



