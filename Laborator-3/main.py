import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import copy


def generareSerieDeTimp(dimensiune):
    SCALAR = 0.1
    MEDIE = 0.0
    DEVIATIE = 1.0

    z = np.random.normal(loc=MEDIE, scale=DEVIATIE, size=dimensiune)
    p = np.random.uniform(low=0.0, high=1.0)

    v = np.zeros(dimensiune)
    v[0] = np.random.uniform(low=-1.0, high=1.0) * SCALAR
    for t in range(1, dimensiune):
        prob = np.random.uniform(low=0.0, high=1.0)
        if prob < p:
            v[t] = v[t - 1]
        else:
            v[t] = np.random.uniform(low=-1.0, high=1.0) * SCALAR

    x = np.zeros(dimensiune)
    x[0] = np.random.uniform(low=-1.0, high=1.0) * SCALAR
    for t in range(1, dimensiune):
        x[t] = x[t - 1] + v[t - 1]

    y = x + z
    return x, y


x, y = generareSerieDeTimp(1000)


def desenareSerieDeTimp(x, y):
    figura, axe = plt.subplots(1, 2, figsize=(12, 5))

    axe[0].plot(x, color='red')
    axe[0].set_xlabel('Timp')
    axe[0].set_ylabel('Valoare Serie X')
    axe[0].set_title('Serie de Timp X')

    axe[1].plot(y, color='blue')
    axe[1].set_xlabel('Timp')
    axe[1].set_ylabel('Valoare Serie Y')
    axe[1].set_title('Serie de Timp Y')

    plt.tight_layout()
    plt.show()


desenareSerieDeTimp(x, y)


def solutieCVXPY(y, rho):
    y = copy.deepcopy(y)

    D = np.zeros((y.shape[0] - 2, y.shape[0]))
    for i in range(0, D.shape[0]):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    xCvxpy = cp.Variable(y.shape[0])
    yCvxpy = cp.Parameter(y.shape[0], value=y)
    rhoCvxpy = cp.Parameter(nonneg=True, value=rho)
    DCvxpy = cp.Parameter(D.shape, value=D)

    obiectiv = cp.Minimize(0.5 * cp.norm(xCvxpy - yCvxpy, 2) + rhoCvxpy * cp.norm(DCvxpy @ xCvxpy, 1))

    problema = cp.Problem(obiectiv)
    problema.solve()

    solutie = xCvxpy.value
    return solutie


# solutieCVX = solutieCVXPY(y, 0.3)


def desenareSolutie(y, solutie):
    figura, axe = plt.subplots(1, 2, figsize=(12, 5))

    axe[0].plot(y, color='red')
    axe[0].set_xlabel('Timp')
    axe[0].set_ylabel('Valoare Serie Y')
    axe[0].set_title('Serie de Timp Y')

    axe[1].plot(solutie, color='blue')
    axe[1].set_xlabel('Timp')
    axe[1].set_ylabel('Valoare Serie Solutie')
    axe[1].set_title('Serie de Timp Solutie')

    plt.tight_layout()
    plt.show()


# desenareSolutie(y, solutieCVX)


def calculDualitateLagrange(D, miu, y, valoareNegata):
    valoareDualitate = -0.5 * np.linalg.norm(D.T @ miu, 2) ** 2 + miu.T @ D @ y
    if valoareNegata:
        return -valoareDualitate
    else:
        return valoareDualitate


def calculGradientDualitateLagrange(D, miu, y, valoareNegata):
    gradient = -D @ D.T @ miu + D @ y
    if valoareNegata:
        return -gradient
    else:
        return gradient


def alegereAlphaAdaptiv(D, miu, y):
    EPSILON = 10**-10
    LIMITA_GENERARE_ALPHA = 5.0
    c = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    p = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    alpha = np.random.uniform(0.0 + EPSILON, LIMITA_GENERARE_ALPHA - EPSILON)

    c = 2.0 * EPSILON
    p = 0.01
    alpha = LIMITA_GENERARE_ALPHA - 2.0 * EPSILON

    while calculDualitateLagrange(D, miu - alpha * calculGradientDualitateLagrange(D, miu, y, True), y, True) > calculDualitateLagrange(D, miu, y, True) - c * alpha * np.linalg.norm(calculGradientDualitateLagrange(D, miu, y, True), 2) ** 2:
        alpha = p * alpha

    return alpha


def metodaGradientProiectat(y, rho, numarIteratii, pragGradient):
    NUMAR_ITERATII_PRINTARE = 10

    D = np.zeros((y.shape[0] - 2, y.shape[0]))
    for i in range(0, D.shape[0]):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    informatiiPlotare = []

    miu = np.zeros(y.shape[0] - 2)
    for iteratie in range(0, numarIteratii):
        gradient = calculGradientDualitateLagrange(D, miu, y, True)
        if np.linalg.norm(gradient, 2) < pragGradient:
            break
        alpha = alegereAlphaAdaptiv(D, miu, y)
        miu = miu - alpha * gradient
        miu = np.clip(miu, -rho, rho)

        if iteratie % NUMAR_ITERATII_PRINTARE == 0:
            print('Iteratia:', iteratie, 'Valoare Functie:', calculDualitateLagrange(D, miu, y, True))
        informatiiPlotare.append([iteratie, calculDualitateLagrange(D, miu, y, True)])

    plt.plot([informatiiPlotare[i][0] for i in range(len(informatiiPlotare))], [informatiiPlotare[i][1] for i in range(len(informatiiPlotare))], color='blue')
    plt.xlabel('Iteratii')
    plt.ylabel('Valoarea Functiei')
    plt.title('Exercitiul 1 - Metoda Gradient Proiectat')
    plt.show()

    x = y - D.T @ miu
    return x


# solutieMGP = metodaGradientProiectat(y, 1000.0, 100, 10**-3)
# desenareSolutie(y, solutieMGP)


def eliminareGaussianaPentadiagonala(A, y):
    y = copy.deepcopy(y)

    DIAGONALITATE = 5

    print('A:', A)

    x = np.zeros(y.shape[0])
    for iteratie in range(DIAGONALITATE // 2):
        for linie in range(DIAGONALITATE // 2 - iteratie, A.shape[0]):
            stanga = linie - (DIAGONALITATE // 2 - iteratie)
            dreapta = min(stanga + DIAGONALITATE - 1, A.shape[1] - 1)

            factor = A[linie, stanga] / A[linie - 1, stanga]
            for coloana in range(stanga, dreapta + 1):
                A[linie, coloana] -= factor * A[linie - 1, coloana]
            y[linie] -= factor * y[linie - 1]


    for linie in range(A.shape[0] - 1, -1, -1):
        stanga = linie
        dreapta = min(stanga + DIAGONALITATE - 1, A.shape[1] - 1)

        solutieCurenta = y[linie]
        for coloana in range(dreapta, stanga, -1):
            solutieCurenta -= A[linie, coloana] * x[coloana]
        x[stanga] = solutieCurenta / A[linie, stanga]

    return x


def metodaTridiagonala(y, rho):
    D = np.zeros((y.shape[0] - 2, y.shape[0]))
    for i in range(0, D.shape[0]):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    A = np.eye(y.shape[0]) + 2.0 * rho * D.T @ D

    return eliminareGaussianaPentadiagonala(A, y)


solutieTridiagonala = metodaTridiagonala(y, 100.0)
desenareSolutie(y, solutieTridiagonala)


def comparareSolutii(solutie0, solutie1):
    figura, axe = plt.subplots(1, 2, figsize=(12, 5))

    axe[0].plot(solutie0, color='red')
    axe[0].set_xlabel('Timp')
    axe[0].set_ylabel('Valoare Serie 1')
    axe[0].set_title('Serie de Timp 1')

    axe[1].plot(solutie1, color='blue')
    axe[1].set_xlabel('Timp')
    axe[1].set_ylabel('Valoare Serie 2')
    axe[1].set_title('Serie de Timp 2')

    plt.tight_layout()
    plt.show()


# comparareSolutii(solutieMGP, solutieTridiagonala)



