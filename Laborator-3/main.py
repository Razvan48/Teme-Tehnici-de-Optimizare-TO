import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


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


x, y = generareSerieDeTimp(10000)


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


solutie = solutieCVXPY(y, 100.0)


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


desenareSolutie(y, solutie)


