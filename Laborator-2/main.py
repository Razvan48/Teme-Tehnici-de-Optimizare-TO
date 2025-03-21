import numpy as np



# k = raportul dintre cea mai mare valoare proprie si cea mai mica valoare proprie
# Construim matricea A ca fiind o matrice diagonala (valorile proprii vor fi pe diagonala in acest caz,
# deoarece determinantul unei matrici diagonale este produsul elementelor de pe diagonala), cu valori proprii astfel incat k > 10^6.
# Pentru vectorul B
def a1():
    n = 10
    k = 10**6
    VAL_PESTE_K = 10**2
    A = np.diag([1, k + np.random.randint(1, VAL_PESTE_K)] + [1] * (n - 2))
    complementOrtogonal, _ = np.linalg.qr(A)
    B = np.random.rand(n)
    B = B - complementOrtogonal @ (complementOrtogonal.T @ B)

    DISTANTA = 10**3
    VAL_PESTE_DISTANTA = 10**2

    norma_B = np.linalg.norm(B)



# a1()



def alegereAlpha0(L):
    EPSILON = 10**-10
    return np.random.uniform(0.0 + EPSILON, 2.0 / L - EPSILON)


def functie(A, B, X):
    return 0.5 * np.linalg.norm(A @ X - B)**2


def alegereAlpha1(A, B, X, gradient):
    STANGA = 0.0
    DREAPTA = 1.0
    DIMENSIUNE_PAS = 0.01

    minimFunctie = np.inf
    alphaSolutie = None
    for alpha in range(STANGA, DREAPTA, DIMENSIUNE_PAS):
        if functie(A, B, X - alpha * gradient) < minimFunctie:
            minimFunctie = functie(A, B, X - alpha * gradient)
            alphaSolutie = alpha

    return alphaSolutie


def alegereAlpha2(A, B, X, gradient):
    EPSILON = 10**-10
    c = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    p = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    LIMITA_GENERARE_ALPHA = 1.0
    alpha = np.random.uniform(0.0 + EPSILON, LIMITA_GENERARE_ALPHA - EPSILON)

    while functie(A, B, X - alpha * gradient) > functie(A, B, X) - c * alpha * np.linalg.norm(gradient)**2:
        alpha = p * alpha

    return alpha



def b1(A, B, pragGradient, numarIteratii, L):
    X = np.random.rand(A.shape[1])
    for iteratieCurenta in range(numarIteratii):
        gradient = A.T @ (A @ X - B)

        alpha = alegereAlpha0(L)
        # alpha = alegereAlpha1(A, B, X, gradient)
        # alpha = alegereAlpha2(A, B, X, gradient)

        urmatorulX = X - alpha * gradient
        if np.linalg.norm(gradient) < pragGradient:
            break
        X = urmatorulX
    return X



def c1(A, B, pragGradient, numarIteratii, L, dimensiuneBatch):
    X = np.random.rand(A.shape[1])
    for iteratieCurenta in range(numarIteratii):
        liniiAlese = np.random.choice(A.shape[0], dimensiuneBatch, replace=False)

        gradient = A[liniiAlese, :].T @ (A[liniiAlese, :] @ X - B[liniiAlese])

        alpha = alegereAlpha0(L)
        # alpha = alegereAlpha1(A, B, X, gradient)
        # alpha = alegereAlpha2(A, B, X, gradient)

        urmatorulX = X - alpha * gradient
        if np.linalg.norm(gradient) < pragGradient:
            break
        X = urmatorulX
    return X



def generareGradient(A, B, X):
    STANGA = 0.01
    DREAPTA = 0.1

    gradientReal = A.T @ (A @ X - B)
    pozitieInGradient = np.random.randint(0, gradientReal.shape[0])

    gradient = gradientReal.copy()
    gradient[pozitieInGradient] += np.random.choice([-1.0, 1.0]) * np.random.uniform(STANGA, DREAPTA)

    return gradient


def d1(A, B, pragGradient, numarIteratii, L):
    X = np.random.rand(A.shape[1])
    for iteratieCurenta in range(numarIteratii):

        gradient = generareGradient(A, B, X)

        alpha = alegereAlpha0(L)
        # alpha = alegereAlpha1(A, B, X, gradient)
        # alpha = alegereAlpha2(A, B, X, gradient)

        urmatorulX = X - alpha * gradient
        if np.linalg.norm(gradient) < pragGradient:
            break
        X = urmatorulX
    return X


