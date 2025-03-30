import numpy as np
import matplotlib.pyplot as plt
import copy


# Exercitiul 1


# k (numarul de conditionare) = raportul dintre cea mai mare valoare proprie si cea mai mica valoare proprie
# Construim matricea A ca fiind o matrice diagonala (valorile proprii vor fi pe diagonala in acest caz,
# deoarece determinantul unei matrici diagonale este produsul elementelor de pe diagonala).
# Alegem valorile proprii astfel incat k > 10^6.
# Aceasta matrice diagonala este inversabila, deoarece toate elementele de pe diagonala sunt nenule.
# Solutia problemei este x = A^(-1) * b si este unica.
def a1():
    EPSILON = 10**-5
    VAL_PESTE_K = 10**1
    VAL_MIN = 10**3 + EPSILON
    VAL_PESTE_MIN = 10**1
    VALOARE_PROPRIE_MIN = 0.000001

    n = 2
    k = 10**6 + EPSILON

    A = np.diag([VALOARE_PROPRIE_MIN, VALOARE_PROPRIE_MIN * k] + [VALOARE_PROPRIE_MIN] * (n - 2))
    b = np.array([VAL_MIN * VALOARE_PROPRIE_MIN / np.sqrt(n), VAL_MIN * VALOARE_PROPRIE_MIN * k / np.sqrt(n)] + [VAL_MIN * VALOARE_PROPRIE_MIN / np.sqrt(n)] * (n - 2)).reshape(n, 1)
    print('b:', b)

    valoriProprii = np.linalg.eigvals(A)
    print('Valori proprii:', valoriProprii)
    print('k:', max(valoriProprii) / min(valoriProprii))

    x = np.linalg.inv(A) @ b
    print('Solutia:', x)
    print('V*:', np.linalg.norm(x))

    return A, b, x


A, b, x = a1()


def alegereAlpha0(L):
    EPSILON = 10**-10
    return np.random.uniform(0.0 + EPSILON, 2.0 / L - EPSILON)


def functie(A, b, x):
    return 0.5 * np.linalg.norm(A @ x - b)**2


def alegereAlpha1(A, b, x, gradient):
    STANGA = 0.0
    DREAPTA = 1.0
    DIMENSIUNE_PAS = 0.01

    minimFunctie = np.inf
    alphaSolutie = None
    for alpha in np.arange(STANGA, DREAPTA, DIMENSIUNE_PAS):
        if functie(A, b, x - alpha * gradient) < minimFunctie:
            minimFunctie = functie(A, b, x - alpha * gradient)
            alphaSolutie = alpha

    return alphaSolutie


def alegereAlpha2(A, b, x, gradient):
    EPSILON = 10**-10
    LIMITA_GENERARE_ALPHA = 1.0
    c = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    p = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    alpha = np.random.uniform(0.0 + EPSILON, LIMITA_GENERARE_ALPHA - EPSILON)

    while functie(A, b, x - alpha * gradient) > functie(A, b, x) - c * alpha * (np.linalg.norm(gradient)**2):
        alpha = p * alpha

    return alpha


def b1(A, b, pragGradient, numarIteratii):
    NUMAR_ITERATII_PRINTARE = 10
    x = np.array([500.0] * A.shape[1]).reshape(A.shape[1], 1)
    L = max(np.linalg.eigvals(A))
    informatii = []
    for iteratieCurenta in range(numarIteratii):
        gradient = A.T @ (A @ x - b)

        # alpha = alegereAlpha0(L)
        # alpha = alegereAlpha1(A, b, x, gradient)
        alpha = alegereAlpha2(A, b, x, gradient)

        urmatorulX = x - alpha * gradient
        if np.linalg.norm(gradient) < pragGradient:
            break

        if iteratieCurenta % NUMAR_ITERATII_PRINTARE == 0:
            print('Iteratia:', iteratieCurenta, 'x:', x, 'Functie:', functie(A, b, x))
        informatii.append((x, functie(A, b, x)))

        x = urmatorulX

    plt.plot([i for i in range(len(informatii))], [informatie[1] for informatie in informatii], color='blue')
    plt.xlabel('Iteratii')
    plt.ylabel('Functie Obiectiv')
    plt.title('Exercitiul 1 - Punctul b)')
    plt.show()

    return x


b1(A, b, 10**-13, 15)

def calculareGradientStohastic0(A, b, dimensiuneBatch):
    SCALAR = 1000.0
    gradientStohastic = np.zeros((A.shape[1], 1))
    for valoareCurenta in range(dimensiuneBatch):
        xAleator = np.random.uniform(size=(A.shape[1], 1)) * SCALAR
        gradientStohastic += A.T @ (A @ xAleator - b)
    return gradientStohastic / dimensiuneBatch


def calculareGradientStohastic1(A, b, x, dimensiuneBatch):
    gradientReal = A.T @ (A @ x - b)

    if dimensiuneBatch > gradientReal.shape[0]:
        return gradientReal

    indexiLinii = np.random.choice(gradientReal.shape[0], size=dimensiuneBatch, replace=False)

    gradientStohastic = np.zeros(gradientReal.shape)
    gradientStohastic[indexiLinii] = gradientReal[indexiLinii]

    return gradientStohastic


def c1(A, b, pragGradient, numarIteratii, dimensiuneBatch):
    NUMAR_ITERATII_PRINTARE = 10
    x = np.array([500.0] * A.shape[1]).reshape(A.shape[1], 1)
    L = max(np.linalg.eigvals(A))
    informatii = []
    for iteratieCurenta in range(numarIteratii):
        # gradientStohastic = calculareGradientStohastic0(A, b, dimensiuneBatch)
        gradientStohastic = calculareGradientStohastic1(A, b, x, dimensiuneBatch)

        # alpha = alegereAlpha0(L)
        # alpha = alegereAlpha1(A, b, x, gradientStohastic)
        alpha = alegereAlpha2(A, b, x, gradientStohastic)

        urmatorulX = x - alpha * gradientStohastic
        if np.linalg.norm(gradientStohastic) < pragGradient:
            break

        if iteratieCurenta % NUMAR_ITERATII_PRINTARE == 0:
            print('Iteratia:', iteratieCurenta, 'x:', x, 'Functie:', functie(A, b, x))
        informatii.append((x, functie(A, b, x)))

        x = urmatorulX

    plt.plot([i for i in range(len(informatii))], [informatie[1] for informatie in informatii], color='blue')
    plt.xlabel('Iteratii')
    plt.ylabel('Functie Obiectiv')
    plt.title('Exercitiul 1 - Punctul c)')
    plt.show()

    return x


c1(A, b, 10**-13, 15, 1)


def generarePseudoGradient(A, b, x):
    STANGA = 0.01
    DREAPTA = 0.1

    gradientReal = A.T @ (A @ x - b)

    perturbare = np.random.uniform(size=gradientReal.shape)

    normaInitialaPerturbare = np.linalg.norm(perturbare)
    normaDoritaPerturbare = np.random.uniform(STANGA, DREAPTA)
    pseudoGradient = gradientReal + perturbare * normaDoritaPerturbare / normaInitialaPerturbare

    return pseudoGradient


def d1(A, b, pragGradient, numarIteratii):
    NUMAR_ITERATII_PRINTARE = 10
    x = np.array([500.0] * A.shape[1]).reshape(A.shape[1], 1)
    L = max(np.linalg.eigvals(A))
    informatii = []
    for iteratieCurenta in range(numarIteratii):
        pseudoGradient = generarePseudoGradient(A, b, x)

        # alpha = alegereAlpha0(L)
        # alpha = alegereAlpha1(A, b, x, pseudoGradient)
        alpha = alegereAlpha2(A, b, x, pseudoGradient)

        urmatorulX = x - alpha * pseudoGradient
        if np.linalg.norm(pseudoGradient) < pragGradient:
            break

        if iteratieCurenta % NUMAR_ITERATII_PRINTARE == 0:
            print('Iteratia:', iteratieCurenta, 'x:', x, 'Functie:', functie(A, b, x))
        informatii.append((x, functie(A, b, x)))

        x = urmatorulX

    plt.plot([i for i in range(len(informatii))], [informatie[1] for informatie in informatii], color='blue')
    plt.xlabel('Iteratii')
    plt.ylabel('Functie Obiectiv')
    plt.title('Exercitiul 1 - Punctul d)')
    plt.show()

    return x


d1(A, b, 10**-13, 15)


# Exercitiul 2


def calculH(x, W, p):
    suma = 0.0
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            suma += W[i, j] * (np.linalg.norm(x[i] - x[j]) ** p)
    return suma


def gradientH(x, W, p, esteSediu):
    EPSILON = 10**-10
    gradient = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if not esteSediu[i]:
            for j in range(x.shape[0]):
                if i != j:
                    if i < j:
                        for k in range(x.shape[1]):
                            gradient[i, k] += W[i, j] * p * (np.linalg.norm(x[i] - x[j]) ** (p - 1)) / (2.0 * np.linalg.norm(x[i] - x[j]) + EPSILON) * 2.0 * (x[i, k] - x[j, k])
                    else:
                        for k in range(x.shape[1]):
                            gradient[i, k] -= W[i, j] * p * (np.linalg.norm(x[j] - x[i]) ** (p - 1)) / (2.0 * np.linalg.norm(x[j] - x[i]) + EPSILON) * 2.0 * (x[i, k] - x[j, k])
    return gradient


def alegereAlpha3(x, W, p, esteSediu):
    STANGA = 0.0
    DREAPTA = 1.0
    DIMENSIUNE_PAS = 0.01

    minimFunctie = np.inf
    alphaSolutie = None
    for alpha in np.arange(STANGA, DREAPTA, DIMENSIUNE_PAS):
        valoareFunctie = calculH(x - alpha * gradientH(x, W, p, esteSediu), W, p)
        if valoareFunctie < minimFunctie:
            minimFunctie = valoareFunctie
            alphaSolutie = alpha

    return alphaSolutie


def alegereAlpha4(x, W, p, esteSediu):
    EPSILON = 10**-10
    LIMITA_GENERARE_ALPHA = 1.0
    c = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    pIntern = np.random.uniform(0.0 + EPSILON, 1.0 - EPSILON)
    alpha = np.random.uniform(0.0 + EPSILON, LIMITA_GENERARE_ALPHA - EPSILON)

    while calculH(x - alpha * gradientH(x, W, p, esteSediu), W, p) > calculH(x, W, p) - c * alpha * (np.linalg.norm(gradientH(x, W, p, esteSediu))**2):
        alpha = pIntern * alpha

    return alpha


def coborarePeGradient(x, W, p, esteSediu, pragGradient, numarIteratii):
    NUMAR_ITERATII_PRINTARE = 10
    informatii = []
    for iteratieCurenta in range(numarIteratii):
        gradient = gradientH(x, W, p, esteSediu)

        # alpha = alegereAlpha3(x, W, p, esteSediu)
        alpha = alegereAlpha4(x, W, p, esteSediu)

        urmatorulX = x - alpha * gradient
        if np.linalg.norm(gradient) < pragGradient:
            break

        valoareFunctie = calculH(x, W, p)
        if iteratieCurenta % NUMAR_ITERATII_PRINTARE == 0:
            print('Iteratia:', iteratieCurenta, 'x:', x, 'Functie:', valoareFunctie)
        informatii.append((x, valoareFunctie))

        x = urmatorulX

    plt.plot([i for i in range(len(informatii))], [informatie[1] for informatie in informatii], color='blue')
    plt.xlabel('Iteratii')
    plt.ylabel('Functie Obiectiv')
    plt.title('Exercitiul 2 - Coborare pe Gradient')
    plt.show()

    return x


def calculareGradientStohasticH0(x, W, p, esteSediu, dimensiuneBatch):
    SCALAR = 1000.0
    gradientStohastic = np.zeros(x.shape)
    for valoareCurenta in range(dimensiuneBatch):
        xAleator = np.random.uniform(size=x.shape) * SCALAR
        gradientStohastic += gradientH(xAleator, W, p, esteSediu)
    return gradientStohastic / dimensiuneBatch


def calculareGradientStohasticH1(x, W, p, esteSediu, dimensiuneBatch):
    gradientReal = gradientH(x, W, p, esteSediu)

    if dimensiuneBatch > gradientReal.shape[0]:
        return gradientReal

    indexiLinii = np.random.choice(gradientReal.shape[0], size=dimensiuneBatch, replace=False)

    gradientStohastic = np.zeros(gradientReal.shape)
    gradientStohastic[indexiLinii] = gradientReal[indexiLinii]

    return gradientStohastic


def coborarePeGradientStohastic(x, W, p, esteSediu, pragGradient, numarIteratii, dimensiuneBatch):
    NUMAR_ITERATII_PRINTARE = 10
    informatii = []
    for iteratieCurenta in range(numarIteratii):
        # gradientStohastic = calculareGradientStohasticH0(x, W, p, esteSediu, dimensiuneBatch)
        gradientStohastic = calculareGradientStohasticH1(x, W, p, esteSediu, dimensiuneBatch)

        # alpha = alegereAlpha3(x, W, p, esteSediu)
        alpha = alegereAlpha4(x, W, p, esteSediu)

        urmatorulX = x - alpha * gradientStohastic
        #if np.linalg.norm(gradientStohastic) < pragGradient:
        #    break

        valoareFunctie = calculH(x, W, p)
        if iteratieCurenta % NUMAR_ITERATII_PRINTARE == 0:
            print('Iteratia:', iteratieCurenta, 'x:', x, 'Functie:', valoareFunctie)
        informatii.append((x, valoareFunctie))

        x = urmatorulX

    plt.plot([i for i in range(len(informatii))], [informatie[1] for informatie in informatii], color='blue')
    plt.xlabel('Iteratii')
    plt.ylabel('Functie Obiectiv')
    plt.title('Exercitiul 2 - Coborare pe Gradient Stohastic')
    plt.show()

    return x


def desenarePuncte(xInitial, xFinal, esteSediu):
    for i in range(xInitial.shape[0]):
        if esteSediu[i]:
            plt.scatter(xInitial[i, 0], xInitial[i, 1], color='black')
        else:
            plt.scatter(xInitial[i, 0], xInitial[i, 1], color='red')
    for i in range(xFinal.shape[0]):
        if not esteSediu[i]:
            plt.scatter(xFinal[i, 0], xFinal[i, 1], color='green')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Exercitiul 2 - Desenare Puncte')
    plt.show()


def exercitiul2():
    x = np.array([
        [-25.0, -25.0],
        [-25.0, 25.0],
        [25.0, -25.0],
        [25.0, 25.0],

        [1.0, 1.0],
        [5.0, 1.0],
        [9.0, 1.0],
        [1.0, 5.0],
        [1.0, 9.0],
        [5.0, 9.0],
        [9.0, 5.0],
        [9.0, 9.0],
    ])
    W = np.ones((x.shape[0], x.shape[0]))
    p = 2
    esteSediu = np.array([False, False, False, False, True, True, True, True, True, True, True, True])

    xInitial = copy.deepcopy(x)
    xFinal = coborarePeGradient(x, W, p, esteSediu, 10**-13, 7)
    # xFinal = coborarePeGradientStohastic(x, W, p, esteSediu, 10**-13, 20, 1)

    # Cu rosu desenam pozitiile initiale ale punctelor mobile, cu negru pozitiile initiale ale punctelor fixe, iar cu verde pozitiile finale ale punctelor mobile.
    desenarePuncte(xInitial, xFinal, esteSediu)

    print('xInitial:', xInitial)
    print('xFinal:', xFinal)


exercitiul2()




