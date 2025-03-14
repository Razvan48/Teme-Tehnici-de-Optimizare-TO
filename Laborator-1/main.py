from scipy.optimize import minimize
from scipy.optimize import linprog
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pyomo
import cv2 as cv



# Problema 1



def creareImagineTonuriGri():
    imagine = cv.imread('imagineCorupta.png')
    imagine = cv.cvtColor(imagine, cv.COLOR_BGR2GRAY)
    cv.imwrite('imagineCoruptaTonuriGri.png', imagine)



def problema1A():
    imagineOriginala = cv.imread('imagineOriginalaTonuriGri.png')[:,:,0]
    imagineCorupta = cv.imread('imagineCoruptaTonuriGri.png')[:,:,0]

    pixeliFixati = np.zeros(imagineOriginala.shape)
    for i in range(imagineOriginala.shape[0]):
        for j in range(imagineOriginala.shape[1]):
            if imagineOriginala[i, j] == imagineCorupta[i, j]:
                pixeliFixati[i, j] = 1

    figura, axe = plt.subplots(1, 3, figsize=(10, 5))
    axe[0].imshow(imagineOriginala, cmap='gray')
    axe[0].set_title('Imagine Originala')
    axe[0].axis('off')
    axe[1].imshow(imagineCorupta, cmap='gray')
    axe[1].set_title('Imagine Corupta')
    axe[1].axis('off')

    U = cp.Variable(shape=imagineOriginala.shape)
    obiectiv = cp.Minimize(cp.tv(U))
    constrangeri = [cp.multiply(pixeliFixati, U) == cp.multiply(pixeliFixati, imagineCorupta)]
    problema = cp.Problem(obiectiv, constrangeri)

    problema.solve(verbose=True, solver=cp.SCS)

    print('Valoarea optima a obiectivului: ', obiectiv.value)

    axe[2].imshow(U.value, cmap='gray')
    axe[2].set_title('Imagine Rezultata')
    axe[2].axis('off')

    plt.show()

    cv.imwrite('imagineRezultataTonuriGriA.png', U.value)



# Problema 1B
#
# Derivam expresia ce trebuie minimizata in raport cu U si obtinem gradientul, unde vom egala cu 0 fiecare componenta a sa, obtinand astfel constrangerile.
# O componenta oarecare din gradient este de forma: U[i, j] - Y[i, j] + 2.0 * rho * (U[i + 1, j] + U[i, j + 1] - 2.0 * U[i, j])
# Shape-ul gradientului este acelasi cu shape-ul lui U, (m, n).
def problema1B():
    imagineOriginala = cv.imread('imagineOriginalaTonuriGri.png')[:,:,0]
    imagineCorupta = cv.imread('imagineCoruptaTonuriGri.png')[:,:,0]

    U = cp.Variable(shape=imagineOriginala.shape)
    Y = cp.Parameter(shape=imagineCorupta.shape, value=imagineCorupta)
    Z = cp.Parameter(shape=imagineOriginala.shape, value=imagineOriginala)
    rho = 5.0

    obiectiv = cp.Minimize(cp.Constant(0.0))

    constrangeri = [U[:-1,:-1] - Y[:-1,:-1] + 2.0 * rho * (U[1:,:-1] + U[:-1,1:] - 2.0 * U[:-1,:-1]) == 0.0]

    problema = cp.Problem(obiectiv, constrangeri)

    problema.solve(verbose=True, solver=cp.SCS, ignore_dpp=True)

    print('Valoarea optima a obiectivului: ', obiectiv.value)

    figura, axe = plt.subplots(1, 3, figsize=(10, 5))
    axe[0].imshow(imagineOriginala, cmap='gray')
    axe[0].set_title('Imagine Originala')
    axe[0].axis('off')
    axe[1].imshow(imagineCorupta, cmap='gray')
    axe[1].set_title('Imagine Corupta')
    axe[1].axis('off')
    axe[2].imshow(U.value, cmap='gray')
    axe[2].set_title('Imagine Rezultata')
    axe[2].axis('off')

    plt.show()

    cv.imwrite('imagineRezultataTonuriGriB.png', U.value)



# creareImagineTonuriGri()
# problema1A()
# problema1B()



# Problema 2



def rosenbrock(X) -> float:
    return 100.0 * (X[1] - X[0] ** 2) ** 2 + (1.0 - X[0]) ** 2



def problema2():
    solutiiNM = []
    solutiiPowell = []
    solutiiCG = []

    solutiiCVXPY = []

    NUM_SOLUTII = 5
    MINIM = -100.0
    MAXIM = 100.0
    for _ in range(NUM_SOLUTII):
        X0 = np.random.uniform(MINIM, MAXIM, 2)
        solutiiNM.append(minimize(rosenbrock, X0, method='Nelder-Mead'))
    for _ in range(NUM_SOLUTII):
        X0 = np.random.uniform(MINIM, MAXIM, 2)
        solutiiPowell.append(minimize(rosenbrock, X0, method='Powell'))
    for _ in range(NUM_SOLUTII):
        X0 = np.random.uniform(MINIM, MAXIM, 2)
        solutiiCG.append(minimize(rosenbrock, X0, method='CG'))
    '''
    for _ in range(NUM_SOLUTII):
        X0 = np.random.uniform(MINIM, MAXIM, 2)
        X0 = cp.Variable(shape=2, value=X0)
        obiectiv = cp.Minimize(100.0 * (X0[1] - X0[0] ** 2) ** 2 + (1.0 - X0[0]) ** 2)
        constrangeri = []
        problema = cp.Problem(obiectiv, constrangeri)
        problema.solve(verbose=True)
        solutiiCVXPY.append(X0.value)
    '''

    print('Nelder-Mead')
    for sol in solutiiNM:
        print(sol.x)
    print('Powell')
    for sol in solutiiPowell:
        print(sol.x)
    print('CG')
    for sol in solutiiCG:
        print(sol.x)

    NUM_PUNCTE = 100
    X = np.linspace(MINIM, MAXIM, NUM_PUNCTE)
    Y = np.linspace(MINIM, MAXIM, NUM_PUNCTE)
    X, Y = np.meshgrid(X, Y)
    Z = rosenbrock([X, Y])
    figura = plt.figure(figsize=(10, 5))
    axe = figura.add_subplot(111, projection='3d')
    axe.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    axe.set_title('Functia Rosenbrock')
    axe.set_xlabel('OX')
    axe.set_ylabel('OY')
    axe.set_zlabel('OZ')

    for solutie in solutiiNM:
        axe.scatter(solutie.x[0], solutie.x[1], rosenbrock(solutie.x), c='red', s=100)
    for solutie in solutiiPowell:
        axe.scatter(solutie.x[0], solutie.x[1], rosenbrock(solutie.x), c='green', s=100)
    for solutie in solutiiCG:
        axe.scatter(solutie.x[0], solutie.x[1], rosenbrock(solutie.x), c='blue', s=100)
    for solutie in solutiiCVXPY:
        axe.scatter(solutie[0], solutie[1], rosenbrock(solutie), c='yellow', s=100)

    axe.view_init(elev=90, azim=90)

    plt.show()



# problema2()



# Problema 3



'''
Functia obiectiv este suma elementelor matricei rezultate in urma inmultirii element-wise a matricei C cu matricea X.
Constrangerile:
- suma elementelor de pe fiecare linie a matricei X trebuie sa fie egala cu elementul corespunzator din vectorul A (dintr-o locatie de plecare vor pleca exact atatea produse cat sunt disponibile)
- suma elementelor de pe fiecare coloana a matricei X trebuie sa fie egala cu elementul corespunzator din vectorul B (intr-o destinatie vor ajunge exact atatea produse cat sunt necesare)
- numarul de elemente luate si transportate trebuie sa fie un numar intreg
Minimizam functia obiectiv.
'''



def problema3CVXPY():
    # A.shape[0] == C.shape[0]
    # B.shape[0] == C.shape[1]
    valoriA = np.array([
        1.0, 2.0, 3.0, 4.0
    ], dtype=np.float64)
    valoriC = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ], dtype=np.float64)
    valoriB = np.array([
        1.0, 2.0, 7.0
    ], dtype=np.float64)

    A = cp.Parameter(shape=valoriA.shape, value=valoriA)
    B = cp.Parameter(shape=valoriB.shape, value=valoriB)
    C = cp.Parameter(shape=valoriC.shape, value=valoriC)

    X = cp.Variable(shape=C.shape, integer=True)

    obiectiv = cp.Minimize(cp.sum(cp.multiply(C, X)))

    constrangeri = [
        X >= 0,
        cp.sum(X, axis=0) == B,
        cp.sum(X, axis=1) == A,
    ]

    problema = cp.Problem(obiectiv, constrangeri)

    problema.solve(verbose=True)

    print('Valoarea optima a obiectivului: ', obiectiv.value)
    print('Matricea X:')
    print(X.value)



def problema3SCIPY():
    # A.shape[0] == C.shape[0]
    # B.shape[0] == C.shape[1]
    valoriA = np.array([
        1.0, 2.0, 3.0, 4.0
    ], dtype=np.float64)
    valoriC = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ], dtype=np.float64)
    valoriB = np.array([
        1.0, 2.0, 7.0
    ], dtype=np.float64)

    A = []
    B = []
    C = valoriC.flatten()

    for i in range(valoriA.shape[0]):
        constrangereLinie = np.zeros(valoriC.shape)
        constrangereLinie[i, :] = 1
        A.append(constrangereLinie.flatten())
        B.append(valoriA[i])

    for j in range(valoriB.shape[0]):
        constrangereColoana = np.zeros(valoriC.shape)
        constrangereColoana[:, j] = 1
        A.append(constrangereColoana.flatten())
        B.append(valoriB[j])

    A = np.array(A)
    B = np.array(B)

    limite = [(0, None) for _ in range(valoriC.shape[0] * valoriC.shape[1])]

    solutie = linprog(C, A_eq=A, b_eq=B, bounds=limite, method='highs')

    print('Valoarea optima a obiectivului: ', solutie.fun)
    print('Matricea X:')
    print(solutie.x.reshape(valoriC.shape))



# problema3CVXPY()
# problema3SCIPY()



# Problema 4



'''
X va avea shape-ul (x, 1)
Q va avea shape-ul (q, x)
C va avea shape-ul (x, 1)
Rezultatul functiei obiectiv va avea shape-ul (1, 1)
A va avea shape-ul (a, x)
B va avea shape-ul (a, 1)
'''



def problema4():
    x = 3
    q = 4
    a = 2

    np.random.seed(7)

    MINIM = -10.0
    MAXIM = 10.0
    valoriX = np.random.uniform(0.0, MAXIM, x)
    valoriQ = np.random.uniform(MINIM, MAXIM, (q, x))
    valoriC = np.random.uniform(MINIM, MAXIM, (x, 1))
    valoriA = np.random.uniform(MINIM, MAXIM, (a, x))
    valoriB = np.random.uniform(MINIM, MAXIM, (a, 1))

    X = cp.Variable(shape=(x, 1), nonneg=True)
    Q = cp.Parameter(shape=(q, x), value=valoriQ)
    C = cp.Parameter(shape=(x, 1), value=valoriC)
    A = cp.Parameter(shape=(a, x), value=valoriA)
    B = cp.Parameter(shape=(a, 1), value=valoriB)

    Qsimetric = valoriQ.T @ valoriQ

    obiectiv = cp.Minimize(0.5 * cp.quad_form(X, Qsimetric) + C.T @ X)

    constrangeri = [
        A @ X == B
    ]

    problema = cp.Problem(obiectiv, constrangeri)

    problema.solve(verbose=True)

    print('CVXPY')
    print('Valoarea optima a obiectivului: ', obiectiv.value)
    print('Valoarea optima a variabilelor:')
    print(X.value)



    def obiectiv(X):
        return 0.5 * X.T @ Qsimetric @ X + valoriC.T @ X

    def constrangere(X):
        return valoriA @ X - valoriB.flatten()

    constrangeri = {
        'type': 'eq',
        'fun': constrangere
    }

    limite = [(0, None) for _ in range(x)]

    solutie = minimize(obiectiv, valoriX, constraints=constrangeri, bounds=limite)

    print('SCIPY')
    print('Valoarea optima a obiectivului: ', solutie.fun)
    print('Valoarea optima a variabilelor:')
    print(solutie.x)



# problema4()


