import numpy as np
import random
import math
from solution import solution
import time
def NRBO(fobj, lowb, upp, dim, N, MaxIt):
    DF = 0.6  # Deciding Factor for Trap Avoidance Operator

    # Initialize the bounds for each dimension
    lb = np.full(dim, lowb)
    ub = np.full(dim, upp)

    # Initialize the population and their fitness values
    Position = np.random.uniform(lb, ub, (N, dim))
    # Your specified first position
    # Pass metrics_callback to fobj during fitness calculation
    Fitness = np.array([fobj(ind) for ind in Position])

    # Determine the best and worst fitness in the initial population
    Best_Score = np.min(Fitness)
    Best_Pos = Position[np.argmin(Fitness)]
    Worst_Cost = np.max(Fitness)
    Worst_Pos = Position[np.argmax(Fitness)]

    # Initialize convergence curve
    CG_curve = []
    s = solution()
    timerStart = time.time()
    print('NRBO is optimizing  "' + fobj.__name__ + '"')

    # Main optimization loop
    for it in range(1, MaxIt + 1):
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / MaxIt) ** 3) ** 2  # Eq.(14.2)
        alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(3 * np.pi / 2 * beta)))  # Eq.(14.1)
        
        delta = (1 - ((2 * it) / MaxIt)) ** 5  # Dynamic parameter delta
        for i in range(N):
            # Randomly select two different indices
            P1 = np.random.choice(np.delete(np.arange(N), i), 2, replace=False)
            a1, a2 = P1

            # Calculate the step size rho
            rho = np.random.rand() * (Best_Pos - Position[i]) + np.random.rand() * (Position[a1] - Position[a2])

            # Apply Newton-Raphson Search Rule
            NRSR = SearchRule(Best_Pos, Worst_Pos, Position[i], rho, Flag=1)
            X1 = Position[i] - NRSR + rho
            X2 = Best_Pos - NRSR + rho

            # Initialize Xupdate
            Xupdate = np.zeros(dim)
            for j in range(dim):
                rand_val = np.random.randn()  # Generate a single random value
                
                # Ensure scalar output from all parts of the equation
                term1 = X1[j]  # Scalar value
                term2 = np.random.rand() * (Best_Pos[j] - Position[i, j])  # Scalar value
                term3 = np.abs(X1[j] - Best_Pos[j]) + rand_val  # Scalar value

                if np.random.rand() < DF:
                    Xupdate[j] = np.multiply(np.add(term1, term2), term3)
                else:
                    term4 = np.random.rand() * X1[j] - X2[j]  # Scalar value
                    Xupdate[j] = (X1[j] - X2[j]) + np.random.rand() * np.abs(term4 + rand_val)

            # Trap Avoidance Operator
            if np.random.rand() < DF:
                theta1 = -1 + 2 * np.random.rand()
                theta2 = -0.5 + np.random.rand()
                beta = np.random.rand() < 0.5
                u1 = beta * 3 * np.random.rand() + (1 - beta)
                u2 = beta * np.random.rand() + (1 - beta)
                if u1 < 0.5:
                    X_TAO = Xupdate + theta1 * (u1 * Best_Pos - u2 * Position[i]) + theta2 * delta * (u1 * np.mean(Position, axis=0) - u2 * Position[i])
                else:
                    X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position[i]) + theta2 * delta * (u1 * np.mean(Position, axis=0) - u2 * Position[i])
                Xnew = X_TAO
            else:
                Xnew = Xupdate

                Xnew = np.clip(Xnew, lb, ub)
                Xnew_Cost = fobj(Xnew)
                if Xnew_Cost < Fitness[i]:
                    Position[i] = Xnew
                    Fitness[i] = Xnew_Cost

                    # Update global best solution
                    if Xnew_Cost < Best_Score:
                        Best_Score = Xnew_Cost
                        Best_Pos = Xnew

                # Update global worst solution
                if Fitness[i] > Worst_Cost:
                    Worst_Cost = Fitness[i]
                    Worst_Pos = Position[i]

            # Update convergence curve
        CG_curve.append(Best_Score)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.best = Best_Score
    s.bestIndividual = Best_Pos
    s.convergence = CG_curve
    s.optimizer = "NRBO"
    s.objfname = fobj.__name__

    return s

def SearchRule(Best_Pos, Worst_Pos, Position, rho, Flag):
    dim = len(Position)
    DelX = np.random.rand(dim) * np.abs(Best_Pos - Position)

    # Initial Newton-Raphson step
    NRSR = np.random.randn() * ((Best_Pos - Worst_Pos) * DelX) / (2 * (Best_Pos + Worst_Pos - 2 * Position))

    # Adjust position based on flag
    Xa = Position - NRSR + rho if Flag == 1 else Best_Pos - NRSR + rho

    # Further refine the Newton-Raphson step
    r1, r2 = np.random.rand(), np.random.rand()
    yp = r1 * (np.mean(Xa + Position) + r1 * DelX)
    yq = r2 * (np.mean(Xa + Position) - r2 * DelX)
    NRSR = np.random.randn() * ((yp - yq) * DelX) / (2 * (yp + yq - 2 * Position))

    return NRSR