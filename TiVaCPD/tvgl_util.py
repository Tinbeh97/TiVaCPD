def update_z_all(slice_size, S, rho, alpha, beta, theta, z0, z1, z2, u0, u1, u2, p_type='l1'):
    #update z_0
    for i in range(len(z0)):
        A = theta[i] + u0[i]
        for m in range(A.shape[0]):
            for n in range(m + 1, A.shape[1]):
                if abs(A[m, n]) <= alpha / rho:
                    A[m, n] = 0
                    A[n, m] = 0
                else:
                    if A[m, n] > 0:
                        A[m ,n] = A[m, n] - alpha / rho
                        A[n ,m] = A[n, m] - alpha / rho
                    else:
                        A[m, n] = A[m, n] + alpha / rho
                        A[n, m] = A[n, m] + alpha / rho
        z0[i] = A

    #update z_1, z_2
    eta = 2 * beta / rho
    for i in range(1, len(z1)):
        A = theta[i] - theta[i - 1] + u2[i] - u1[i - 1]
        for m in range(A.shape[0]):
            for n in range(m, A.shape[1]):
                if (p_type=='l2'):
                    A[m, n] = (1 / (1 + 2 * eta)) * A[m, n]
                    A[n, m] = (1 / (1 + 2 * eta)) * A[n, m]
                elif (p_type=='l1'):
                    if abs(A[m, n]) <= eta:
                        A[m, n] = 0
                        A[n, m] = 0
                    else:
                        if A[m, n] > 0:
                            A[m, n] = A[m, n] - eta
                            A[n, m] = A[n, m] - eta
                        else:
                            A[m, n] = A[m, n] + eta
                            A[n, m] = A[n, m] + eta
        z1[i - 1] = (1 / 2) * (theta[i - 1] + theta[i] + u1[i - 1] + u2[i]) - (1 / 2) * A
        z2[i] = (1 / 2) * (theta[i - 1] + theta[i] + u1[i - 1] + u2[i]) + (1 / 2) * A

    return z0, z1, z2