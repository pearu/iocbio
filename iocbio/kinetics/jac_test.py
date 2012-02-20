
import numpy
import scipy.linalg

import matplotlib.pyplot as plt

jac_array_A = numpy.loadtxt('generated/time_1930.0_jac_array.txt')
jac_array_B = numpy.loadtxt('generated/time_1940.0_jac_array.txt')
jac_array_C = numpy.loadtxt('generated/time_1950.0_jac_array.txt')


e_values_A, e_vectors_A = scipy.linalg.eig(jac_array_A)
e_values_B, e_vectors_B = scipy.linalg.eig(jac_array_B)
e_values_C, e_vectors_C = scipy.linalg.eig(jac_array_C)

za = zip(e_values_A, e_vectors_A)
zb = zip(e_values_B, e_vectors_B)
zc = zip(e_values_C, e_vectors_C)

sza = sorted(za)
szb = sorted(zb)
szc = sorted(zc)

e_values_A, e_vectors_A = zip(*sza)
e_values_B, e_vectors_B = zip(*szb)
e_values_C, e_vectors_C = zip(*szc)

tol = 1e-8
for i in range(len(za)):

    eva = e_values_A[i]
    evb = e_values_B[i]
    evc = e_values_C[i]

    if abs(eva) < tol:
        assert abs(eva - evb) < tol
        assert abs(eva - evc) < tol
    
        for j in range(len(e_vectors_A[i])):
            print e_values_A[i], e_vectors_A[i][j]
            print e_values_B[i], e_vectors_B[i][j]
            print e_values_C[i], e_vectors_C[i][j]
            print
    


fig = plt.figure(figsize=(8.5,11))

ax = fig.add_subplot(111)
ax.plot([1930], [e_values_A], ':')
ax.plot([1940], [e_values_B], ':')
ax.plot([1950], [e_values_C], ':')

#plt.show()
