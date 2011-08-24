import numpy as np
import scipy.linalg
import mlpython.mathutils.linalg as linalg
import mlpython.mathutils.nonlinear as nonlinear

print "Testing product_matrix_vector"
A = np.random.rand(30,20)
b = np.random.rand(10)
c = np.zeros((40))
linalg.product_matrix_vector(A[:10,:].T,b,c[10:30])
print "Numpy vs mathutils.linalg diff.:",sum(np.abs(c[10:30] - np.dot(A[:10,:].T,b)))
A = np.random.rand(20,30)
b = np.random.rand(30)
c = np.zeros((20))
linalg.product_matrix_vector(A,b,c)
print "Numpy vs mathutils.linalg diff.:",sum(np.abs(c - np.dot(A,b)))
A = np.random.rand(30,20)
b = np.random.rand(20)
c = np.zeros((30))
linalg.product_matrix_vector(A,b,c)
print "Numpy vs mathutils.linalg diff.:",sum(np.abs(c - np.dot(A,b)))


print "Testing product_matrix_matrix"
A = np.random.rand(20,30)
B = np.random.rand(30,40)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A,B,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A,B)))
A = np.random.rand(30,20)
B = np.random.rand(30,40)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A.T,B,C)
np.dot(A.T,B)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A.T,B)))
A = np.random.rand(20,30)
B = np.random.rand(40,30)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A,B.T,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A,B.T)))
A = np.random.rand(30,20)
B = np.random.rand(40,30)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A.T,B.T,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A.T,B.T)))
A = np.random.rand(20,30)
B = np.random.rand(30,40)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A,B,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A,B)))
A = np.random.rand(30,20)
B = np.random.rand(30,40)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A.T,B,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A.T,B)))
A = np.random.rand(20,30)
B = np.random.rand(40,30)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A,B.T,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A,B.T)))
A = np.random.rand(30,20)
B = np.random.rand(40,30)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A.T,B.T,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A.T,B.T)))

print "Testing product_matrix_matrix"
A = np.zeros((20,30),order='fortran')
A[:] = np.random.rand(20,30)
B = np.random.rand(30,40)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A,B,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A,B)))
A = np.zeros((30,20),order='fortran')
A[:] = np.random.rand(30,20)
B = np.random.rand(30,40)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A.T,B,C)
np.dot(A.T,B)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A.T,B)))
A = np.zeros((20,30),order='fortran')
A[:] = np.random.rand(20,30)
B = np.random.rand(40,30)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A,B.T,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A,B.T)))
A = np.zeros((30,20),order='fortran')
A[:] = np.random.rand(30,20)
B = np.random.rand(40,30)
C = np.zeros((20,40))
linalg.product_matrix_matrix(A.T,B.T,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A.T,B.T)))
A = np.zeros((20,30),order='fortran')
A[:] = np.random.rand(20,30)
B = np.random.rand(30,40)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A,B,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A,B)))
A = np.zeros((30,20),order='fortran')
A[:] = np.random.rand(30,20)
B = np.random.rand(30,40)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A.T,B,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A.T,B)))
A = np.zeros((20,30),order='fortran')
A[:] = np.random.rand(20,30)
B = np.random.rand(40,30)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A,B.T,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A,B.T)))
A = np.zeros((30,20),order='fortran')
A[:] = np.random.rand(30,20)
B = np.random.rand(40,30)
C = np.zeros((40,20))
linalg.product_matrix_matrix(A.T,B.T,C.T)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C.T - np.dot(A.T,B.T)))

A = np.random.rand(1,30)
B = np.random.rand(30,1)
C = np.zeros((1,1))
linalg.product_matrix_matrix(A,B,C)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(C - np.dot(A,B)))

print "Testing getdiag"
A = np.random.rand(30,20)
x = np.zeros((20))
linalg.getdiag(A,x)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(x-np.diag(A)))
A = np.random.rand(30,20).T
x = np.zeros((20))
linalg.getdiag(A,x)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(x-np.diag(A)))

print "Testing setdiag"
A = np.random.rand(30,20)
x = np.random.rand(20)
linalg.setdiag(A,x)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(x-np.diag(A)))
A = np.random.rand(30,20).T
x = np.random.rand(20)
linalg.setdiag(A,x)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(x-np.diag(A)))

print "Testing solve"
A = np.random.rand(30,30)
A = np.dot(A,A.T)
B = np.random.rand(30,20)
X = np.zeros((30,20))
linalg.solve(A,B,X)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(X-np.linalg.solve(A,B)))
A = np.random.rand(30,30)
A = np.dot(A,A.T)
B = np.random.rand(30,30)
X = np.zeros((30,30))
linalg.solve(A,B,X)
print "Numpy vs mathutils.linalg diff.:",np.sum(np.abs(X-np.linalg.solve(A,B)))

print "Testing lu decomposition"

A = np.random.rand(4,6)
p = np.zeros((4),dtype='i')
L = np.zeros((4,4))
U = np.zeros((4,6))
linalg.lu(A,p,L,U)
# Writing permutation vector p in matrix form
P = np.zeros((4,4))
for P_row,p_el in zip(P.T,p):
    P_row[p_el] = 1
P2,L2,U2 = scipy.linalg.lu(A)
print "Scipy vs mathutils.linalg diff. P:",np.sum(np.abs(P-P2))
print "Scipy vs mathutils.linalg diff. L:",np.sum(np.abs(L-L2))
print "Scipy vs mathutils.linalg diff. U:",np.sum(np.abs(U-U2))

A = np.random.rand(20,30)
p = np.zeros((20),dtype='i')
L = np.zeros((20,20))
U = np.zeros((20,30))
linalg.lu(A,p,L,U)
# Writing permutation vector p in matrix form
P = np.zeros((20,20))
for P_row,p_el in zip(P.T,p):
    P_row[p_el] = 1
P2,L2,U2 = scipy.linalg.lu(A)
print "Scipy vs mathutils.linalg diff. P:",np.sum(np.abs(P-P2))
print "Scipy vs mathutils.linalg diff. L:",np.sum(np.abs(L-L2))
print "Scipy vs mathutils.linalg diff. U:",np.sum(np.abs(U-U2))

A = np.random.rand(30,20)
p = np.zeros((30),dtype='i')
L = np.zeros((30,20))
U = np.zeros((20,20))
linalg.lu(A,p,L,U)
# Writing permutation vector p in matrix form
P = np.zeros((30,30))
for P_row,p_el in zip(P.T,p):
    P_row[p_el] = 1
P2,L2,U2 = scipy.linalg.lu(A)
print "Scipy vs mathutils.linalg diff. P:",np.sum(np.abs(P-P2))
print "Scipy vs mathutils.linalg diff. L:",np.sum(np.abs(L-L2))
print "Scipy vs mathutils.linalg diff. U:",np.sum(np.abs(U-U2))

print 'Testing nonlinear sigmoid'
input = np.random.randn(30,20)
output = np.zeros((30,20))
nonlinear.sigmoid(input,output)
print 'Numpy vs mathutils.nonlinear diff. output:',np.sum(np.abs(output-1/(1+np.exp(-input))))

print 'Testing nonlinear sigmoid deriv.'
dinput = np.zeros((30,20))
doutput = np.random.randn(30,20)
nonlinear.dsigmoid(output,doutput,dinput)
print 'Numpy vs mathutils.nonlinear diff. output:',np.sum(np.abs(dinput-doutput*output*(1-output)))

print 'Testing nonlinear softmax'
input = np.random.randn(20)
output = np.zeros((20))
nonlinear.softmax(input,output)
print 'Numpy vs mathutils.nonlinear diff. output:',np.sum(np.abs(output-np.exp(input)/np.sum(np.exp(input))))

print 'Testing nonlinear reclin'
input = np.random.randn(30,20)
output = np.zeros((30,20))
nonlinear.reclin(input,output)
print 'Numpy vs mathutils.nonlinear diff. output:',np.sum(np.abs(output-(input>0)*input))

print 'Testing nonlinear reclin deriv.'
dinput = np.zeros((30,20))
doutput = np.random.randn(30,20)
nonlinear.dreclin(output,doutput,dinput)
print 'Numpy vs mathutils.nonlinear diff. output:',np.sum(np.abs(dinput-(input>0)*doutput))
