import numpy as np
from numpy.linalg import eig
from numpy.linalg import norm
data = open('pageRank-gr0.California.txt', 'r')

# Calculating size of edge list
# data_tot_lines = data.readlines()
# print(len(data_tot_lines)) # 25815 16149

a = True
edge_list = np.empty((16150, 2), dtype=int)
i = 0
M=np.zeros((9664,9664), dtype=float) # 9663 total nodes, create empty matrix to store edge relationships

# Reading data and creating edge list
while a:
    data_line = data.readline()
    if not data_line:
        print("eof")
        a = False
        break
    if data_line[0] == 'e':
        t = data_line.split()
        edge_list[i][0] = t[1]
        edge_list[i][1] = t[2]
        i = i + 1

# Creating link matrix, binary
for i in range(len(edge_list)):
    M[edge_list[i][1]][edge_list[i][0]] = 1 

# Scaling with number of outlinks for each column
column_sum = M.sum(axis=0)

# Create d
d = np.zeros(len(column_sum))
for x in range(len(column_sum)):
 if(column_sum[x] == 0):
     d[x]=1

for x in range(len(column_sum)):
    if column_sum[x] == 0:
        column_sum[x] = 1

Mt=M.T
column_sum_t = column_sum.T
Mt = Mt/column_sum_t
M=Mt.T

e = np.ones(len(column_sum))
ed = np.outer(e,d)

P = M + 1/len(column_sum) * ed

alpha = 0.85
eeT = np.outer(e,e)
A = alpha * P + (1 - alpha) * 1/len(column_sum) * eeT



epsilon = 10e-4
residual = 1
i = 0

while residual > epsilon and i < 100:
    z = np.ones(len(column_sum)) / len(column_sum)
    v = np.ones(len(column_sum)) / len(column_sum)
    yhat = alpha * M * z
    beta = 1 - np.linalg.norm(yhat, ord=1)
    y = yhat + beta*v
    residual = np.linalg.norm(y - z, ord=1)
    i = i + 1


print(z)
