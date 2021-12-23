#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename      :Problem2.py
@Time          :2021/12/17 09:25:15
@author        :Songtao Bai
@version       :1.0
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import solution_functions as sol_funs

# E: modulus of Elasticity
# A: Area of cross section
# L: length of bar
# p: loads of bar
E = 30e6
A = 1
EA = E*A
L = 90
p = 50

# generation of coordinates and connectivities
# numberElements: number of elements
number_Elements = 3
# generation equal spaced coordinates
node_coordinates = np.linspace(0, L, number_Elements+1)
xx = node_coordinates
# number_Ndes: number of nodes
number_Nodes = np.size(node_coordinates)
# element_nodes : coonections at elements
ii = np.linspace(1, number_Elements, number_Elements)
element_nodes = np.zeros([3, 2], dtype='int')
element_nodes[:, 0] = ii
element_nodes[:, 1] = ii + 1

'''
    for structure:
        displacement: displacement vector
        force       : force vector
        stiffness   : stiffness matrix
'''
displacement = np.zeros(number_Nodes)
force = np.zeros(number_Nodes)
stiffness = np.zeros([number_Nodes, number_Nodes])

# compution of the system stiffness matrix and force vector
for e in range(number_Elements):

    # ele_dof : element degrees of freedom
    ele_dof = element_nodes[e, :].astype(int)
    nn = len(ele_dof)

    # compute Jacobian
    length_element = node_coordinates[ele_dof[1] -
                                      1] - node_coordinates[ele_dof[0]-1]
    det_Jacobian = length_element/2
    inv_Jacobian = 1/det_Jacobian

    # central Gauss point (xi=0, weight W=2)
    # shape: shape function
    # natural_DEricatives
    [shape, natural_Derivatives] = sol_funs.shapeFunctionL2(0.0)
    Xderivatives = natural_Derivatives*inv_Jacobian

    # B matrix
    B = np.ones([1, nn])
    B = np.reshape(Xderivatives, [1, nn])
    BB = np.dot(np.transpose(B), B)
    # print(BB)

    # stiffness matrix
    ele_dof = np.reshape(ele_dof, [2, 1])
    row = np.transpose(ele_dof) - 1
    col = ele_dof - 1
    stiffness[row, col] = stiffness[row, col] + BB*2*det_Jacobian*EA
    # print(stiffness)

    # force vector
    force[row] = force[row] + 2*shape*p*det_Jacobian
    # print(force)

#  prescribed dofs 约束自由度
prescribed_dofs = np.zeros([2, 1], dtype="int")
prescribed_dofs[0] = np.where(xx == min(node_coordinates))[0]
prescribed_dofs[1] = np.where(xx == max(node_coordinates))[0]

# free dofs
active_Dof = np.setdiff1d(np.arange(number_Nodes), prescribed_dofs)

# solution
G_dof = number_Nodes
displacements = sol_funs.solution_problem_2(
    G_dof, active_Dof, stiffness, force)

# output displacements/reactions
sol_funs.output_Displacements_Reactions(
    displacements, stiffness, G_dof, prescribed_dofs)

# stresses at elements
sigma = np.zeros([number_Elements, 1])
for e in range(number_Elements):
    # ele_dof : element degrees of freedom
    ele_dof = element_nodes[e, :].astype(int)
    nn = len(ele_dof)

    # compute Jacobian
    length_element = node_coordinates[ele_dof[1] -
                                      1] - node_coordinates[ele_dof[0]-1]
    a = np.array([[-1, 1]])
    b = displacements[ele_dof-1]
    sigma[e] = E/length_element*(np.dot(a, b))

# drawing nodal displacements

plt.figure(1,figsize=[5,4])
plt.scatter(np.reshape(node_coordinates, [4, 1]), displacements,
            c='none', marker='o', edgecolors='black')

# graphical representation with interpolation for each element

interpNodes = 10

for e in range(number_Elements):
    node_A = element_nodes[e, 0]-1
    node_B = element_nodes[e, 1]-1
    XX = np.linspace(node_coordinates[node_A],
                     node_coordinates[node_B], interpNodes)
    ll = node_coordinates[node_B] - node_coordinates[node_A]
    # dimensionless coordinate
    xi = (XX - node_coordinates[node_A]) * 2/ll - 1
    # linear shape function
    phi1 = 0.5 * (1-xi)
    phi2 = 0.5 * (1+xi)
    # displacement at the element
    u = phi1 * displacements[node_A] + phi2 * displacements[node_B]
    # fmt = '[marker][line][color]'
    plt.figure(1)
    plt.plot(XX, u, color='black', linestyle='-', linewidth=1.5)
    plt.plot(XX, p*L*XX/2/EA*(1-XX/L), '--b', linewidth=1.5)

    # stress and at element
    sigma = E/ll * np.ones([1, interpNodes], dtype='float64') * (
        displacements[node_B] - displacements[node_A])
    XX = np.reshape(XX, [10, 1])
    sigma = np.reshape(sigma, [10, 1])
    plt.figure(2,figsize=[5,4])
    # plt.plot([1,2,3,4],[1,2,3,4])
    plt.plot(XX, sigma, color='black', linestyle='-', linewidth=1.5)
    plt.plot(XX, p * L/A * (0.5-XX/L), '--b', linewidth=1.5)

plt.figure(1)
plt.xlim(0, L)
plt.ticklabel_format(style='sci',scilimits=[-1,2],axis='y')

plt.figure(2)
plt.xlim(0, L)
plt.show()
