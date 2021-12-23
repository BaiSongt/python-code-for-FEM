#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:09:59 2021

@author: songtaobai
"""

# python code for Finite Element Analysis
# Problem1.py
# Baisongtao 20211202

import numpy as np
import solution_functions as sol_fun

# element_nodes : connections at elements
element_nodes = np.array([[1, 2], [2, 3], [2, 4]])

# number_elements: number of Elements
number_elements = np.size(element_nodes, 0)

# number_nodes : number of nodes
number_nodes = 4

''' for structure:
        displacements : displacement vector
        force : force vector
        stiffness : stiffness matrix
'''
displacements = np.zeros((number_nodes, 1))
force = np.zeros((number_nodes, 1))
stiffness = np.zeros((number_nodes, number_nodes))
stiffness_ele = np.array([[1, -1], [-1, 1]])
# applied load at node 2
force[1] = 10.0

# computation of the system stiffness matrix
for e in range(number_elements):
    # element_DOF
    element_DOF = element_nodes[e, :] - 1
    element_DOF = np.reshape(element_DOF, [2, 1])
    row = np.transpose(element_DOF)
    col = element_DOF
    stiffness[row, col] = stiffness[row, col] + stiffness_ele


# boundary conditions and solution
# prescribed dofs
prescribed_Dof = np.array([[1, 3, 4]])
# free Dof: active_Dof
active_Dof = np.setdiff1d(np.arange(1, number_nodes), prescribed_Dof)

# solution
displacements[active_Dof - 1] = force[active_Dof - 1] / \
    stiffness[active_Dof - 1, active_Dof - 1]

# output stiffness
print('\n', "Stiffness matrix:")
print(stiffness)

# output displacements/reactions
sol_fun.output_Displacements_Reactions(
    displacements, stiffness, number_nodes, prescribed_Dof-1)
