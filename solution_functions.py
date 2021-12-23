
import numpy as np

def output_Displacements_Reactions (displacements,stiffness,G_dof,prescribed_Dof):
# output of displacements and reactions in tabular form
# GDof: total number of degrees of freedom of the problem

    # displacements
    print('\n', "Displacements:")
    jj = np.arange(0,G_dof)
    for i in jj:
        print('第{0:1d}行  '.format(jj[i]) , displacements[i,0])
    
    # reactions
    F =  np.matmul(stiffness,displacements)
    reactions = F[prescribed_Dof]
    print('\n', "Recations:")
    print(reactions)


def shapeFunctionL2 (xi):
# shape function and derivatives for L2 elements
# shape: Shape functions
# xi: natural coordinates (-1 ... +1)
    shape_N = np.array([1-xi,1+xi])
    shape_N = np.transpose(shape_N/2)
    a = np.array([-1,1])
    natural_Derivatives = a/2
    return(shape_N,natural_Derivatives)

# def stiffness_row_col(element_DOF,e):
# #   e : for loop  from stiffness
#     #element_DOF = element_nodes[e, :] - 1
#     element_DOF = np.reshape(element_DOF, [2, 1])
#     row = np.transpose(element_DOF) - 1
#     col = element_DOF
#     return row,col


def solution_problem_2(G_Dof, active_Dof, stiffness, force):
    #     function to find solution in terms of global displacements
    #     GDof: number of degree of freedom
    #     prescribedDof: bounded boundary dofs
    #     stiffness: stiffness matrix
    #     force: force vector
    a_dofs = np.reshape(active_Dof, [2, 1])
    row = np.transpose(a_dofs)
    col = a_dofs
    stiffness = stiffness[row,col]

    stiffness_inv = np.linalg.inv(stiffness)
    U = np.dot(stiffness_inv,force[active_Dof])

    displacements = np.zeros([G_Dof,1])

    displacements[active_Dof] = np.reshape(U,[len(active_Dof),1])
    return displacements




