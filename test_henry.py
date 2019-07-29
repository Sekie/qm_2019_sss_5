import noble_gas_model
import hartree_fock as hf
import numpy as np

ar2 = noble_gas_model.NobleGasModl()
atomic_coordinates = np.array([[0.0,0.0,0.0], [3.0,4.0,5.0]])
myHF = hf.HartreeFock(ar2, atomic_coordinates)
#print("Interaction matrix\n", list(myHF.interaction_matrix.flatten()))
#print("Hamiltonian matrix\n", list(myHF.hamiltonian_matrix.flatten()))
#print("Density matrix\n", list(myHF.density_matrix.flatten()))
print("Interaction matrix\n", list(myHF.interaction_matrix.flatten()))
print("Hamiltonian matrix\n", list(myHF.hamiltonian_matrix.flatten()))
#myHF.calculate_chi_tensor(ar2)
fock = myHF.calculate_fock_matrix()
print(fock)
fock = myHF.calculate_fock_matrix_fast(myHF.hamiltonian_matrix, myHF.interaction_matrix, myHF.density_matrix, ar2.model_parameters)
print(fock)

# print("atom")
# for i in range(8):
#     print(ar2.atom(i))

# print("orb")
# for i in range(8):
#     print(ar2.orb(i))

# print("ao_index")
# for i in range(2):
#     for o in ar2.orbital_types:
#         print(ar2.ao_index(i, o))

# print("chi_on_atom")
# for i in ar2.orbital_types:
#     for j in ar2.orbital_types:
#         for k in ar2.orbital_types:
#             print(myHF.chi_on_atom(i, j, k, ar2))