import numpy as np
import mp2_no_hf as mp2_no_hf
import hartree_fock as hf
import noble_gas_model as noble_gas_model
import fock_fast
import qm_project
from mpi4py import MPI

if __name__ == "__main__":
    NobleGasModel = noble_gas_model.NobleGasModl()
    atomic_coordinates = np.array([[0.0,0.0,0.0], [3.0,4.0,5.0]])
    #hartree_fock_instance = hf.HartreeFock(NobleGasModel, atomic_coordinates)
    #hartree_fock_instance.density_matrix = hartree_fock_instance.calculate_atomic_density_matrix(NobleGasModel)
    #hartree_fock_instance.density_matrix, hartree_fock_instance.fock_matrix = hartree_fock_instance.scf_cycle(NobleGasModel)

    dims = [2 ** i for i in range(10, 11)]
    for dim in dims:
        h = np.random.rand(dim, dim)
        h = h + h.transpose()
        trV = np.random.rand(dim, dim)
        trV = trV + trV.transpose()
        P = np.random.rand(dim, dim)
        P = P + P.transpose()
        chi = np.random.rand(dim, dim, dim)

        start_time = MPI.Wtime()
        fock_s = qm_project.calculate_fock_matrix(h, trV, P, chi)
        end_time = MPI.Wtime()
        slow_fock = end_time - start_time

        start_time = MPI.Wtime()
        fock_f = fock_fast.test_fock_matrix_fast(h, trV, P, NobleGasModel.model_parameters, int(dim / 2))
        end_time = MPI.Wtime()
        fast_fock = end_time - start_time

        print(dim, str(slow_fock), str(fast_fock))

