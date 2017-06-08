"""
Created on Mon Jul 25 17:05:28 2016
Core functions of SAFE calculations

@author: MKK
"""
from __future__ import print_function
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt

class WaveElementAxisym(object):

    """Class for an axisymmetric wave element.

    WaveElementAxisym object contains the mesh and the properties of the
    cross-section of the waveguide and allows for calculating its dispersion
    curves and forced response.
    """
    def __init__(self, mesh):
        """ Initiate a WaveElementAxisym object

        Parameters:
            mesh : a Mesh object from SAFE_mesh
        """
        self.Mesh = mesh
        self.evaluated = False
        self.k_ready = None
        self.k = None
        self.psi_hat = None
        self.er = None
        self.K2_hat = None
        self.K3_hat = None
        self.KF_hat = None
        self.KFt_hat = None
        self.T = None
        self.B = None
        self.w = None
        self.e_p = None
        self.e_k = None
        self.p = None
        self.en_vel = None
        self.propagating_indices = None
#        if not self.Mesh.is_axisym:
#            print('Mesh error. An axisymmetric mesh must be provided.')

    def t_transform(self):
        """ Applies a T-transformation and makes the skew-symmetric matrices
        symmetric.

        The transformation is based on the observation that the radial
        displacement is in quadrature with the axial and tangential components.
        If these relationships are enforces using a diagonal transformation
        matrix T, respective matrices which are skew-symmetric become symmetric
        as the whole eigensystem does.
        """
        T = sps.diags((self.Mesh.Tdiag))
        self.K2_hat = T.dot(self.Mesh.K2).dot(T.conj())/(-1j)
        self.K3_hat = T.dot(self.Mesh.K3).dot(T.conj())/(-1j)
        self.KF_hat = T.dot(self.Mesh.KF).dot(T.conj())/(-1j)
        self.KFt_hat = T.dot(self.Mesh.KFt).dot(T.conj())/(-1j)
        self.T = T

    def solve(self, f, no_of_waves='full', central_wavespeed=0):
        """ Finds and sorts the wavenumbers and associated wave mode shapes
        for a given frequency range. Since the problem is symmetric, the left
        eigenvectors are the same as the right eigenvectors.

        This method solves the SAFE eigenvalue problem at each frequency
        specified in the frequency vector. It can either perform a full
        solution or use a sparse eigen solver. The former is preferred as it
        is expected to me more efficient for most axisymmetric configurations.
        If the sparse solution is desired, the solver uses the shift-invert
        mode around the wavenumber calculated from the central wavespeed.

        At each frequency step the wave solutions are sorted according to the
        biorthogonality criterion which in most cases provides continuous
        disperion curves for each wave.

        Parameters:
            f : frequency vector
            no_of_waves : number of waves to be found for sparse eigensolvers
                        to request a full solution use 'full' (default)
            central_wavespeed : wavespeed around which the solutions will be
                        sought if using the shift-invert mode.


        """
        print('Calculating dispersion curves...')
        self.t_transform()
        n = self.Mesh.n
        self.w = 2*np.pi*f
        if no_of_waves == 'full':
            self.k = np.zeros([len(f), 2*self.Mesh.no_of_dofs], 'complex')
            self.psi_hat = np.zeros([len(f), 2*self.Mesh.no_of_dofs,
                                     2*self.Mesh.no_of_dofs],
                                    'complex')
        else:
            self.k = np.zeros([len(f), no_of_waves], 'complex')
            self.psi_hat = np.zeros([len(f), 2*self.Mesh.no_of_dofs, no_of_waves],
                                    'complex')

        self.B = sps.bmat([[None, self.Mesh.K6.tocsr()],
                           [self.Mesh.K6.tocsr(), n*self.Mesh.K4.tocsr() +
                            self.K3_hat.tocsr()]])
        Binv = spla.inv(self.B.tocsc())
        for i in range(len(f)):
            print(str(i) + ' out of ' + str(len(f)))
            if self.Mesh.is_coupled:
                A = sps.bmat([[self.Mesh.K6.tocsr(), None],
                              [None, -(self.Mesh.K1.tocsr() +
                                     n**2*self.Mesh.K5.tocsr() +
                                     n*self.K2_hat.tocsr() -
                                     self.w[i]**2*self.Mesh.M.tocsr() +
                                     1j*self.w[i]*self.Mesh.K1_coupling.tocsr())]])
            else:
                A = sps.bmat([[self.Mesh.K6.tocsr(), None],
                              [None, -(self.Mesh.K1.tocsr() +
                                     n**2*self.Mesh.K5.tocsr() +
                                     n*self.K2_hat.tocsr() -
                                     self.w[i]**2*self.Mesh.M.tocsr())]])
            S = Binv.dot(A.tocsc())
            if no_of_waves == 'full':
                val, vec = la.eig(S.toarray())
            else:
                val, vec = spla.eigs(S.tocsc(), k=no_of_waves,
                                     sigma=(self.w[i]/central_wavespeed),
                                     which='LM')
            # create the normalisation factor (biorthogonality)
            norm_factor = 1/np.diag(vec.T.dot(np.asarray(self.B.toarray())).dot(vec))**0.5

            # sort the waves based on the biorthogonality relation
            # zeroth set - noi changes
            if i == 0:
                self.k[i] = val
                # normalize eigenvectros with respect to biorthogonality
                self.psi_hat[i] = vec*norm_factor
                # separate the solutions into positive and negative going
            else:
                # previous eigenvector
                ups1 = self.psi_hat[i - 1]
                # current eigenvector
                psi2 = vec*norm_factor
                # biorthogonality relation
                biorth = ups1.T.dot(np.asarray(self.B.toarray())).dot(psi2)
                # identify the maxima - similarity between the eigenvectors
                new_order = np.argmax(abs(biorth), axis=1)
                # sanity check - are they are close to unity?
                valid = np.round(abs(biorth[range(len(biorth)), list(new_order)]), 2) > 0.9
                # leave only those which make sense
                valid_entries = np.delete(new_order, np.where(valid == False))
                # if not all are close to unity
                if not valid.all():
                    # identify those which do not match witn any preceding eigenvectors
                    unassigned = list(set(range(len(biorth))) - set(valid_entries))
                    # go through all validity checks
                    j = 0
                    for ii in range(len(valid)):
                        # if there is no match
                        if not valid[ii]:
                            # append one of the unassigned solutions at the end
                            new_order[ii] = unassigned[j]
                            j += 1
                # reorder the eigenvectors
                new_psi = psi2[:, list(new_order)]
                # reorder the eigenvalues vector
                new_k = val[new_order]
                # assign final versions
                self.k[i] = new_k
                self.psi_hat[i] = new_psi
            self.evaluated = True
            self.k_ready = 0

    def energy_ratio(self, core_sets=None):
        """ Evaluates the ratio between the kinetic energies in the PML and
        in the whole waveguide.

        Parameters:
            core_sets : a list with user-defined core sets. If empty, the
                        method determines the core sets and PML sets by itself.

        """
        # check if the solution is stored
        if self.evaluated:
            print('Calculating the energy ratio...')
            # identify the PML and non-PML elements and dofs
            PML_dofs, PML_elements = [], []
            core_dofs, core_elements = [], []
            for key, val in self.Mesh.dofs_per_elset.items():
                if 'PML' in key:
                    PML_dofs.extend(val)
                    PML_elements.extend(self.Mesh.element_sets[key])
                elif 'PML' not in key and core_sets is None:
                    core_dofs.extend(val)
                    core_elements.extend(self.Mesh.element_sets[key])
            # since sets are not exclusive, the 'difference' between the
            # two lists is taken
            core_elements = list(set(core_elements) - set(PML_elements))
            PML_dofs.sort()
            core_dofs.sort()

            # determine core stes if no user-defined were specified
            if core_dofs is None:
                for elset in core_sets:
                    core_dofs.extend(self.Mesh.dofs_per_elset[elset])
                    core_elements.extend(self.Mesh.element_sets[elset])
            else:
                core_dofs = list(set([dof for element in core_elements
                                      for dof in self.Mesh.LM[element]]))

            # create mesh subsets corresponding to the element sets
            core_subset = self.Mesh.assemble_subset(core_elements)
            PML_subset = self.Mesh.assemble_subset(PML_elements)

            # extract mass matrices for subsets
            M_core = core_subset['M'].toarray()
            M_PML = PML_subset['M'].toarray()
            M_total = self.Mesh.M.toarray()

            # acoustic dofs need to be multiplied by -1 to get the physical
            # information.
            if self.Mesh.dof_domains['fluid'] != []:
                multiplier = np.ones(M_core.shape[0])
                multiplier[self.Mesh.dof_domains['fluid']] = -1
                M_core *= multiplier
                M_PML *= multiplier
                M_total *= multiplier


            # subset matrices are full size but contain zero rows/columns for
            # the dofs not included in the set
            # these zero rows/columns are filtered out below
            M_core = M_core[~np.all(M_core == 0, axis=0)][:, ~np.all(M_core == 0, axis=0)]
            M_PML = M_PML[~np.all(M_PML == 0, axis=0)][:, ~np.all(M_PML == 0, axis=0)]

            T = np.tile(self.T.toarray().conj(), (len(self.w), 1, 1))
            # get the physical displacement by reverting the T-transformation
            q_total = np.matmul(T, self.psi_hat[:, self.Mesh.no_of_dofs:, :])

            # extract the subsets
#            q_core = q_total[:, core_dofs, :]
            q_PML = q_total[:, PML_dofs, :]

            # calculate the energies and the energy ratio ER
            E_tot = (self.w.reshape(-1, 1))**2/4*\
                                (np.matmul(q_total.conj().transpose(0, 2, 1),
                                M_total).transpose(0, 2, 1)*q_total).sum(axis=1)
            E_PML = (self.w.reshape(-1, 1))**2/4*\
                                (np.matmul(q_PML.conj().transpose(0, 2, 1),
                                M_PML).transpose(0, 2, 1)*q_PML).sum(axis=1)
#            E_core = (self.w.reshape(-1, 1))**2/4*\
#                                (np.matmul(q_core.conj().transpose(0, 2, 1),
#                                M_core).transpose(0, 2, 1)*q_core).sum(axis=1)
            self.er = abs(E_PML)/abs(E_tot)
            # set a flag that the ER has been calculated
        else:
            print('No solution is availble. Solve the system first.')

    def k_propagating(self, imag_threshold=10, ER_threshold=0.9):
        """ Extracts positive-going, propagating solutions.

        Identifies the indices corresponding to positive-going, propagating
        waves based on the energy ratio threshold and a maximum allowable
        imaginary part of the wavenumber. Note that since the propagating
        indices  refer to wave solutions which at any frequency point fullfil
        the criteria described above (not necessarily across the whole
        frequency range).

        Parameters:
            imag_threshold : maximum allowable imaginary part (both + and -)
            ER_threshold : energy ratio threshold; solutionf with
                        ER > ER_threshold are regarded as non-propagating
        """
        if self.er is not None:
            k = np.copy(self.k)
            k[abs(k.imag) > imag_threshold] = 0
            if ER_threshold != 0:
                k[self.er > ER_threshold] = 0
            k[k.real < 0] = 0
#            k[k.imag > 1e-7] = 0
            #k[k.imag > 1e-3] = 0
            self.propagating_indices = np.where(~(k == 0).all(0))[0]
            # create a propagating wavenumbers array (remove the zero columns)
            self.k_ready = k[:, self.propagating_indices]
        else:
            print('Warning. Energy ratio must be evaluated first. ' +\
                  'Use the trace_back method')

    def energy_velocity(self, core_sets=None, only_propagating=True):
        """ Calculates the energy velocity.

        This method calculates the kinetic energy, the potential energy and
        the power flow density (Poynting vector) for desired wave solutions.
        Based on these, the energy velocity is calculated. By default the
        aforementioned quantities are evaluated for all but the PML layers.
        However, the user may specify his own set of core sets.

        Parameters:
            core_sets : list - sets for which the energies, power and the energy
                        velocity are calculated. By default it is empty,
                        and all layers but the PML are taken into calcualtion.
            only_propagating : boolean; determines if the calculation should
                            be performed only for the propagating solutions.
                            default is True.
        """
        # extract the physical displacements
        PML_dofs, PML_elements = [], []
        core_dofs, core_elements = [], []
        for key, val in self.Mesh.dofs_per_elset.items():
            if 'PML' in key  and len(core_sets) == 0:
                PML_dofs.extend(val)
                PML_elements.extend(self.Mesh.element_sets[key])
            if 'PML' not in key and len(core_sets) == 0:
                core_dofs.extend(val)
                core_elements.extend(self.Mesh.element_sets[key])
        # since sets are not exclusive, the 'difference' between the two lists is taken
        core_elements = list(set(core_elements) - set(PML_elements))

        # determine core stes if no user-defined were specified
        if core_dofs is None:
            for elset in core_sets:
                core_dofs.extend(self.Mesh.dofs_per_elset[elset])
                core_elements.extend(self.Mesh.element_sets[elset])
        else:
            core_dofs = list(set([dof for element in core_elements
                                  for dof in self.Mesh.LM[element]]))

        # create the subset
        core_subset = self.Mesh.assemble_subset(core_elements)
        # extract subset matrices
        M = core_subset['M'].toarray()
        K_1 = core_subset['K1'].toarray()
        K_2 = core_subset['K2'].toarray()
        K_3 = core_subset['K3'].toarray()
        K_4 = core_subset['K4'].toarray()
        K_5 = core_subset['K5'].toarray()
        K_6 = core_subset['K6'].toarray()
        KFt = core_subset['KFt'].toarray()
        KF = core_subset['KF'].toarray()

        # acoustic dofs need to be multiplied by -1 to get the physical
        # values
        if self.Mesh.dof_domains['fluid'] != []:
            multiplier = np.ones(self.Mesh.no_of_dofs)
            multiplier[self.Mesh.dof_domains['fluid']] = -1
            M *= multiplier
            K_1 *= multiplier
            K_2 *= multiplier
            K_5 *= multiplier
            K_6 *= multiplier
            KFt *= multiplier
            KF *= multiplier

        # remove the all zero rows/columns from the subset matrices
        # non-zerorows and columns
        nonzero = ~np.all(M == 0, axis=0)
        M = M[nonzero][:, nonzero]
        K_1 = K_1[nonzero][:, nonzero]
        K_2 = K_2[nonzero][:, nonzero]
        K_3 = K_3[nonzero][:, nonzero]
        K_4 = K_4[nonzero][:, nonzero]
        K_5 = K_5[nonzero][:, nonzero]
        K_6 = K_6[nonzero][:, nonzero]
        KF = KF[nonzero][:, nonzero]
        KFt = KFt[nonzero][:, nonzero]

        # get the physical displacement by reverting the T-transformation
        T = np.tile(self.T.toarray().conj(), (len(self.w), 1, 1))
        disp = np.matmul(T, self.psi_hat[:, self.Mesh.no_of_dofs:, :])
        if not only_propagating:
            disp_prop = disp
            k = self.k[:, :, np.newaxis, np.newaxis]
        else:
            if self.propagating_indices == []:
                self.k_propagating()
            disp_prop = disp[:, :, self.propagating_indices]
            k = self.k[:, self.propagating_indices, np.newaxis, np.newaxis]

        n = self.Mesh.n
        disp_core = disp_prop[:, core_dofs, :]
#        negative_disp_core = disp[:, core_dofs, negative_prop]
        for_e_p = K_1 + 1j*n*K_2 + n**2*K_5 + 1j*k.conj()*KF - 1j*k*KF.T + \
                  k*n*KFt.T + k.conj()*n*KFt + k*k.conj()*K_6
        for_p = KF - 1j*n*KFt - 1j*k*K_6

        self.e_k = (self.w.reshape(-1, 1))**2/4*\
                    (np.matmul(disp_core.conj().transpose(0, 2, 1),
                               M).transpose(0, 2, 1)*\
                               disp_core).sum(axis=1)
        self.e_p = 1./4*np.matmul(np.matmul(
                disp_core[:, :, :, np.newaxis].conj().transpose(0, 2, 3, 1),
                for_e_p),
                disp_core[:, :, :, np.newaxis].transpose(0, 2, 1, 3)).squeeze()
        self.p = -self.w.reshape(-1, 1)/2*np.imag(np.matmul(np.matmul(
                disp_core[:, :, :, np.newaxis].conj().transpose(0, 2, 3, 1), for_p),
                disp_core[:, :, :, np.newaxis].transpose(0, 2, 1, 3)).squeeze())
        self.en_vel = self.p/(self.e_k + self.e_p)

    def point_excited_waves(self, theta_0, r_f, f_ext, propagating=False):
        """ Calculates excited wave amplitudes for a point force.

        This method calculates the excited wave amplitudes induced by a point
        force. It is represented by a fourier series along the tangential
        coordinate for a small box function with length theta_0 taken as very
        small - this results in the sinc function appearing below. See
        Marzani (2008) for details.

        Parameters:
            theta_0 : a value for the small angle used to represent the point
                        excitation along the tangential direction
            r_f : a radial coordinate where the force is applied
            f_ext : a vector with zeros everywhere except for the degrees of
                    freedom at which the force is applied
            propaating : boolean; if True, only propagating waves are taken
                    for the solution; default is False.

        """
        force_vector = np.r_[np.zeros(self.Mesh.no_of_dofs),
                             f_ext].reshape(-1, 1)
        # determine the positive- and negative-going solutions
        if not propagating:
            return 1j*np.sinc(self.Mesh.n*theta_0/np.pi)/(2*np.pi*r_f)*\
                    np.matmul(self.psi_hat.transpose(0, 2, 1),
                              np.tile(force_vector, (len(self.w), 1, 1)))

    def calculate_frf(self, theta_0, r_f, f_ext, distance=0):
        """ Calculates the frequency response function for point force.

        This method calculates the displacement response to a point force at
        a given distance. The output is an array with all modal contributions
        which then need to be summed to obtain the total response.

        Parameters:
            theta_0 : a value for the small angle used to represent the point
                        excitation along the tangential direction
            r_f : a radial coordinate where the force is applied
            f_ext : a vector with zeros everywhere except for the degrees of
                    freedom at which the force is applied
            distance : distance at which the response is computed (positive);
                     by default it is an input FRF.

        """
        # create the three-dimensional propagation matrix

        # positive going wawes
        propagation = np.zeros([len(self.w), len(self.propagating_indices),
                                len(self.propagating_indices)], 'complex')
        for i in range(len(self.w)):
            propagation[i] = np.diag((np.exp(-1j*\
                       self.k[i, self.propagating_indices]*distance)))

        # calculate the excited waves and propagate them as required
        amps = np.matmul(propagation,
                      self.point_excited_waves(theta_0, r_f, f_ext)[:,
                                              self.propagating_indices])

        # extract the mode shapes
        shapes = self.psi_hat[:, self.Mesh.no_of_dofs:,
                              self.propagating_indices]

        # return the induced displacement for each wave
        return shapes*amps.transpose(0, 2, 1)


    # Plot results
    def plot_cp(self):
        """
        Plots phase velocity dispersion curves for waves solutions marked as
        progpatating according to a chosen (or default) criterion
        """
        if self.k_ready is None:
            if self.er is None:
                self.energy_ratio()
                self.k_propagating()
            else:
                self.k_propagating()
        k = self.k[:, self.propagating_indices]
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.plot(self.w/2e3/np.pi, 1e-3*self.w.reshape(-1, 1)/k.real, 'o')
        ax2.plot(self.w/2e3/np.pi, -20*np.log10(np.e)*k.imag, 'o')
        ax1.set_ylim([0, 10])
        ax2.set_ylim([0, 900])
        ax2.set_xlabel('frequency in kHz')
        ax1.set_ylabel(' cp in km/s')
        ax1.legend()
        ax2.set_ylabel(' attenuation in dB/m')
        plt.show()
