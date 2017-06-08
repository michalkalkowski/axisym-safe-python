"""
Created on Thu Jul 21 15:57:46 2016

Pre-processing library for SAFE calculation routines

@author: michal
"""
from __future__ import print_function

from cmath import exp, pi
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from shape_functions import build_GLL_quadrature, build_GLJ_quadrature
import elements
 
def suggest_PML_parameters(PML_start, wavelenghts_within, waveguide_material,
                           surr_material, low_f, high_f, alpha=6, beta=7, att=1):
    """
    Function calculates suggested parameters for the exponential PML profile
    based on the properties of the waveguide and the surrounding medium and
    geometrial parameters. The values for alpha and beta are suggested
    as default based on the experience.

    Parameters:
        PML_start : start of the PML (physical units)
        wavelengths_within : desired number of 'shortened' wavelengths across
                            the PML (approx. between 2 and 8)
        waveguide_material : list with properties of the waveguide (as defined for sets)
        surr_material : list with properties of the surrounding medium (as defined for sets)
        low_f : low frequency limit
        high_f : high frequency limie
        alpha : exponential profile parameter (controlling shortening of the wavelength)
        beta : exponential profile parameter (controlling attenuation)
        att : assumed imaginary part of the propagating wavenumber

    Returns:
        h_long : suggested thickness for the PML to satisfy requirements
                at the long wavelength limit
        h_short : suggested thickness for the PML to satisfy requirements
                at the short wavelength limit
        order : suggested order of the PML GLL element
        R_short : reflection coefficient for the PML at the short wavelength limit
        R_long : reflection coefficient for the PML at the long wavelength limit
        R_short_at_long : reflection coefficient for the PML at the long wavelength limit
                            but with parameters for the short wavelength limit
    """
    w_test = 2*pi*np.array([low_f, high_f])
    cL_g = ((waveguide_material[0] + 2*waveguide_material[1])/waveguide_material[2])**0.5
    cS_g = ((waveguide_material[1])/waveguide_material[2])**0.5

    kL_g = w_test/cL_g - 1j*att
    kS_g = w_test/cS_g - 1j*att

    if len(surr_material) == 4:
        cL_s = ((surr_material[0] + 2*surr_material[1])/surr_material[2])**0.5
        cS_s = ((surr_material[1])/surr_material[2])**0.5
        kL_s = w_test/cL_s
        kS_s = w_test/cS_s
        kr_SS = (kS_s**2 - kS_g**2)**0.5
        kr_SL = (kS_s**2 - kL_g**2)**0.5
        kr_LS = (kL_s**2 - kS_g**2)**0.5
        kr_LL = (kL_s**2 - kL_g**2)**0.5

        kr_short = max([kr_LL[1], kr_LS[1], kr_SL[1], kr_SS[1]])
        kr_long = min([kr_LL[0], kr_LS[0], kr_SL[0], kr_SS[0]])
    else:
        cL_s = (surr_material[0]/surr_material[1])**0.5
        kL_s = w_test/cL_s
        kr_LS = (kL_s**2 - kS_g**2)**0.5
        kr_LL = (kL_s**2 - kL_g**2)**0.5

        kr_short = max([kr_LL[1], kr_LS[1]])
        kr_long = min([kr_LL[0], kr_LS[0]])

    kr_PML_long = kr_long.real*(exp(alpha) - 1)/alpha
    kr_PML_short = kr_short.real*(exp(alpha) - 1)/alpha
    d_long = 4*pi**2*wavelenghts_within**2/kr_PML_long**2 + \
             8*pi*PML_start*wavelenghts_within/kr_PML_long
    d_short = 4*pi**2*wavelenghts_within**2/kr_PML_short**2 + \
              8*pi*PML_start*wavelenghts_within/kr_PML_short

    h_long = (2*pi*wavelenghts_within/kr_PML_long + d_long**0.5)/2
    h_short = (2*pi*wavelenghts_within/kr_PML_short + d_short**0.5)/2

    element_order = pi*wavelenghts_within + 3

    alpha_factor = (exp(alpha) - 1)/alpha
    beta_factor = (beta - exp(beta) + 1)/beta
    R_short = np.exp(2*h_short*kr_short.imag*alpha_factor)*\
              np.exp(2*h_short*kr_short.real*beta_factor)

    R_long = np.exp(2*h_long*kr_long.imag*alpha_factor)*\
              np.exp(2*h_long*kr_long.real*beta_factor)

    R_short_at_long = np.exp(2*h_short*kr_long.imag*alpha_factor)*\
                      np.exp(2*h_short*kr_long.real*beta_factor)
    return h_long.real, h_short.real, int(np.ceil(element_order)),\
            R_short.real, R_long.real, R_short_at_long.real

class Mesh(object):
    """ Class for a mesh """

    def __init__(self):
        self.element_types, self.element_size, self.min_wavelengths = dict(), [], []
        self.glob_nodes, self.IEN, self.element_sets = dict(), dict(), dict()
        self.rev_PMLs = dict()
        self.element_props = dict()
        self.node_sets = dict()
        self.uncoupled_nodes, self.uncoupled_dofs = [], []
        self.nodes_types = dict()
        self.nodes_in_element = dict()
        self.struct_acoust = dict()
        self.struct_acoust_pairs = []
        self.n = None
        self.PML_props = None
        self.elements = dict()
        self.is_coupled = None
        self.dofs_per_elset = dict()
        self.no_of_nodes = 0
        self.no_of_dofs = 0
        # initialise a dictionary with degrees of freedom per node
        self.ID = dict()
        # initialise a dictionary which will contain dofs for both fluid and solid parts
        self.dof_domains = dict()
        # create a location dictionary that maps a list of degrees of freedom to
        # each element.
        self.LM = dict()

        self.K1 = None
        self.K2 = None
        self.K3 = None
        self.K4 = None
        self.K5 = None
        self.K6 = None
        self.KF = None
        self.KFt = None
        self.KFz = None
        self.M = None
        self.Tdiag = None
        self.K1_coupling = None


    def create(self, sets, max_frequency, lubricated_coupling=0, PML=0, PML_props=0,
               domain='axisymmetric'):
        """
        Function that generates 1D meshes for axisymmetric elements.
        Parameters:
            sets : list of lists containing properties of each layer in the form
                    [name, start, thickness, material, type of elements,
                    order of elements, number of elements]
                    (two last are optional and can be
                    calculated from the material properties)
            max_frequency : maximum frequencyof interest
            lubricated_coupling : a tuple specifying names of layers which are
                                coupled via normal forces only
            PML : indicate which set is a PML
            PML_props : if a PML is set up, this is a list containing its properties
                        [PML_start, PML_thickness, alpha + 1j*beta]
            domain : domain type - axisymmeric (default) for cylinders,
                                   plate for plates (not working currently)
        Returns:
            glob_nodes, element_sets, element_types, IEN, uncoupled_nodes, uncoupled_dofs
        """
        no_of_elements = []
        vertices, thicknesses, materials = [], [], []

        start_node = 1
        element_number = 1
        # unpack sets
        starting_r = sets[0][1]
        for layer in sets:
            vertices.append(layer[1])
            thicknesses.append(layer[2])
            # format of material list [Re(cL), Re(cT), rho, loss_factor] (solid)
            # [Re(c), rho, loss_factor] (fluid)
            materials.append(layer[3])
            # calculate min wavelength
            if len(layer[3]) == 4:
                min_wavespeed = ((layer[3][1])/layer[3][2])**0.5
                self. min_wavelengths.append(min_wavespeed/max_frequency)
            elif len(layer[3]) == 3:
                min_wavespeed = ((layer[3][0])/layer[3][1])**0.5
                self.min_wavelengths.append(min_wavespeed/max_frequency)

            if len(layer) > 5 and layer[5] != 0:
                element_order = layer[5]
            else:
                element_order = int(np.ceil(np.pi*thicknesses[-1]/
                                            self.min_wavelengths[-1]) + 3) # default
            if len(layer) > 6 and layer[6] != 0:
                no_of_elements.append(layer[6])
            else:
                # how many wavelengths in a layer
                #no_of_elements.append(int(np.ceil(thicknesses[-1]/self.min_wavelengths[-1])))
                no_of_elements.append(1)
            self.element_size.append(thicknesses[-1]/no_of_elements[-1])
            elements_in_set = []
            for element in range(no_of_elements[-1]):
                # calculate the nodes
                if starting_r == 0.0 and domain == 'axisymmetric':
                    nodes, _ = build_GLJ_quadrature(element_order + 1)
                else:
                    nodes, _ = build_GLL_quadrature(element_order + 1)
                # modify them so that they start from zero
                nodes += 1
                nodes *= 0.5
                # raise a flag if this is the last element of a layer involved
                # in lubricated contact
                repeat_node = False
                if lubricated_coupling is not 0:
                    if lubricated_coupling[1] == layer[0] and element == 0:
                        repeat_node = True
                # calculate nodal positions making sure that nodes don't overlap
                if repeat_node:
                    start_node += 1
                node_positions = list(starting_r + nodes*self.element_size[-1])

                node_labels = range(start_node, start_node + len(node_positions))
                start_node += len(node_positions) - 1
                # write global nodes into a dict
                for i, label in enumerate(node_labels):
                    if list(self.glob_nodes.keys()).count(label) == 0:
                        self.glob_nodes.update({label: [node_positions[i]]})
                # update element connectivity dict
                self.IEN.update({element_number: list(node_labels)})
                # specify which elements belong to a certain set
                elements_in_set.append(element_number)
                self.element_types.update({element_number: layer[4]})
                element_number += 1
                starting_r = node_positions[-1]
            # update the dicts
            self.element_sets.update({layer[0]: elements_in_set})
            # update the element properties
            for element in elements_in_set:
                this = dict([('material', materials[-1])])
                self.element_props.update({element: this})

            # construct node sets based on element sets
            self.node_sets.update({layer[0]: [node for element in elements_in_set \
                                              for node in self.IEN[element]]})
            # write PML elements to a separate dict
        if PML is not 0:
            self.element_sets.update({'PML': self.element_sets[PML]})
            self.PML_props = PML_props
            for element in self.element_sets['PML']:
                self.rev_PMLs.update({element: 'PML'})
        if lubricated_coupling is not 0:
            element_1 = self.element_sets[lubricated_coupling[0]][-1]
            element_2 = self.element_sets[lubricated_coupling[1]][0]
            self.uncoupled_nodes.append([self.IEN[element_1][-1], self.IEN[element_2][0]])
            self.uncoupled_dofs.append([1, 2])
        else:
            self.uncoupled_nodes = []
            self.uncoupled_dofs = []
        # for each node check the type of elements that are defined over that node
        element_idxs_sorted = list(self.IEN.keys())
        element_idxs_sorted.sort()
        for element in element_idxs_sorted:
            for node in self.IEN[element]:
                if list(self.nodes_types.keys()).count(node) == 0:
                    self.nodes_types.update({node: []})
                    self.nodes_in_element.update({node: []})
                self.nodes_types[node].append(self.element_types[element])
                self.nodes_in_element[node].append(element)

        # check if node is assigned to elements of different type
        for node, types in self.nodes_types.items():
            if len(types) == 2 and (types[0] != types[1]) and (types[0][0] != types[1][0]):
                self.struct_acoust_pairs.append(self.nodes_in_element[node])
                if 'SL' in types[0] and 'AL' in types[1]:
                    for el in self.nodes_in_element[node]:
                        self.struct_acoust.update({el: 'solid-fluid'})
                elif 'AL' in types[0] and 'SL' in types[1]:
                    for el in self.nodes_in_element[node]:
                        self.struct_acoust.update({el: 'fluid-solid'})
        # sort the lists in case the keys were not pulled in order
        for entry in self.struct_acoust_pairs:
            entry.sort()

    def assemble_matrices(self, n='none'):
        """
        A method that assembles system matrices for a given mesh

        Parameters:
            n : circumferential order

        """
        # initialise a list for the global T transformation matrix
        T_diag = []
        # list of elements
        IENs = list(self.IEN.keys())
        IENs.sort()
        self.n = n

        # go through the list fo elements and create a dictionary with appropriate
        # element objects
        self.elements = dict()
        for element in IENs:
            nodes_at = np.empty(0)
            for node in list(self.IEN[element]):
                nodes_at = np.append(nodes_at, self.glob_nodes[int(node)][0])
            if self.element_types[element] == 'SLAX6' or \
                self.element_types[element] == 'SLAX6_core':
                if 0 in nodes_at:
                    self.elements.update({element: ElementLibrary.SLAX6_core(nodes_at, n)})
                    self.element_types.update({element: 'SLAX6_core'})
                else:
                    self.elements.update({element: ElementLibrary.SLAX6(nodes_at, n)})

            elif self.element_types[element] == 'ALAX6' or \
                 self.element_types[element] == 'ALAX6_core':
                if 0 in nodes_at:
                    self.elements.update({element: ElementLibrary.ALAX6_core(nodes_at, n)})
                    self.element_types.update({element: 'ALAX6_core'})
                else:
                    self.elements.update({element: ElementLibrary.ALAX6(nodes_at, n)})

            elif self.element_types[element] == 'SLAX6_PML':
                self.elements.update({element: ElementLibrary.SLAX6_PML(nodes_at, n,
                                                                        self.PML_props)})
            elif self.element_types[element] == 'ALAX6_PML':
                self.elements.update({element: ElementLibrary.ALAX6_PML(nodes_at, n,
                                                                        self.PML_props)})
            # add properties to each element
            self.elements[element].add_properties(self.element_props[element]['material'])

        solid_dofs = []
        fluid_dofs = []
        current_id = 0

        # create a dictionary of globa degrees of freedom
        # the procedure below is a bit complicated as it allows for specifying
        # decoupling of certain dofs at certain interfaces.
        for element in IENs:
            for idx, node in enumerate(list(self.IEN[element])):
                # check if the particular node has not been processed already
                # or whether this is a structural acoustic coupling node
                if list(self.ID.keys()).count(node) == 0 or \
                    (len(set(self.nodes_types[node])) == 2 and \
                             self.nodes_types[node][0][0] != self.nodes_types[node][1][0]):
                    dofs = self.elements[element].dofs_per_node
                    domains = self.elements[element].dofs_domain
                    # if all nodes are coupled or the element is of different type
                    # than SLAX6 (decoupling works only for SLAX6)
                    if self.uncoupled_nodes == [] or self.element_types[element] != 'SLAX6' or\
                        self.uncoupled_nodes.count == 0:
                        # if this is a structural acoustic coupling, add more dofs
                        # for the element of a different domain
                        if len(set(self.nodes_types[node])) == 2 and \
                                   self.nodes_in_element[node].index(element) == 1:
                            self.ID[int(node)].extend(
                                [current_id + i for i in range(dofs[idx])])
                        # otherwise just add the dofs
                        else:
                            self.ID.update({int(node):
                                            [current_id + i for i in range(dofs[idx])]})
                        # append to a list with T-transformation diagonal
                        T_diag.extend(self.elements[element].T_components[idx])
                        # specify fluid and solid dofs
                        for i, domain in enumerate(domains[idx]):
                            if domain == 's':
                                solid_dofs.append(current_id + i)
                            elif domain == 'f':
                                fluid_dofs.append(current_id + i)
                        current_id += dofs[idx]
                    # if there are uncouple nodes
                    else:
                        # chec whether this is on eof the uncoupled nodes
                        for unc_i, case in enumerate(self.uncoupled_nodes):
                            if node in case:
                                # if this is the first one
                                if case.index(node) == 0:
                                    self.ID.update({int(node):
                                                    [current_id + i for i in \
                                                     range(dofs[idx])]})
                                    # specify fluid and solid dofs
                                    for i, domain in enumerate(domains[idx]):
                                        if domain == 's':
                                            solid_dofs.append(current_id + i)
                                        elif domain == 'f':
                                            fluid_dofs.append(current_id + i)
                                    current_id += dofs[idx]
                                    T_diag.extend(self.elements[element].T_components[idx])
                                # or the second one
                                elif case.index(node) == 1:
                                    if self.uncoupled_dofs[unc_i] == [1, 2]:
                                        previous_r = self.ID[case[0]][0]
                                        local_id = [previous_r, current_id,
                                                    current_id + 1]
                                        self.ID.update({int(node): local_id})
                                        solid_dofs.append(current_id)
                                        solid_dofs.append(current_id + 1)
                                        current_id += 2
                                        T_diag.extend([1j, 1j])
                            # if the current node is not one of the uncoupled
                            # continue as for most elements
                            else:
                                if len(set(self.nodes_types[node])) == 2 and \
                                   self.nodes_in_element[node].index(element) == 1:
                                    self.ID[int(node)].extend(
                                        [current_id + i for i in range(dofs[idx])])
                                else:
                                    self.ID.update({int(node):
                                                    [current_id + i for i in range(dofs[idx])]})
                                T_diag.extend(self.elements[element].T_components[idx])
                                # specify fluid and solid dofs
                                for i, domain in enumerate(domains[idx]):
                                    if domain == 's':
                                        solid_dofs.append(current_id + i)
                                    elif domain == 'f':
                                        fluid_dofs.append(current_id + i)
                                current_id += dofs[idx]
        # total number of dofs
        self.no_of_dofs = current_id
        # total number of nodes
        self.no_of_nodes = len(self.glob_nodes.keys())
        # T-transformation matrix diagonal
        self.Tdiag = T_diag
        self.dof_domains.update({'solid': solid_dofs})
        self.dof_domains.update({'fluid': fluid_dofs})

        # Below a variety of different data structures used for mesh processing are
        # created
        #%%
        # create a dictionary with elements as keys and dofs in that element as values
        dofs_per_el = dict()
        for key, value in self.IEN.items():
            dofs_per_el.update({key: dict()})
            for idx, node_no in enumerate(list(value)):
                dofs_per_el[key].update({node_no: len(self.ID[node_no])})

        #%%

        for key, val in self.IEN.items():
            id_sequence = []
            for i, node in enumerate(list(val)):
                # check if this is a node that couples a fluid to a solid
                if self.nodes_in_element[node] not in self.struct_acoust_pairs:
                    id_sequence.extend(self.ID[node])
                # if this node couples a fluid to a solid, separate the dofs appropriately
                else:
                    if key == self.nodes_in_element[node][0]:
                        id_sequence.extend(self.ID[node][:self.elements[key].dofs_per_node[-1]])
                    else:
                        id_sequence.extend(self.ID[node][self.elements[key - 1].dofs_per_node[-1]:])
            self.LM.update({key: id_sequence})
        #%%
        # miscallaneous: create a dict which maps dofs to the element set
        self.dofs_per_elset = dict()
        for key, val in self.element_sets.items():
            ids = []
            for element in val:
                ids.extend(self.LM[element])
            self.dofs_per_elset.update({key: list(set(ids))})

        #%%
        # global matrix assembly
        # check if there is structural acoustic coupling
        self.is_coupled = len(self.struct_acoust) > 0

        # predefine matrix identificators
        if self.is_coupled:
            mat_idxs = ['k'+str(i) for i in range(1, 7)] + \
                       ['m', 'kfz', 'kft', 'kf', 'k1_coupling']
        else:
            mat_idxs = ['k'+str(i) for i in range(1, 7)] + \
                       ['m', 'kfz', 'kft', 'kf']

        # create matrix-creator dictionaries
        mat_creators = dict()
        for idx in mat_idxs:
            mat_creators.update({idx: dict()})
            mat_creators[idx].update({'rows': np.empty(0)})
            mat_creators[idx].update({'cols': np.empty(0)})
            mat_creators[idx].update({'data': np.empty(0)})

        # evaluate element matrices
        for element in self.LM:
            self.elements[element].calculate_matrices()

        # create structural-acoustic coupling matrices
        # with regard to the order (fluid-soild or solid-fluid)
        for pair in self.struct_acoust_pairs:
            pair.sort()
            if self.element_types[pair[0]] == 'ALAX6' or \
               self.element_types[pair[0]] == 'ALAX6_core' or \
               self.element_types[pair[0]] == 'AL':
                Hf = self.elements[pair[0]].Hf1
                Hs = self.elements[pair[1]].Hs0
                H = -Hs.dot(Hf.T)
                K1_coupling = np.r_[np.c_[np.zeros([H.shape[1]]*2), H.T],
                                    np.c_[H, np.zeros([H.shape[0]]*2)]]
                self.elements[pair[0]].K_1_coupling = K1_coupling

            if self.element_types[pair[0]] == 'SLAX6' or \
               self.element_types[pair[0]] == 'SLAX6_core' or \
               self.element_types[pair[0]] == 'SL' or \
               self.element_types[pair[0]] == 'SL_yz':
                Hs = self.elements[pair[0]].Hs1
                Hf = self.elements[pair[1]].Hf0
                H = Hs.dot(Hf.T)
                K1_coupling = np.r_[np.c_[np.zeros([H.shape[0]]*2), H],
                                    np.c_[H.T, np.zeros([H.shape[1]]*2)]]
                self.elements[pair[0]].K_1_coupling = K1_coupling

        # start the assembly process
        # populate matrix creators with respective entries from element matrices.
        for element in self.LM:
            for idx in mat_idxs:
                if idx == 'k1':
                    matrix = self.elements[element].K_1
                elif idx == 'k2':
                    matrix = self.elements[element].K_2
                elif idx == 'k3':
                    matrix = self.elements[element].K_3
                elif idx == 'k4':
                    matrix = self.elements[element].K_4
                elif idx == 'k5':
                    matrix = self.elements[element].K_5
                elif idx == 'k6':
                    matrix = self.elements[element].K_6
                elif idx == 'kf':
                    matrix = self.elements[element].K_F
                elif idx == 'kfz':
                    matrix = self.elements[element].K_Fz
                elif idx == 'kft':
                    matrix = self.elements[element].K_Ft
                elif idx == 'm':
                    matrix = self.elements[element].M
                elif idx == 'k1_coupling':
                    if len(self.struct_acoust_pairs) > 0 and \
                        element in self.struct_acoust:
                        for pair in self.struct_acoust_pairs:
                            if pair[0] == element:
                                matrix = self.elements[element].K_1_coupling
                            else:
                                continue
                    else:
                        continue

                # check which entries are non-zero
                non_zero = np.where(matrix != 0)

                # update matrix creators
                if idx == 'k1_coupling' and self.is_coupled and \
                    element in self.struct_acoust:
                    for pair in self.struct_acoust_pairs:
                        if pair[0] == element:
                            lm = self.LM[element] + self.LM[element + 1]
                            mat_creators[idx].update({'rows': np.append(mat_creators[idx]['rows'],
                                            [lm[i] for i in non_zero[0]])})
                            mat_creators[idx].update({'cols': np.append(mat_creators[idx]['cols'],
                                            [lm[i] for i in non_zero[1]])})
                            mat_creators[idx].update({'data': np.append(mat_creators[idx]['data'],
                                                matrix[non_zero])})
                else:
                    mat_creators[idx].update({'rows': np.append(mat_creators[idx]['rows'],
                                                [self.LM[element][i] for i in non_zero[0]])})
                    mat_creators[idx].update({'cols': np.append(mat_creators[idx]['cols'],
                                                [self.LM[element][i] for i in non_zero[1]])})
                    mat_creators[idx].update({'data': np.append(mat_creators[idx]['data'],
                                                matrix[non_zero])})

        # store the global matrices in a sparse format
        self.K1 = sps.coo_matrix((mat_creators['k1']['data'],
                                  (mat_creators['k1']['rows'],
                                   mat_creators['k1']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.K2 = sps.coo_matrix((mat_creators['k2']['data'],
                                  (mat_creators['k2']['rows'],
                                   mat_creators['k2']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.K3 = sps.coo_matrix((mat_creators['k3']['data'],
                                  (mat_creators['k3']['rows'],
                                   mat_creators['k3']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.K4 = sps.coo_matrix((mat_creators['k4']['data'],
                                  (mat_creators['k4']['rows'],
                                   mat_creators['k4']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.K5 = sps.coo_matrix((mat_creators['k5']['data'],
                                  (mat_creators['k5']['rows'],
                                   mat_creators['k5']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.K6 = sps.coo_matrix((mat_creators['k6']['data'],
                                  (mat_creators['k6']['rows'],
                                   mat_creators['k6']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.KF = sps.coo_matrix((mat_creators['kf']['data'],
                                  (mat_creators['kf']['rows'],
                                   mat_creators['kf']['cols'])),
                                 shape=(self.no_of_dofs, self.no_of_dofs))
        self.KFz = sps.coo_matrix((mat_creators['kfz']['data'],
                                   (mat_creators['kfz']['rows'],
                                    mat_creators['kfz']['cols'])),
                                  shape=(self.no_of_dofs, self.no_of_dofs))
        self.KFt = sps.coo_matrix((mat_creators['kft']['data'],
                                   (mat_creators['kft']['rows'],
                                   mat_creators['kft']['cols'])),
                                  shape=(self.no_of_dofs, self.no_of_dofs))
        self.M = sps.coo_matrix((mat_creators['m']['data'],
                                 (mat_creators['m']['rows'],
                                   mat_creators['m']['cols'])),
                                shape=(self.no_of_dofs, self.no_of_dofs))
        if self.is_coupled:
            self.K1_coupling = sps.coo_matrix((mat_creators['k1_coupling']['data'],
                                               (mat_creators['k1_coupling']['rows'],
                                                mat_creators['k1_coupling']['cols'])),
                                              shape=(self.no_of_dofs, self.no_of_dofs))

    def assemble_subset(self, elements):
        """
        This method assembles system matrices for a defined set of elements
        (submatrices of the global matrices)

        Parameters:
            element : list of elements for which the global submatrices are to be
                        evaluated
        Returns:
            matrices_subset : a dictionary with respective global submatrices. Note
                                that these have dimensions of the full global matrices
                                but the matrix elements not corresponding to the chosen
                                set of SAFE elements are zero
        """
        # matrix assembly
        # predefine matrix identificators

        if self.is_coupled:
            mat_idxs = ['k'+str(i) for i in range(1, 7)] + \
                       ['m', 'kfz', 'kft', 'kf', 'k1_coupling']
        else:
            mat_idxs = ['k'+str(i) for i in range(1, 7)] + \
                       ['m', 'kfz', 'kft', 'kf']

        # create matrix-creator dictionaries
        mat_creators = dict()
        for idx in mat_idxs:
            mat_creators.update({idx: dict()})
            mat_creators[idx].update({'rows': np.empty(0)})
            mat_creators[idx].update({'cols': np.empty(0)})
            mat_creators[idx].update({'data': np.empty(0)})
        # create matrix-creator dictionaries
        for element in elements:
            for idx in mat_idxs:
                if idx == 'k1':
                    matrix = self.elements[element].K_1
                elif idx == 'k2':
                    matrix = self.elements[element].K_2
                elif idx == 'k3':
                    matrix = self.elements[element].K_3
                elif idx == 'k4':
                    matrix = self.elements[element].K_4
                elif idx == 'k5':
                    matrix = self.elements[element].K_5
                elif idx == 'k6':
                    matrix = self.elements[element].K_6
                elif idx == 'kf':
                    matrix = self.elements[element].K_F
                elif idx == 'kfz':
                    matrix = self.elements[element].K_Fz
                elif idx == 'kft':
                    matrix = self.elements[element].K_Ft
                elif idx == 'm':
                    matrix = self.elements[element].M
                elif idx == 'k1_coupling':
                    if len(self.struct_acoust_pairs) > 0 and \
                        element in self.struct_acoust:
                        for pair in self.struct_acoust_pairs:
                            if pair[0] == element:
                                matrix = self.elements[element].K_1_coupling
                            else:
                                continue
                    else:
                        continue
                # check which entries are non-zero
                non_zero = np.where(matrix != 0)
                # check which entries are non-zero
                if idx == 'k1_coupling' and self.is_coupled and \
                    element in self.struct_acoust:
                    for pair in self.struct_acoust_pairs:
                        if pair[0] == element:
                            lm = self.LM[element] + self.LM[element + 1]
                            mat_creators[idx].update({'rows': np.append(mat_creators[idx]['rows'],
                                            [lm[i] for i in non_zero[0]])})
                            mat_creators[idx].update({'cols': np.append(mat_creators[idx]['cols'],
                                            [lm[i] for i in non_zero[1]])})
                            mat_creators[idx].update({'data': np.append(mat_creators[idx]['data'],
                                                matrix[non_zero])})
                else:
                    mat_creators[idx].update({'rows': np.append(mat_creators[idx]['rows'],
                                                [self.LM[element][i] for i in non_zero[0]])})
                    mat_creators[idx].update({'cols': np.append(mat_creators[idx]['cols'],
                                                [self.LM[element][i] for i in non_zero[1]])})
                    mat_creators[idx].update({'data': np.append(mat_creators[idx]['data'],
                                                matrix[non_zero])})

        # check which entries are non-zero
        matrices_subset = dict()
        matrices_subset.update({'K1': sps.coo_matrix((mat_creators['k1']['data'],
                                                      (mat_creators['k1']['rows'],
                                                       mat_creators['k1']['cols'])),
                                                     shape=(self.no_of_dofs,
                                                             self.no_of_dofs))})
        matrices_subset.update({'K2': sps.coo_matrix((mat_creators['k2']['data'],
                                  (mat_creators['k2']['rows'],
                                   mat_creators['k2']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'K3': sps.coo_matrix((mat_creators['k3']['data'],
                                  (mat_creators['k3']['rows'],
                                   mat_creators['k3']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'K4': sps.coo_matrix((mat_creators['k4']['data'],
                                  (mat_creators['k4']['rows'],
                                   mat_creators['k4']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'K5': sps.coo_matrix((mat_creators['k5']['data'],
                                  (mat_creators['k5']['rows'],
                                   mat_creators['k5']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'K6': sps.coo_matrix((mat_creators['k6']['data'],
                                  (mat_creators['k6']['rows'],
                                   mat_creators['k6']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'KF': sps.coo_matrix((mat_creators['kf']['data'],
                                  (mat_creators['kf']['rows'],
                                   mat_creators['kf']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'KFz': sps.coo_matrix((mat_creators['kfz']['data'],
                                  (mat_creators['kfz']['rows'],
                                   mat_creators['kfz']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'KFt': sps.coo_matrix((mat_creators['kft']['data'],
                                  (mat_creators['kft']['rows'],
                                   mat_creators['kft']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        matrices_subset.update({'M': sps.coo_matrix((mat_creators['m']['data'],
                                  (mat_creators['m']['rows'],
                                   mat_creators['m']['cols'])),
                                   shape=(self.no_of_dofs, self.no_of_dofs))})
        if self.is_coupled:
            matrices_subset.update({'K1_coupling': sps.coo_matrix((mat_creators['k1_coupling']['data'],
                                      (mat_creators['k1_coupling']['rows'],
                                       mat_creators['k1_coupling']['cols'])),
                                       shape=(self.no_of_dofs, self.no_of_dofs))})
        return matrices_subset

    def plot_mesh(self):
        """
        Plots the mesh (node locations and elements)
        """
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111)
        for entry in self.glob_nodes:
            ax.plot(0, self.glob_nodes[entry][0], 'o',
                    markersize=10, markerfacecolor='red')
            ax.annotate(str(entry), color='red',
                        xy=(0, self.glob_nodes[entry][0]),
                        textcoords='offset points', xytext=(12, 0))
        for element, nodes in self.IEN.items():
            ax.annotate(str(element), color='blue',
                        xy=(0, self.glob_nodes[nodes[1]][0]),
                        textcoords='offset points',
                        xytext=(-30, 0))
        plt.show()
