from fenics import *
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import csv
from ufl import nabla_div
import math
import os

srlife = True

n_days = 50      # simulate for 10 days
temp_prof_hours = 12    # temperature for 12 hors are given, repeat it for 2*n_days times

###################################################################
# fit the material properties with temperature curve with polynomial
###################################################################
# print('define material constant')

tol= 1e-8   # tolerance 

nu = Constant(0.31)                                            # Poisson ratio

T_ref = Constant(300.0)                                       # reference temp, at T_ref, zeo stress state

rho = Constant(0.00000805)                                      # Desnity unit of kg/mm^3
f = Constant((0,-9.8*1000.0*rho,0))                                # gravity mm/s*2
K = 0.02                                            # stiffness of the foundation at back surface 0.02Mpa/mm

# refer to # reference: srlife: A Fast Tool for High Temperature Receiver
# Design and Analysis, page 54
cutoff_strain_lower_700 = 2.0/1000
cut_off_strain_higher_700 = 1.0/1000               # cutoff strain for fatigue prediction

# print('fit youngs modulus and thermal expansion coefficient with temperature curve with polynomial')

T_raw = np.array([20,100,200,300,400,500,600,700,800,900])+273.15 # in K

# reference: Continuation Report: Creep-fatigue Behavior and Damage Accumulation of a Candidate Structural Material 
# for Concentrating Solar Thermal Receiver
E_raw = np.array([221,218,212,206,200,193,186,178,169,160])*10**3       # the unit of geometry is in mm, E should be in N/mm^2=1 Mpa, 

# reference:Design Guidance for High Temperature Concentrating Solar Power Components
# alpha_raw = np.array([12.38,12.38,13.55,14.32,15.12,15.55,16.0,17.68,20.39,16.51])*10**(-6)  # instantaneous CTE
alpha_raw = np.array([12.38,12.38,13.55,14.32,15.12,15.55,16.0,17.68,20.39,16.51])*10**(-6)  # instantaneous CTE

# # creep strain rate coefficient, refer to # reference: srlife: A Fast Tool for High Temperature Receiver
# # Design and Analysis, page 45
# epsilon_dot_0 = 1.0e13 # unit is /hr
# b = 2.53e-7 # unit is mm
# A = -9.6295
# B = -0.147
# K_bz = 1.38e-20 #in unit of N*mm/K = J*1000/K

# # creep strain rate coefficient, refer to # reference: srlife: A Fast Tool for High Temperature Receiver
# # Design and Analysis, in github
# epsilon_dot_0 = 1.0e13 # unit is /hr
# b = 2.53e-7 # unit is mm
# A = -0.103848
# B = -0.015267
# K_bz = 1.38e-20 #in unit of N*mm/K = J*1000/K

# creep strain rate coefficient, refer to # reference: Design Guidance for High Temperature Concentrating 
# Solar Power Components, page 27
epsilon_dot_0 = 1.19e10 # unit is /hr
b = 2.53e-7 # unit is mm
A = -10.98557
B = -0.53098
K_bz = 1.38064e-20 #in unit of N*mm/K = J*1000/K

# reference: CREEP AND CREEP-FATIGUE DEFORMATION AND LIFE ASSESSMENT OF NIBASED ALLOY 740H AND ALLOY 617
QQ = 523000.0
R = 8.314

# Life Estimation of Pressurized-Air Solar-Thermal Receiver Tubes
A_creep = (3.4*10**31)*3600 # /h
m = 5.34
R_gasconstant = 8.314 # J/mol/K
U_creep = 450000.0 #J/mol

#########################################
# import mesh and marked facet from gmesh
#########################################

# print('import mesh and marked facet from gmesh')

mesh = Mesh('mesh_1mm.xml')
facet_func = MeshFunction('size_t',mesh,'mesh_1mm_facet_region.xml') # top surface is marked as 90, back surface is marked as 89.

# print(np.max(coordinates[:, 1]), np.min(coordinates[:,1]))
# print(np.max(coordinates[:, 2]), np.min(coordinates[:,2]))
# exit()

class bottom_surface(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1])<tol

bottom_surface = bottom_surface()

bottom_surface.mark(facet_func, 3)  # mark bottom surface as 3

ds = Measure('ds', domain = mesh, subdomain_data = facet_func)

####################################
# Define function space and functions
####################################

# print('Define function space and functions')

Q = FunctionSpace(mesh,'P', 1)            # function space for temperature first order
V = VectorFunctionSpace(mesh, 'P', 2)     # function space for displacement
W = TensorFunctionSpace(mesh, 'P', 1)     # function for strain and stress tensor

v2d = vertex_to_dof_map(Q)                # ith component is the ID of DOF of vertex whose vertex ID is i.
coordinates = mesh.coordinates()          # array, the ith row is the coordinates of vertex whose ID is i.# print(np.max(coordinates[:, 0]), np.min(coordinates[:,0]))

u = TrialFunction(V)                       # trial function
v = TestFunction(V)                        # test function

T = Function(Q)                            # function for temperature
E = Function(Q)
CTE = Function(Q)
inte_ICTE = Function(Q)  # integral of ICTE to temperature

sigma_tensor = Function(W)
epsilon_tensor = Function(W)
epsilon_m_tensor = Function(W)
von_mises_stress = Function(Q)       # von mises stress
von_mises_strain = Function(Q)       # von mises strain
von_mises_creep_strain = Function(Q)
modified_von_mises = Function(Q)     # modified equivalent strain
von_mises_delta_strain = Function(Q)
max_vm_delta_strain = Function(Q)

used_for_test = Function(Q)

D_c = Function(Q)    # creep damage factor
D_f = Function(Q) # fatigue damage facor

delta_epsilon_CR = Function(W)        # change of creep strain tensor
epsilon_CR = Function(W)              # total creep strain tensor
delta_strain = Function(W)           # epsilon at time j minus epsilon at time k, for fatigue calculation
CR_strain_rate = Function(Q)
a11 = Function(Q)
a22 = Function(Q)

n = FacetNormal(mesh)                            # normal vector


# interpolate the ICTE and E:
ICTE_refT = alpha_raw[0] + (alpha_raw[1]-alpha_raw[0])/(T_raw[1]-T_raw[0])*(T_ref-T_raw[0]) # value of ICTE at reference temp


###########################
# assign boundary conditions
###########################

# print('assign boundary conditions')

bc_topsurface1 = DirichletBC(V.sub(1), Constant(0), facet_func, 90)  # uy = 0 at the top surface
# bc_topsurface2 = DirichletBC(V.sub(2), Constant(0), facet_func, 2)  # uz = 0 at the top surface
bc_backsurface = DirichletBC(V.sub(0), Constant(0), facet_func, 89) # ux = 0 at the back surface
bc_bottom_surface = DirichletBC(V.sub(1), Constant(-63), facet_func, 3) # uy = 50 at the bottom surface

tol = 1e-8

def front_line(x, on_boundary):
    return on_boundary and (abs(x[0]+280.8262)<tol) and (abs(x[2]+40.386144)<tol)
def fixed_point1(x, on_boundary):                                  # point A
    tol = 10e-8
    return (abs(x[0]-0)<tol) and (abs(x[2]+80.80883)<tol) and (abs(x[1]-6500)<tol) 
def fixed_point2(x, on_boundary):                                  # point G
    tol = 10e-8
    return (abs(x[0])<tol) and (abs(x[2])<tol) and (abs(x[1]-6500)<tol)
def fixed_point3(x, on_boundary):                                  # point D
    tol = 10e-8
    return (abs(x[0]+280.8262)<tol) and (abs(x[2]+40.386144)<tol) and (abs(x[1]-6500)<tol)
def fixed_point4(x, on_boundary):                                  # point A0
    tol = 10e-8
    return (abs(x[0]-0)<tol) and (abs(x[2]+80.80883)<tol) and (abs(x[1]-0)<tol) 
def fixed_point5(x, on_boundary):                                  # point G0
    tol = 10e-8
    return (abs(x[0])<tol) and (abs(x[2])<tol) and (abs(x[1]-0)<tol)
def fixed_point6(x, on_boundary):                                  # point D0
    tol = 10e-8
    return (abs(x[0]+280.8262)<tol) and (abs(x[2]+40.386144)<tol) and (abs(x[1]-0)<tol)

bc_point1 = DirichletBC(V.sub(2), Constant(0), fixed_point1, method='pointwise')
bc_point2 = DirichletBC(V.sub(2), Constant(0), fixed_point2, method='pointwise')
bc_point3 = DirichletBC(V.sub(2), Constant(0), fixed_point3, method='pointwise')

bc_point4 = DirichletBC(V.sub(2), Constant(0), fixed_point4, method='pointwise')
bc_point5 = DirichletBC(V.sub(2), Constant(0), fixed_point5, method='pointwise')
bc_point6 = DirichletBC(V.sub(2), Constant(0), fixed_point6, method='pointwise')

# bc_front_line = DirichletBC(V.sub(2), Constant(0), front_line, method = 'pointwise')

bc_u = [bc_topsurface1, bc_point1, bc_point4]

##################################
# Define strain and stress tensors
##################################

# print('Define strain and stress tensors')

def epsilon(u):    # define strain tensor, total strain
    return sym(nabla_grad(u))

def epsilon_T(inte_ICTE):   # thermal strain, integral of instantenuous CTE
    return inte_ICTE*Identity(3) #(alpha_T_fit[0]*T+alpha_T_fit[1])*(T-T_ref)*Identity(3)#

def epsilon_m(u, inte_ICTE, epsilon_CR):   # strain which cause stress, total strain minus thermal strain minus creep strain
    return epsilon(u)-epsilon_T(inte_ICTE)-epsilon_CR

def sigma_with_thermal(u, inte_ICTE, epsilon_CR):     # define stress tensot
    return lambda_*tr(epsilon_m(u, inte_ICTE, epsilon_CR))*Identity(3)+2*mu*epsilon_m(u, inte_ICTE, epsilon_CR)     # use cte(T)dT
    # return lambda_*tr(epsilon(u)-(T-T_ref)*CTE*Identity(3))*Identity(3)+2*mu*(epsilon(u)-(T-T_ref)*CTE*Identity(3)) # use CTE*deltaT
def sigma_without_thermal(u, epsilon_CR):     # define stress tensot
    return lambda_*tr(epsilon(u)-epsilon_CR)*Identity(3)+2*mu*(epsilon(u)-epsilon_CR)
########################################################################
#Read the temperature file and convert it to be an array from txt file
########################################################################

# print('Read the temperature file and convert it to be an array from txt file')

max_von_mises = []
max_modified_vonmises = []
max_equi_strain = []
max_equi_creep_strain = []
max_creep_damage_factor = []
max_fatigue_damage_factor = []

###########################
# create file to save data
###########################
save_dir_name = "results_"+str(n_days)+"_days_with_srlife_VM_backward_740H"
os.makedirs(save_dir_name, exist_ok=True)

File(save_dir_name+"/mesh.pvd") << mesh  # save mesh
File(save_dir_name+'/meshfunc_f.pvd') << facet_func               # import mesh function, surface where y=6.5m is marked as 2, surfaces at x=0 are marked as 1

xdmffile_E = XDMFFile(f"{save_dir_name}/E.xdmf")
xdmffile_CTE = XDMFFile(f"{save_dir_name}/CTE.xdmf")
xdmffile_T = XDMFFile(f"{save_dir_name}/T.xdmf")
xdmffile_u = XDMFFile(f"{save_dir_name}/u.xdmf")
xdmffile_sigma_tensor = XDMFFile(f"{save_dir_name}/signa_tensor.xdmf")
xdmffile_epsilon_tensor = XDMFFile(f"{save_dir_name}/epsilon_tensor.xdmf")
xdmffile_epsilon_m_tensor = XDMFFile(f"{save_dir_name}/epsilon_m_tensor.xdmf")
xdmffile_epsilon_CR = XDMFFile(f"{save_dir_name}/epsilon_CR.xdmf")
xdmffile_von_mises_stress = XDMFFile(f"{save_dir_name}/von_mises_stress.xdmf")
xdmffile_von_mises_strain = XDMFFile(f"{save_dir_name}/von_mises_strain.xdmf")
xdmffile_von_mises_creep_strain = XDMFFile(f"{save_dir_name}/von_mises_creep_strain.xdmf")
xdmffile_modified_von_mises = XDMFFile(f"{save_dir_name}/modified_von_mises.xdmf")
xdmffile_modified_D_c = XDMFFile(f"{save_dir_name}/D_c.xdmf")
xdmffile_modified_D_f = XDMFFile(f"{save_dir_name}/D_f.xdmf")

T_value_vector_list = []
E_value_vector_list = []
CTE_value_vector_list = []
inte_ICTE_value_vector_list = []

for i in range(temp_prof_hours):
        

    print('interpolate the temperature field', i, temp_prof_hours)
    file_name = 'real_temp_hours_' + str(i) + '.txt'
    readin_file = np.loadtxt(file_name)
    imported_coor = readin_file[:,:3]
    # make the coordinate system be the same between temp profile and mesh
    imported_coor[:, 0] = imported_coor[:, 0]-np.max(imported_coor[:, 0])
    imported_coor[:, 1] = imported_coor[:, 1]-np.min(imported_coor[:, 1])
    imported_coor[:, 2] = imported_coor[:, 2]-np.max(imported_coor[:, 2])

    temp_read = readin_file[:,3]    # in K

    ######################################################################
    # assign temperature field and get material properties at each location
    ######################################################################


    for i_coor in range(np.shape(coordinates)[0]):
        inte_ICTE_seg = 0
        value=griddata(imported_coor,temp_read,coordinates[i_coor]/1000, method = 'nearest')[0] # evaluate the temperature at vertex whole vertex id is i
        T.vector()[v2d[i_coor]] = value                           # assign value for temperature function
        # T_raw = np.array([20,100,200,300,400,500,600,700,800,900])+273.15
        
        for i_p in range(np.shape(E_raw)[0]-1):
            
            if value > T_raw[i_p] and value <= T_raw[i_p+1]:

                E.vector()[v2d[i_coor]] = (E_raw[i_p] + (E_raw[1+i_p]-E_raw[i_p])/(T_raw[i_p+1]-T_raw[i_p])*(value-T_raw[i_p]))
                CTEvalue = (alpha_raw[i_p] + (alpha_raw[1+i_p]-alpha_raw[i_p])/(T_raw[i_p+1]-T_raw[i_p])*(value-T_raw[i_p]))
                CTE.vector()[v2d[i_coor]] = CTEvalue

                if i_p == 0:
                    inte_ICTE.vector()[v2d[i_coor]] = ((CTEvalue+ICTE_refT)*(value-T_ref)/2)
                    break
                else:
                    int_CTE_seg = (alpha_raw[1]+ICTE_refT)*(T_raw[1]-T_ref)/2.0
                    for i_pp in range(1,i_p):
                        int_CTE_seg += (alpha_raw[i_pp]+alpha_raw[i_pp+1])*(T_raw[i_pp+1]-T_raw[i_pp])/2.0
                    inte_ICTE.vector()[v2d[i_coor]] = (int_CTE_seg+(alpha_raw[i_p]+CTEvalue)*(value-T_raw[i_p])/2.0)

                    break
    T_value_vector_list.append(T.vector()[:])
    E_value_vector_list.append(E.vector()[:])
    CTE_value_vector_list.append(CTE.vector()[:])
    inte_ICTE_value_vector_list.append(inte_ICTE.vector()[:])

time_list = []

u_old = Function(V)
u_n = Function(V)

for ii in range(int(n_days*2)):
    # fatigue is calculated for each period
    total_strain_time_list = [] # save strain tensor function and used for fatigue analysis
    for i in range(temp_prof_hours):
        
        tt = ii*temp_prof_hours + i
        time_list.append(tt)
        print(tt, n_days*2*temp_prof_hours)

        ######################################################################
        # assign temperature field and get material properties at each location
        ######################################################################

        T.vector()[:] = T_value_vector_list[i]
        E.vector()[:] = E_value_vector_list[i]
        CTE.vector()[:] = CTE_value_vector_list[i]
        inte_ICTE.vector()[:] = inte_ICTE_value_vector_list[i]

        
        vtkfile1 = File(save_dir_name+'/E.pvd')
        vtkfile1<<E

        vtkfile1 = File(save_dir_name+'/CTE.pvd')
        vtkfile1<<CTE

        vtkfile1 = File(save_dir_name+'/inte_CTE.pvd')
        vtkfile1<<inte_ICTE

        lambda_=E*nu/(1+nu)/(1-2*nu)    # Lame constants
        mu = E/2/(1+nu)

        ###############################
        # Define the variational form
        ###############################

        # print('Define the variational form')
        F = inner(sigma_with_thermal(u, inte_ICTE, epsilon_CR), nabla_grad(v))*dx + K*dot(u, n)*dot(n, v)*ds(89) #- dot(f,v)*dx
      
        a = lhs(F)
        L = rhs(F)

        ############################ 
        # compute displacement
        ############################
        change_of_disp = 1.0

        # Newton interation: we want to use the solved stress to predict the creep strain rate. when the steress
        # is solved, use this solved stress to calculate the creep strain rate, if the change of displacement
        # is small enough, we assume the Newton interation converges.

        while change_of_disp > 1.0e-6:
            print("change_of_disp:", change_of_disp)
            solve(a==L, u_n, bc_u)

            ####################
            # post process
            ###################

            # print('post processing')
            sigma_tensor = sigma_with_thermal(u_n, inte_ICTE, epsilon_CR)       # stress tensor
            s = sigma_with_thermal(u_n, inte_ICTE, epsilon_CR) - (1./3)*tr(sigma_with_thermal(u_n, inte_ICTE, epsilon_CR))*Identity(3)

            sigma_tensor = project(sigma_tensor, W)        # unit is MPA  

            epsilon_tensor = epsilon(u_n)   # strain tensor
            epsilon_tensor = project(epsilon_tensor, W)   # strain function

            epsilon_m_tensor = epsilon_m(u_n, inte_ICTE, epsilon_CR)   # strain tensor
            epsilon_m_tensor = project(epsilon_m_tensor, W)   # strain function

            von_mises_stress = sqrt(3./2*inner(s, s))
            von_mises_stress = project(von_mises_stress, Q)

            aa = von_mises_stress.vector()[:]
            aa[aa<=0] = 1e-10
            von_mises_stress.vector()[:] = aa


            e = epsilon(u_n) - (1./3)*tr(epsilon(u_n))*Identity(3)
            von_mises_strain = sqrt(3./2*inner(e, e))
            von_mises_strain = project(von_mises_strain, Q)

            epsilon_et = von_mises_strain/(1+nu)
            
            
            if srlife:
                # reference: srlife: A Fast Tool for High Temperature Receiver, page 45
                delta_epsilon_CR_rate = epsilon_dot_0*exp(B*E/2/(1+nu)*b**3/A/K_bz/T)*((von_mises_stress*2*(1+nu)/E)**(-E/2/(1+nu)*b**3/A/K_bz/T))*s/von_mises_stress # creep strain rate
                # reference: Creep and Creep-Fatigue Deformation and Life Assessment of Ni-Based Alloy 740H and Alloy 617, page 3
                # delta_epsilon_CR_rate = 9.58e17*exp(4414.5*von_mises_stress/E)*exp(-QQ/R/T)*s/von_mises_stress # creep strain rate
            
                if tt==0:
                    delta_epsilon_CR.vector()[:] = 0
                    change_of_disp = 0.0
                    
                else:
                    delta_epsilon_CR  = delta_epsilon_CR_rate*1.0
                    delta_epsilon_CR = project(delta_epsilon_CR, W)
                    change_of_disp = np.linalg.norm(u_old.vector()[:]-u_n.vector()[:], 2)
            else:

                # reference: 
                # delta_epsilon_CR_rate = C1*exp(von_mises_stress/C2)*exp(-C3/(T-273.15))  # creep strain rate
            
                # # reference: CREEP AND CREEP-FATIGUE DEFORMATION AND LIFE ASSESSMENT OF NIBASED ALLOY 740H AND ALLOY 617
                # delta_epsilon_CR_rate = 9.58e17*exp(4414.5*von_mises_stress/E)*exp(-QQ/R/T)

                # reference Life Estimation of Pressurized-Air Solar-Thermal Receiver Tubes
                # delta_epsilon_CR_rate = A_creep*((von_mises_stress/E)**m)*exp(-U_creep/R_gasconstant/T)   # unit: /h
                delta_epsilon_CR_rate = epsilon_dot_0*exp(B*E/2/(1+nu)*b**3/A/K_bz/T)*((von_mises_stress*2*(1+nu)/E)**(-E/2/(1+nu)*b**3/A/K_bz/T))
                CR_strain_rate = project(delta_epsilon_CR_rate, Q)

                A1 = Constant([[1,0,0],[0,0,0],[0,0,0]])
                A2 = Constant([[0,0,0],[0,1,0],[0,0,0]])
                A3 = Constant([[0,0,0],[0,0,0],[0,0,1]])
                
                # change of creep strain in each hour's period
                if tt==0:
                    delta_epsilon_CR.vector()[:] = 0
                    change_of_disp = 0.0
                else:
                    delta_epsilon_CR = CR_strain_rate * 1.0/epsilon_et/2.0/(1+nu)*(6*epsilon(u_n)-tr(epsilon(u_n))*Identity(3)-dot(dot(A1,3*epsilon(u_n)), A1)-dot(dot(A2,3*epsilon(u_n)), A2)-dot(dot(A3,3*epsilon(u_n)), A3))
                    delta_epsilon_CR = project(delta_epsilon_CR, W)
                    change_of_disp = np.linalg.norm(u_old.vector()[:]-u_n.vector()[:], 2)

            epsilon_CR += delta_epsilon_CR
            epsilon_CR = project(epsilon_CR, W)

            u_old.vector()[:] = u_n.vector()[:]
        
        max_von_mises.append(np.max(von_mises_stress.vector()[:]))
        max_equi_strain.append(np.max(von_mises_strain.vector()[:]))

        # total creep strain
        
        e_cr = epsilon_CR - (1./3)*tr(epsilon_CR)*Identity(3)
        von_mises_creep_strain = sqrt(3./2*inner(e_cr, e_cr))
        von_mises_creep_strain = project(von_mises_creep_strain, Q)
        max_equi_creep_strain.append(np.max(von_mises_creep_strain.vector()[:]))    
        # print('max von mises cr strain', max_equi_creep_strain)
        # creep damage factor
        # invariants
        J1 = tr(sigma_tensor)
        J2 = sigma_tensor[0,0]*sigma_tensor[1,1] + sigma_tensor[2,2]*sigma_tensor[1,1]+sigma_tensor[0,0]*sigma_tensor[2,2] - pow(sigma_tensor[0,1],2) - pow(sigma_tensor[0,2],2) -pow(sigma_tensor[1,2],2)

        S = pow(pow(J1, 2) - 2*J2, 0.5)

        modified_von_mises = von_mises_stress/0.9*exp(0.24*J1/S-1)
        modified_von_mises = project(modified_von_mises, Q)

        # from project, vonmises stress could be negative, convert negative von mises stress to small positive value which makes the rupture time be high
        aa = modified_von_mises.vector()[:]
        aa[aa<=0] = 1e-10
        modified_von_mises.vector()[:] = aa

        # rupture time, refer to srlife: A Fast Tool for High Temperature Receiver
        # Design and Analysis, page 48
        t_rupture = 10**((36280.36575 - 5884.39274*np.log10(von_mises_stress.vector()[:]))/T.vector()[:] - 18.29014549) #t_rupture is in h

        index_nan_t = np.where(np.isnan(t_rupture))[0]
        index_not_nan_t = np.where(~np.isnan(t_rupture))[0]
        
        D_c.vector()[index_not_nan_t] += 1.0/t_rupture[index_not_nan_t]

        D_c.vector()[index_nan_t] += 0.0

        max_creep_damage_factor.append(np.max(D_c.vector()[:]))

        # save data
        xdmffile_u.write(u_n, tt)
        xdmffile_T.write(T, tt)
        xdmffile_E.write(E, tt)
        xdmffile_CTE.write(CTE, tt)

        xdmffile_sigma_tensor.write(sigma_tensor, tt)
        xdmffile_epsilon_tensor.write(epsilon_tensor, tt)
        xdmffile_epsilon_m_tensor.write(epsilon_m_tensor, tt)
        xdmffile_epsilon_CR.write(epsilon_CR, tt)
        xdmffile_von_mises_stress.write(von_mises_stress, tt)
        xdmffile_von_mises_strain.write(von_mises_strain, tt)
        xdmffile_von_mises_creep_strain.write(von_mises_creep_strain, tt)
        xdmffile_modified_von_mises.write(modified_von_mises, tt)
        xdmffile_modified_D_c.write(D_c, tt)
        
        # fatigue damage:

        # Step 1. Divide the cycle period into time points.
        # Step 2. Select any time point as a reference time point, labeled as point 𝑜.
        # Step 3. For a time point 𝑖 in the cycle, subtract the strains at time point 𝑜 from the strains at
        # time point 𝑖.
        # Step 4. Use these strain differences to determine an equivalent strain range ∆𝜀(𝑒𝑞𝑢𝑖𝑣,𝑖) using
        # Step 5. Define ∆𝜀𝑚𝑎𝑥 as the maximum value of the above calculated equivalent strain ranges
        # for all time points associated with a reference point 𝑜, and for all possible reference time point
        # within the cycle.
        # reference: Identifying Limitations of ASME Section III Division 5
        # For Advanced SMR Designs
        
        # save total strain to list and used for fatigue analysis
        total_strain_time_list.append(epsilon_tensor.vector()[:])
        pre_vm_delta_strain = np.zeros((np.shape(von_mises_strain.vector()[:])[0], ))
    for j in range(temp_prof_hours):
        for k in range(temp_prof_hours):
            epsilon_0 = total_strain_time_list[k]
            delta_strain.vector()[:] = np.asarray(total_strain_time_list[j])-np.asarray(epsilon_0)
            e_delta_strain = delta_strain - (1./3)*tr(delta_strain)*Identity(3)
            von_mises_delta_strain = sqrt(3./2*inner(e_delta_strain, e_delta_strain))/1.5
            von_mises_delta_strain = project(von_mises_delta_strain, Q)
            current_vm_delta_strain = von_mises_delta_strain.vector()[:]
            pre_vm_delta_strain = np.maximum(pre_vm_delta_strain, current_vm_delta_strain)
    max_vm_delta_strain.vector()[:] = pre_vm_delta_strain     # strain range of each point for current period
    
    # reference: srlife: A Fast Tool for High Temperature Receiver
    # Design and Analysis, page 42
    T_vector = T.vector()[:]
    T_lower_700 = np.where(T_vector<700+273.15)[0]
    T_higher_700 = np.where(T_vector>=700+273.15)[0]

    num_dof_lower_700 = len(T_lower_700)
    cut_off_strain_lower_700_list = np.ones(num_dof_lower_700)*cutoff_strain_lower_700
    fatigue_strain_lower_700 = np.maximum(cut_off_strain_lower_700_list, max_vm_delta_strain.vector()[T_lower_700])

    num_dof_higher_700 = len(T_higher_700)
    cut_off_strain_higher_700_list = np.ones(num_dof_higher_700)*cut_off_strain_higher_700
    fatigue_strain_higher_700 = np.maximum(cut_off_strain_higher_700_list, max_vm_delta_strain.vector()[T_higher_700])

    N_higher_700 = pow(10, -0.98*pow(np.log10(fatigue_strain_higher_700), 2)-9.131*np.log10(fatigue_strain_higher_700)-10.95)
    D_f.vector()[T_higher_700] += 1.0/N_higher_700

    N_lower_700 = pow(10, -1.675*pow(np.log10(fatigue_strain_lower_700), 2)-13.365*np.log10(fatigue_strain_lower_700)-15.915)
    D_f.vector()[T_lower_700] += 1.0/N_lower_700
    
    max_fatigue_damage_factor.append(np.max(D_f.vector()[:]))

    xdmffile_modified_D_f.write(D_f, ii)

np.savetxt(f"{save_dir_name}/max_VM_stress_hdays.txt", max_von_mises)
np.savetxt(f"{save_dir_name}/max_VM_strain_hdays.txt", max_equi_strain)
np.savetxt(f"{save_dir_name}/max_VM_CR_strain_hdays.txt", max_equi_creep_strain)
np.savetxt(f"{save_dir_name}/max_D_c.txt", max_creep_damage_factor)
np.savetxt(f"{save_dir_name}/max_D_f.txt", max_fatigue_damage_factor)

fig1 = plt.figure()
plt.plot(time_list, max_von_mises, '.')
plt.xlabel('time (h)')
plt.ylabel('maximum equivalent stress (MPa)')
plt.savefig('max_equi_stress_hdays_QQ', dpi=300)

fig2 = plt.figure()
plt.plot(time_list, max_equi_strain, '.')
plt.xlabel('time (h)')
plt.ylabel('maximum equivalent strain')
plt.savefig('max_equi_strain_hdays_QQ', dpi=300)

fig3 = plt.figure()
plt.plot(time_list, max_equi_creep_strain, '.')
plt.xlabel('time (h)')
plt.ylabel('maximum equivalent creep strain')
plt.savefig('max_equi_creep_strain_hdays_QQ', dpi=300)
