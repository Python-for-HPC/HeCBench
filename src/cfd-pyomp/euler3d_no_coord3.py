"""
  euler3d.py
  : parallelized code of CFD

  - Python translation from HeCbench cfd-omp euler3d.cpp.
  - parallelization with OpenCL API has been applied by
"""

from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device
import numba
import numpy as np
import sys
import math
import collections

# ----------------------------------------------------------------

DEBUG = False
GAMMA = numba.float64(1.4)
#GAMMA = numba.float32(1.4)
iterations = 1
#iterations = 2000
block_length = 192

NDIM = 3
NNB = 4

RK = 3  # 3rd order RK
ff_mach = numba.float32(1.2)
deg_angle_of_attack = numba.float32(0.0)

VAR_DENSITY = 0
VAR_MOMENTUM = 1
VAR_DENSITY_ENERGY = (VAR_MOMENTUM + NDIM)
NVAR = (VAR_DENSITY_ENERGY + 1)

BLOCK_SIZE_0 = 192
BLOCK_SIZE_1 = 192
BLOCK_SIZE_2 = 192
BLOCK_SIZE_3 = 192
BLOCK_SIZE_4 = 192

if block_length > 128:
    print("The kernels may fail too launch on some systems if the block length is too large")

#pragma omp declare target
@njit(inline='always')
def compute_velocity(density, momentum_x, momentum_y, momentum_z):
    return (momentum_x / density, momentum_y / density, momentum_z / density)
#pragma omp end declare target

#pragma omp declare target
@njit(inline='always')
def compute_speed_sqd(velocity_x, velocity_y, velocity_z):
    return velocity_x*velocity_x + velocity_y*velocity_y + velocity_z*velocity_z
#pragma omp end declare target

#pragma omp declare target
@njit(inline='always')
def compute_pressure(density, density_energy, speed_sqd, gamma):
    return (gamma - numba.float32(1.0)) * (density_energy - numba.float32(0.5) * density * speed_sqd)
#pragma omp end declare target

# sqrt is a device function
#pragma omp declare target
@njit(inline='always')
def compute_speed_of_sound(density, pressure, gamma):
    return math.sqrt(gamma*pressure/density)
#pragma omp end declare target

#pragma omp declare target
#@njit(inline='always')
@njit
def compute_flux_contribution(density, momentum_x, momentum_y, momentum_z, density_energy, pressure, velocity_x, velocity_y, velocity_z):
  fc_momentum_x_x = velocity_x*momentum_x + pressure
  fc_momentum_x_y = velocity_x*momentum_y
  fc_momentum_x_z = velocity_x*momentum_z

  fc_momentum_y_x = fc_momentum_x_y
  fc_momentum_y_y = velocity_y*momentum_y + pressure
  fc_momentum_y_z = velocity_y*momentum_z

  fc_momentum_z_x = fc_momentum_x_z
  fc_momentum_z_y = fc_momentum_y_z
  fc_momentum_z_z = velocity_z*momentum_z + pressure

  de_p = density_energy + pressure

  fc_density_energy_x = velocity_x*de_p
  fc_density_energy_y = velocity_y*de_p
  fc_density_energy_z = velocity_z*de_p
  return fc_momentum_x_x, fc_momentum_x_y, fc_momentum_x_z, fc_momentum_y_x, fc_momentum_y_y, fc_momentum_y_z, fc_momentum_z_x, fc_momentum_z_y, fc_momentum_z_z, fc_density_energy_x, fc_density_energy_y, fc_density_energy_z 
#pragma omp end declare target


#pragma omp declare target
#@njit(inline='always')
@njit
def copy(dst, src, N):
    with openmp("target teams distribute parallel for thread_limit(256) device(0)"):
        for i in range(N):
            dst[i] = src[i]
#pragma omp end declare target

"""
def dump(h_variables, nel, nelr):
    with open("density", "w") as density_file:
        print(nel, nelr, file=density_file)
        for i in range(nel):
            print(h_variables[i + VAR_DENSITY*nelr], file=density_file)

    with open("momentum", "w") as momentum_file:
        print(nel, nelr, file=momentum_file)
        for i in range(nel):
            for j in range(NDIM):
                print(h_variables[i + (VAR_MOMENTUM+j)*nelr], end=" ", file=momentum_file)
            print("", file=momentum_file)

    with open("density_energy", "w") as de_file:
        print(nel, nelr, file=de_file)
        for i in range(nel):
            print(h_variables[i + VAR_DENSITY_ENERGY*nelr], file=de_file)
"""

#pragma omp declare target
@njit
#@njit(inline='always')
def initialize_buffer(d, val, number_words):
    with openmp("target teams distribute parallel for thread_limit(256) device(0)"):
        for i in range(number_words):
            d[i] = val
#pragma omp end declare target

#pragma omp declare target
#@njit(inline='always')
@njit
def initialize_variables(nelr, variables, ff_variable, block_size_1, nvar):
    with openmp("target teams distribute parallel for thread_limit(block_size_1) device(0)"):
        for i in range(nelr):
            for j in range(nvar):
                variables[i + j*nelr] = ff_variable[j]
#pragma omp end declare target

#pragma omp declare target
#@njit(inline='always')
@njit
def compute_step_factor(nelr, variables, areas, step_factors, block_size_2, var_density, var_momentum, var_density_energy, gamma):
    #with openmp("target teams distribute parallel for thread_limit(block_size_2) map(to: nelr, var_density, var_momentum, var_density_energy, gamma)"):
    with openmp("target teams distribute parallel for thread_limit(block_size_2) firstprivate(nelr, var_density, var_momentum, var_density_energy, gamma) device(0)"):
        for i in range(nelr):
            density = variables[i + var_density*nelr]
            momentum_x = variables[i + (var_momentum+0)*nelr]
            momentum_y = variables[i + (var_momentum+1)*nelr]
            momentum_z = variables[i + (var_momentum+2)*nelr]

            density_energy = variables[i + var_density_energy*nelr]

            velocity_x, velocity_y, velocity_z = compute_velocity(density, momentum_x, momentum_y, momentum_z)
            speed_sqd = compute_speed_sqd(velocity_x, velocity_y, velocity_z)

            pressure = compute_pressure(density, density_energy, speed_sqd, gamma)
            speed_of_sound = compute_speed_of_sound(density, pressure, gamma)
            step_factors[i] = numba.float32(0.5) / (math.sqrt(areas[i]) * (math.sqrt(speed_sqd) + speed_of_sound))
#pragma omp end declare target

#pragma omp declare target
@njit
def compute_flux(
    nelr,
    nvar,
    elements_surrounding_elements,
    normals,
    variables,
    ff_variable,
    fluxes,
    ff_flux_contribution_density_energy_x,
    ff_flux_contribution_density_energy_y,
    ff_flux_contribution_density_energy_z,
    ff_flux_contribution_momentum_x_x,
    ff_flux_contribution_momentum_x_y,
    ff_flux_contribution_momentum_x_z,
    ff_flux_contribution_momentum_y_x,
    ff_flux_contribution_momentum_y_y,
    ff_flux_contribution_momentum_y_z,
    ff_flux_contribution_momentum_z_x,
    ff_flux_contribution_momentum_z_y,
    ff_flux_contribution_momentum_z_z,
    block_size_3,
    nnb,
    var_density,
    var_density_energy,
    var_momentum,
    gamma):

    with openmp("target teams distribute parallel for thread_limit(block_size_3) device(0)"):
      for i in range(nelr):
        smoothing_coefficient = numba.float32(0.2)

        density_i = variables[i + var_density*nelr]
        momentum_i_x = numba.float32(variables[i + (var_momentum+0)*nelr])
        momentum_i_y = numba.float32(variables[i + (var_momentum+1)*nelr])
        momentum_i_z = numba.float32(variables[i + (var_momentum+2)*nelr])

        density_energy_i = variables[i + var_density_energy*nelr]

        velocity_i_x, velocity_i_y, velocity_i_z = compute_velocity(density_i, momentum_i_x, momentum_i_y, momentum_i_z)
        speed_sqd_i = compute_speed_sqd(velocity_i_x, velocity_i_y, velocity_i_z)
        speed_i = math.sqrt(speed_sqd_i)
        pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i, gamma)
        speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i, gamma)
        flux_contribution_i_momentum_x_x, flux_contribution_i_momentum_x_y, flux_contribution_i_momentum_x_z, flux_contribution_i_momentum_y_x, flux_contribution_i_momentum_y_y, flux_contribution_i_momentum_y_z, flux_contribution_i_momentum_z_x, flux_contribution_i_momentum_z_y, flux_contribution_i_momentum_z_z, flux_contribution_i_density_energy_x, flux_contribution_i_density_energy_y, flux_contribution_i_density_energy_z = compute_flux_contribution(density_i, momentum_i_x, momentum_i_y, momentum_i_z, density_energy_i, pressure_i, velocity_i_x, velocity_i_y, velocity_i_z)

        flux_i_density = numba.float32(0.0)
        flux_i_density_energy = numba.float32(0.0)

        for j in range(nnb):
          nb = elements_surrounding_elements[i + j*nelr]
          normal_x = numba.float32(normals[i + (j + 0*nnb)*nelr])
          normal_y = numba.float32(normals[i + (j + 1*nnb)*nelr])
          normal_z = numba.float32(normals[i + (j + 2*nnb)*nelr])
          normal_len = compute_speed_sqd(normal_x, normal_y, normal_z)

          if nb >= 0:   # a legitimate neighbor
              density_nb = variables[nb + var_density*nelr]
              momentum_nb_x = numba.float32(variables[nb + (var_momentum+0)*nelr])
              momentum_nb_y = numba.float32(variables[nb + (var_momentum+1)*nelr])
              momentum_nb_z = numba.float32(variables[nb + (var_momentum+2)*nelr])
              density_energy_nb = variables[nb + var_density_energy*nelr]
              velocity_nb_x, velocity_nb_y, velocity_nb_z = compute_velocity(density_nb, momentum_nb_x, momentum_nb_y, momentum_nb_z)
              speed_sqd_nb = compute_speed_sqd(velocity_nb_x, velocity_nb_y, velocity_nb_z)
              pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb, gamma)
              speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb, gamma)
              flux_contribution_nb_momentum_x_x, flux_contribution_nb_momentum_x_y, flux_contribution_nb_momentum_x_z, flux_contribution_nb_momentum_y_x, flux_contribution_nb_momentum_y_y, flux_contribution_nb_momentum_y_z, flux_contribution_nb_momentum_z_x, flux_contribution_nb_momentum_z_y, flux_contribution_nb_momentum_z_z, flux_contribution_nb_density_energy_x, flux_contribution_nb_density_energy_y, flux_contribution_nb_density_energy_z = compute_flux_contribution(density_nb, momentum_nb_x, momentum_nb_y, momentum_nb_z, density_energy_nb, pressure_nb, velocity_nb_x, velocity_nb_y, velocity_nb_z)

              # artificial viscosity
              factor = (normal_len * -1)*smoothing_coefficient*numba.float32(0.5)*(speed_i + math.sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb)
              flux_i_density += factor*(density_i-density_nb)
              flux_i_density_energy += factor*(density_energy_i-density_energy_nb)
              flux_i_momentum_x = numba.float32(factor*(momentum_i_x-momentum_nb_x))
              flux_i_momentum_y = numba.float32(factor*(momentum_i_y-momentum_nb_y))
              flux_i_momentum_z = numba.float32(factor*(momentum_i_z-momentum_nb_z))

              # accumulate cell-centered fluxes
              factor = numba.float32(0.5)*normal_x
              flux_i_density += factor*(momentum_nb_x+momentum_i_x)
              flux_i_density_energy += factor*(flux_contribution_nb_density_energy_x+flux_contribution_i_density_energy_x)
              flux_i_momentum_x = numba.float32(factor*(flux_contribution_nb_momentum_x_x+flux_contribution_i_momentum_x_x))
              flux_i_momentum_y = numba.float32(factor*(flux_contribution_nb_momentum_y_x+flux_contribution_i_momentum_y_x))
              flux_i_momentum_z = numba.float32(factor*(flux_contribution_nb_momentum_z_x+flux_contribution_i_momentum_z_x))

              factor = numba.float32(0.5)*normal_y
              flux_i_density += factor*(momentum_nb_y+momentum_i_y)
              flux_i_density_energy += factor*(flux_contribution_nb_density_energy_y+flux_contribution_i_density_energy_y)
              flux_i_momentum_x = numba.float32(factor*(flux_contribution_nb_momentum_x_y+flux_contribution_i_momentum_x_y))
              flux_i_momentum_y = numba.float32(factor*(flux_contribution_nb_momentum_y_y+flux_contribution_i_momentum_y_y))
              flux_i_momentum_z = numba.float32(factor*(flux_contribution_nb_momentum_z_y+flux_contribution_i_momentum_z_y))

              factor = numba.float32(0.5)*normal_z
              flux_i_density += factor*(momentum_nb_z+momentum_i_z)
              flux_i_density_energy += factor*(flux_contribution_nb_density_energy_z+flux_contribution_i_density_energy_z)
              flux_i_momentum_x = numba.float32(factor*(flux_contribution_nb_momentum_x_z+flux_contribution_i_momentum_x_z))
              flux_i_momentum_y = numba.float32(factor*(flux_contribution_nb_momentum_y_z+flux_contribution_i_momentum_y_z))
              flux_i_momentum_z = numba.float32(factor*(flux_contribution_nb_momentum_z_z+flux_contribution_i_momentum_z_z))
          elif nb == -1:  # a wing boundary
              flux_i_momentum_x = numba.float32(normal_x*pressure_i)
              flux_i_momentum_y = numba.float32(normal_y*pressure_i)
              flux_i_momentum_z = numba.float32(normal_z*pressure_i)
          elif nb == -2: # a far field boundary
              pass
              """
              factor = numba.float32(0.5)*normal_x
              flux_i_momentum_x = numba.float32(0.0)
              flux_i_momentum_y = numba.float32(0.0)
              flux_i_momentum_z = numba.float32(0.0)
              flux_i_density += factor*(ff_variable[var_momentum+0]+momentum_i_x)
              flux_i_density_energy += factor*(ff_flux_contribution_density_energy_x+flux_contribution_i_density_energy_x)
              flux_i_momentum_x = numba.float32(factor*(ff_flux_contribution_momentum_x_x + flux_contribution_i_momentum_x_x))
              flux_i_momentum_y = numba.float32(factor*(ff_flux_contribution_momentum_y_x + flux_contribution_i_momentum_y_x))
              flux_i_momentum_z = numba.float32(factor*(ff_flux_contribution_momentum_z_x + flux_contribution_i_momentum_z_x))

              factor = numba.float32(0.5)*normal_y
              flux_i_density += factor*(ff_variable[var_momentum+1]+momentum_i_y)
              flux_i_density_energy += factor*(ff_flux_contribution_density_energy_y+flux_contribution_i_density_energy_y)
              """
              """
              flux_i_momentum_x = numba.float32(factor*(ff_flux_contribution_momentum_x_y + flux_contribution_i_momentum_x_y))
              flux_i_momentum_y = numba.float32(factor*(ff_flux_contribution_momentum_y_y + flux_contribution_i_momentum_y_y))
              flux_i_momentum_z = numba.float32(factor*(ff_flux_contribution_momentum_z_y + flux_contribution_i_momentum_z_y))
              """

              """
              factor = numba.float32(0.5)*normal_z
              flux_i_density += factor*(ff_variable[var_momentum+2]+momentum_i_z)
              flux_i_density_energy += factor*(ff_flux_contribution_density_energy_z+flux_contribution_i_density_energy_z)
              flux_i_momentum_x = numba.float32(factor*(ff_flux_contribution_momentum_x_z + flux_contribution_i_momentum_x_z))
              flux_i_momentum_y = numba.float32(factor*(ff_flux_contribution_momentum_y_z + flux_contribution_i_momentum_y_z))
              flux_i_momentum_z = numba.float32(factor*(ff_flux_contribution_momentum_z_z + flux_contribution_i_momentum_z_z))
              """

        fluxes[i + var_density*nelr] = flux_i_density
        """
        fluxes[i + (var_momentum+0)*nelr] = flux_i_momentum_x
        fluxes[i + (var_momentum+1)*nelr] = flux_i_momentum_y
        fluxes[i + (var_momentum+2)*nelr] = flux_i_momentum_z
        """
        fluxes[i + var_density_energy*nelr] = flux_i_density_energy
#pragma omp end declare target

#pragma omp declare target
@njit
def time_step(j, nelr, old_variables, variables, step_factors, fluxes, block_size_4, rk, var_momentum, var_density, var_density_energy):
    with openmp("target teams distribute parallel for thread_limit(block_size_4) device(0)"):
        for i in range(nelr):
            factor = step_factors[i]/numba.float32((rk+1-j))

            variables[i + var_density*nelr] = old_variables[i + var_density*nelr] + factor*fluxes[i + var_density*nelr]
            variables[i + var_density_energy*nelr] = old_variables[i + var_density_energy*nelr] + factor*fluxes[i + var_density_energy*nelr]
            variables[i + (var_momentum+0)*nelr] = old_variables[i + (var_momentum+0)*nelr] + factor*fluxes[i + (var_momentum+0)*nelr]
            variables[i + (var_momentum+1)*nelr] = old_variables[i + (var_momentum+1)*nelr] + factor*fluxes[i + (var_momentum+1)*nelr]
            variables[i + (var_momentum+2)*nelr] = old_variables[i + (var_momentum+2)*nelr] + factor*fluxes[i + (var_momentum+2)*nelr]
#pragma omp end declare target

@njit
def test(nel, nelr, h_ff_variable, h_areas, h_elements_surrounding_elements, h_normals, h_fluxes, h_old_variables, h_step_factors, h_variables, nvar, block_size_0, block_size_1, block_size_2, block_size_3, block_size_4, rk):
    initialize_variables(nelr, h_variables, h_ff_variable, block_size_1)
    #with openmp("target teams distribute parallel for thread_limit(block_size_1)"):
    #    for i in range(nelr):
    #        for j in range(nvar):
    #            h_variables[i + j*nelr] = h_ff_variable[j]

    return 0

@njit
def core(nel,
         nelr,
         h_ff_variable,
         h_areas,
         h_elements_surrounding_elements,
         h_normals,
         h_fluxes,
         h_old_variables,
         h_step_factors,
         h_variables,
         nvar,
         block_size_0,
         block_size_1,
         block_size_2,
         block_size_3,
         block_size_4,
         rk,
         var_density,
         var_momentum,
         var_density_energy,
         gamma,
         nnb,
         h_ff_flux_contribution_density_energy_x_x,
         h_ff_flux_contribution_density_energy_x_y,
         h_ff_flux_contribution_density_energy_x_z,
         h_ff_flux_contribution_density_energy_y_x,
         h_ff_flux_contribution_density_energy_y_y,
         h_ff_flux_contribution_density_energy_y_z,
         h_ff_flux_contribution_density_energy_z_x,
         h_ff_flux_contribution_density_energy_z_y,
         h_ff_flux_contribution_density_energy_z_z):
    # copy far field conditions to the device
    with openmp("""target data
                     map(to:
                       h_ff_variable,
                       h_areas,
                       h_elements_surrounding_elements,
                       h_normals)
                     map(alloc: h_fluxes,
                                h_old_variables,
                                h_step_factors)
                     map(from: h_variables)
                     device(0)"""):

        #with openmp("""target enter data
        #                 map(to:
        #                   h_ff_variable,
        #                   h_areas,
        #                   h_elements_surrounding_elements,
        #                   h_normals)
        #                 map(alloc: h_fluxes,
        #                            h_old_variables,
        #                            h_step_factors) device(0)"""):
        #    pass

        #with openmp("""target enter data
        #                 map(to:
        #                   h_ff_variable,
        #                   h_areas,
        #                   h_elements_surrounding_elements,
        #                   h_normals)
        #                 map(to: h_fluxes,
        #                         h_old_variables,
        #                         h_step_factors) device(0)"""):
        #    pass

        kernel_start = omp_get_wtime()

        initialize_variables(nelr, h_variables, h_ff_variable, block_size_1, nvar)
        initialize_variables(nelr, h_old_variables, h_ff_variable, block_size_1, nvar)
        initialize_variables(nelr, h_fluxes, h_ff_variable, block_size_1, nvar)
        initialize_buffer(h_step_factors, 0, nelr)

        # Begin iterations
        for n in range(iterations):
            copy(h_old_variables, h_variables, nelr*NVAR)

            # for the first iteration we compute the time step
            """
            if DEBUG:
                with openmp("target update from(h_old_variables) from(h_variables) device(0)"):
                    for i in range(16):
                        print(i, h_old_variables[i], h_variables[i])
            """

            compute_step_factor(nelr, h_variables, h_areas, h_step_factors, block_size_2, var_density, var_momentum, var_density_energy, gamma)

            """
            if DEBUG:
                with openmp("target update from(h_step_factors) device(0)"):
                    for i in range(16):
                        print("step factor:", i, h_step_factors[i])
            """

            for j in range(rk):
                compute_flux(
                    nelr,
                    nvar,
                    h_elements_surrounding_elements,
                    h_normals,
                    h_variables,
                    h_ff_variable,
                    h_fluxes,
                    h_ff_flux_contribution_density_energy_x,
                    h_ff_flux_contribution_density_energy_y,
                    h_ff_flux_contribution_density_energy_z,
                    h_ff_flux_contribution_momentum_x_x,
                    h_ff_flux_contribution_momentum_x_y,
                    h_ff_flux_contribution_momentum_x_z,
                    h_ff_flux_contribution_momentum_y_x,
                    h_ff_flux_contribution_momentum_y_y,
                    h_ff_flux_contribution_momentum_y_z,
                    h_ff_flux_contribution_momentum_z_x,
                    h_ff_flux_contribution_momentum_z_y,
                    h_ff_flux_contribution_momentum_z_z,
                    block_size_3,
                    nnb,
                    var_density,
                    var_density_energy,
                    var_momentum,
                    gamma)
                time_step(j, nelr, h_old_variables, h_variables, h_step_factors, h_fluxes, block_size_4, rk, var_momentum, var_density, var_density_energy)

            kernel_end = omp_get_wtime()

        #with openmp("""target exit data map(from: h_variables) device(0)"""):
        #    pass

    return kernel_end - kernel_start

if __name__ == "__main__":
    """ Main function """
    print(f"WG size of kernel:initialize = {BLOCK_SIZE_1}\nWG size of kernel:compute_step_factor = {BLOCK_SIZE_2}\nWG size of kernel:compute_flux = {BLOCK_SIZE_3}\nWG size of kernel:time_step = {BLOCK_SIZE_4}\n")

    if len(sys.argv) < 2:
        print("Please specify data file name")
        sys.exit(0)

    data_file_name = sys.argv[1]

    h_ff_variable = np.empty(NVAR, dtype=np.float32)

    # set far field conditions and load them into constant memory on the gpu
    angle_of_attack = numba.float32((3.1415926535897931 / 180.0) * deg_angle_of_attack)

    h_ff_variable[VAR_DENSITY] = 1.4

    ff_pressure = numba.float32(1.0)
    ff_speed_of_sound = numba.float32(math.sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]))
    ff_speed = ff_mach * ff_speed_of_sound

    ff_velocity_x = ff_speed * math.cos(angle_of_attack)
    ff_velocity_y = ff_speed * math.sin(angle_of_attack)
    ff_velocity_z = numba.float32(0.0)

    h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY] * ff_velocity_x
    h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY] * ff_velocity_y
    h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY] * ff_velocity_z

    h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY] * (0.5 * (ff_speed * ff_speed)) + (ff_pressure / (GAMMA - 1.0))

    h_ff_momentum_x = h_ff_variable[VAR_MOMENTUM + 0]
    h_ff_momentum_y = h_ff_variable[VAR_MOMENTUM + 1]
    h_ff_momentum_z = h_ff_variable[VAR_MOMENTUM + 2]
    h_ff_flux_contribution_momentum_x_x, h_ff_flux_contribution_momentum_x_y, h_ff_flux_contribution_momentum_x_z, h_ff_flux_contribution_momentum_y_x, h_ff_flux_contribution_momentum_y_y, h_ff_flux_contribution_momentum_y_z, h_ff_flux_contribution_momentum_z_x, h_ff_flux_contribution_momentum_z_y, h_ff_flux_contribution_momentum_z_z, h_ff_flux_contribution_density_energy_x, h_ff_flux_contribution_density_energy_y, h_ff_flux_contribution_density_energy_z = compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum_x, h_ff_momentum_y, h_ff_momentum_z, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity_x, ff_velocity_y, ff_velocity_z)

    with open(data_file_name) as data_file:
        data_file_lines = data_file.readlines()

    def file_word(data_file_lines):
        for line in data_file_lines:
            words = line.split()
            for word in words:
                yield word
    fwg = file_word(data_file_lines)

    nel = int(next(fwg))
    nelr = block_length * ((nel // block_length ) + min(1, nel % block_length))
    print(f"--cambine: nel={nel} nelr={nelr}")

    h_areas = np.empty(nelr, dtype=np.float32)
    h_elements_surrounding_elements = np.empty(nelr * NNB, dtype=np.int32)
    h_normals = np.empty(nelr * NDIM * NNB, dtype=np.float32)

    h_variables = np.empty(nelr*NVAR, dtype=np.float32)
    h_old_variables = np.empty(nelr*NVAR, dtype=np.float32)
    h_step_factors = np.empty(nelr, dtype=np.float32)
    h_fluxes = np.empty(nelr*NVAR, dtype=np.float32)

    # read in data
    for i in range(nel):
        h_areas[i] = numba.float32(next(fwg))
        for j in range(NNB):
            h_elements_surrounding_elements[i + j*nelr] = numba.int32(next(fwg))
            if h_elements_surrounding_elements[i+j*nelr] < 0:
                h_elements_surrounding_elements[i+j*nelr] = -1
            h_elements_surrounding_elements[i + j*nelr]-=1 #it's coming in with Fortran numbering

            for k in range(NDIM):
                h_normals[i + (j + k*NNB)*nelr] = (numba.float32(next(fwg))) * -1

    # fill in remaining data
    last = nel - 1
    for i in range(nel, nelr):
        h_areas[i] = h_areas[last]
        for j in range(NNB):
            # duplicate the last element
            h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr]
            for k in range(NDIM):
                h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr]

    offload_start = omp_get_wtime()

    print("nel=", type(nel), "nelr=", type(nelr), "h_ff_variable=", type(h_ff_variable))
    kernel_time = core(nel,
                       nelr,
                       h_ff_variable,
                       h_areas,
                       h_elements_surrounding_elements,
                       h_normals,
                       h_fluxes,
                       h_old_variables,
                       h_step_factors,
                       h_variables,
                       NVAR,
                       BLOCK_SIZE_0,
                       BLOCK_SIZE_1,
                       BLOCK_SIZE_2,
                       BLOCK_SIZE_3,
                       BLOCK_SIZE_4,
                       RK,
                       VAR_DENSITY,
                       VAR_MOMENTUM,
                       VAR_DENSITY_ENERGY,
                       GAMMA,
                       NNB,
                       h_ff_flux_contribution_momentum_x_x,
                       h_ff_flux_contribution_momentum_x_y,
                       h_ff_flux_contribution_momentum_x_z,
                       h_ff_flux_contribution_momentum_y_x,
                       h_ff_flux_contribution_momentum_y_y,
                       h_ff_flux_contribution_momentum_y_z,
                       h_ff_flux_contribution_momentum_z_x,
                       h_ff_flux_contribution_momentum_z_y,
                       h_ff_flux_contribution_momentum_z_z)
    #kernel_time = test(nel, nelr, h_ff_variable, h_areas, h_elements_surrounding_elements, h_normals, h_fluxes, h_old_variables, h_step_factors, h_variables, NVAR, BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4, RK)

    offload_end = omp_get_wtime()
    print("Device offloading time = ", offload_end - offload_start)
    print("Total execution time of kernels = ", kernel_time)
    print("Done...")

