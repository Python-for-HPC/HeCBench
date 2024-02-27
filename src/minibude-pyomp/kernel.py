from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_wtime, omp_set_num_threads, omp_get_num_threads, omp_get_num_devices, omp_is_initial_device, omp_get_thread_num, omp_get_team_num
import math
import numpy as np

@njit
def fasten_main(teams, 
                block, 
                ntypes, 
                nposes, 
                natlig, 
                natpro, 
                protein_molecule_xyz, 
                protein_molecule_type, 
                ligand_molecule_xyz, 
                ligand_molecule_type, 
                transforms_0, 
                transforms_1, 
                transforms_2, 
                transforms_3, 
                transforms_4, 
                transforms_5, 
                forcefield_rhe, 
                forcefield_hbtype, 
                etotals, 
                NUM_TD_PER_THREAD,
                FLT_MAX):
    g_etot = np.empty((teams,NUM_TD_PER_THREAD), dtype=np.float32)
    g_lpos = np.empty((teams, NUM_TD_PER_THREAD, 3), dtype=np.float32)
    g_transform = np.empty((teams, NUM_TD_PER_THREAD, 4, 3), dtype=np.float32)
    g_local_forcefield_rhe = np.zeros((teams, 64,3), dtype=np.float32)
    g_local_forcefield_hbtype = np.zeros((teams,64), dtype=np.int32)

    with openmp("""target teams num_teams(teams) thread_limit(block) map (to: g_etot, g_lpos, g_transform, g_local_forcefield_rhe, g_local_forcefield_hbtype) device(1)"""):
        with openmp("""parallel"""):
            lid = omp_get_thread_num()
            gid = omp_get_team_num()
            lrange = omp_get_num_threads()
            etot = g_etot[gid]
            lpos = g_lpos[gid]
            transform = g_transform[gid]
            local_forcefield_rhe = g_local_forcefield_rhe[gid]
            local_forcefield_hbtype = g_local_forcefield_hbtype[gid]

            ix = gid * lrange * NUM_TD_PER_THREAD + lid
            ix = ix if (ix < nposes) else (nposes - NUM_TD_PER_THREAD)

            for i in range(lid, ntypes, lrange):
                #local_forcefield_rhe[i, :] = forcefield_rhe[i, :] # See numbaWithOpenmp issue #23.
                local_forcefield_rhe[i, 0] = forcefield_rhe[i, 0]
                local_forcefield_rhe[i, 1] = forcefield_rhe[i, 1]
                local_forcefield_rhe[i, 2] = forcefield_rhe[i, 2]
                local_forcefield_hbtype[i] = forcefield_hbtype[i]
            
            for i in range(NUM_TD_PER_THREAD):
                index = ix + i * lrange
                sx = math.sin(transforms_0[i])
                cx = math.cos(transforms_0[i])
                sy = math.sin(transforms_1[i])
                cy = math.cos(transforms_1[i])
                sz = math.sin(transforms_2[i])
                cz = math.cos(transforms_2[i])
                transform[i, 0, 0] = cy * cz
                transform[i, 1, 0] = sx * sy * cz - cx * sz
                transform[i, 2, 0] = cx * sy * cz + sx * sz
                transform[i, 3, 0] = transforms_3[i]
                transform[i, 0, 1] = cy * sz
                transform[i, 1, 1] = sx * sy * sz + cx * cz
                transform[i, 2, 1] = cx * sy * sz - sx * cz
                transform[i, 3, 1] = transforms_4[i]
                transform[i, 0, 2] = sy * -1
                transform[i, 1, 2] = sx * cy
                transform[i, 2, 2] = cx * cy
                transform[i, 3, 2] = transforms_5[i]
                etot[i] = 0
                
            with openmp("""barrier"""):
                pass

                il = 0
                while il == 0 or il < natlig:
                    lfindex = ligand_molecule_type[il]
                    l_params_rhe = local_forcefield_rhe[lfindex]
                    l_params_hbtype = local_forcefield_hbtype[lfindex]
                    lhphb_ltz = l_params_rhe[0] < 0.0
                    lhphb_gtz = l_params_rhe[0] > 0.0
                    linitpos = ligand_molecule_xyz[il]
                    
                    for i in range(NUM_TD_PER_THREAD):
                        lpos[i, 0] = transform[i, 3, 0] + linitpos[0] * transform[i, 0, 0] + linitpos[1] * transform[i, 1, 0] + linitpos[2] * transform[i, 2, 0]
                        lpos[i, 1] = transform[i, 3, 1] + linitpos[0] * transform[i, 0, 1] + linitpos[1] * transform[i, 1, 1] + linitpos[2] * transform[i, 2, 1]
                        lpos[i, 2] = transform[i, 3, 2] + linitpos[0] * transform[i, 0, 2] + linitpos[1] * transform[i, 2, 2] + linitpos[2] * transform[i, 2, 2]
                    
                    for ip in range(natpro):
                        p_atom_xyz = protein_molecule_xyz[ip]
                        p_atom_type = protein_molecule_type[ip]
                        p_params_rhe = local_forcefield_rhe[p_atom_type]
                        p_params_hbtype = local_forcefield_hbtype[p_atom_type]
                        radij = p_params_rhe[0] + l_params_rhe[0]
                        r_radij = 1.0 / radij
                        elcdst = 4.0 if p_params_hbtype == 70 and l_params_hbtype == 70 else 2.0
                        elcdst1 = 0.25 if p_params_hbtype == 70 and l_params_hbtype == 70 else 0.5
                        type_E = p_params_hbtype == 69 or l_params_hbtype == 69
                        phphb_ltz = p_params_rhe[1] < 0.0
                        phphb_gtz = p_params_rhe[1] > 0.0
                        phphb_nz = p_params_rhe[1] != 0.0
                        p_hphb = p_params_rhe[1] * -1.0 if phphb_ltz and lhphb_gtz else p_params_rhe[1] * 1.0
                        l_hphb = l_params_rhe[1] * -1.0 if phphb_gtz and lhphb_ltz else l_params_rhe[1] * 1.0
                        distdslv = 5.5 if phphb_ltz else 1.0 if lhphb_ltz else (FLT_MAX * -1)
                        r_distdslv = 1.0 / distdslv
                        chrg_init = l_params_rhe[2] * p_params_rhe[2]
                        dslv_init = p_hphb + l_hphb
                        
                        for i in range(NUM_TD_PER_THREAD):
                            x = lpos[i][0] - p_atom_xyz[0]
                            y = lpos[i][1] - p_atom_xyz[1]
                            z = lpos[i][2] - p_atom_xyz[2]
                            distij = math.sqrt(x * x + y * y + z * z)
                            distbb = distij - radij
                            zone1 = distbb < 0.0
                            etot[i] += (1.0 - (distij * r_radij)) * (2 * 38.0 if zone1 else 0.0)
                            chrg_e = chrg_init * ((1 if zone1 else (1.0 - distbb * elcdst1)) * (1 if distbb < elcdst else 0.0))
                            neg_chrg_e = abs(chrg_e) * -1
                            chrg_e = neg_chrg_e if type_E else chrg_e
                            etot[i] += chrg_e * 45.0
                            coeff = 1.0 - (distbb * r_distdslv)
                            dslv_e = dslv_init * (1 if distbb < distdslv and phphb_nz else 0.0)
                            dslv_e *= 1 if zone1 else coeff
                            etot[i] += dslv_e

                    il += 1
                
                td_base = gid * lrange * NUM_TD_PER_THREAD + lid
                if td_base < nposes:
                    for i in range(NUM_TD_PER_THREAD):
                        etotals[td_base + i * lrange] = etot[i] * 0.5
