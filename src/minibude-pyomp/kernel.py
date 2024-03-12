from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import (
    omp_get_num_threads,
    omp_get_thread_num,
    omp_get_team_num,
)
import math
import numpy as np
import numba


@njit
def fasten_main(
    teams,
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
):
    g_etot = np.empty((teams, block, NUM_TD_PER_THREAD), dtype=np.float32)
    g_lpos = np.empty((teams, block, NUM_TD_PER_THREAD, 3), dtype=np.float32)
    g_transform = np.empty((teams, block, NUM_TD_PER_THREAD, 3, 4), dtype=np.float32)
    g_local_forcefield_rhe = np.empty((teams, ntypes, 3), dtype=np.float32)
    g_local_forcefield_hbtype = np.empty((teams, ntypes), dtype=np.int32)

    with openmp(
        """target teams num_teams(teams) thread_limit(block)
                map(alloc: g_etot, g_lpos, g_transform, g_local_forcefield_rhe, g_local_forcefield_hbtype) device(1)"""
    ):
        with openmp("parallel"):
            # Constants cast to float32 for single-precision arithmetic.
            ZERO = np.float32(0.0)
            QUARTER = np.float32(0.25)
            HALF = np.float32(0.5)
            ONE = np.float32(1.0)
            NEGONE = np.float32(-1.0)
            TWO = np.float32(2.0)
            FOUR = np.float32(4.0)
            CNSTNT = np.float32(45.0)
            HARDNESS = np.float32(38.0)
            NPNDIST = np.float32(5.5)
            FLT_MAX = np.float32(3.402823466e38)

            lid = omp_get_thread_num()
            gid = omp_get_team_num()
            lrange = omp_get_num_threads()

            etot = g_etot[gid, lid]
            lpos = g_lpos[gid, lid]
            transform = g_transform[gid, lid]
            local_forcefield_rhe = g_local_forcefield_rhe[gid]
            local_forcefield_hbtype = g_local_forcefield_hbtype[gid]

            if gid * lrange * NUM_TD_PER_THREAD + lid < nposes:
                ix = gid * lrange * NUM_TD_PER_THREAD + lid
            else:
                ix = nposes - NUM_TD_PER_THREAD
            # ix = gid * lrange * NUM_TD_PER_THREAD + lid
            # ix = ix if (ix < nposes) else (nposes - NUM_TD_PER_THREAD)
            # print("lid:", lid, gid, lrange, ix, ntypes, NUM_TD_PER_THREAD)

            for i in range(lid, ntypes, lrange):
                # local_forcefield_rhe[i, :] = forcefield_rhe[i, :] # See numbaWithOpenmp issue #23.
                local_forcefield_rhe[i, 0] = forcefield_rhe[i, 0]
                local_forcefield_rhe[i, 1] = forcefield_rhe[i, 1]
                local_forcefield_rhe[i, 2] = forcefield_rhe[i, 2]
                local_forcefield_hbtype[i] = forcefield_hbtype[i]

            for i in range(NUM_TD_PER_THREAD):
                index = ix + i * lrange
                sx = math.sin(transforms_0[index])
                cx = math.cos(transforms_0[index])
                sy = math.sin(transforms_1[index])
                cy = math.cos(transforms_1[index])
                sz = math.sin(transforms_2[index])
                cz = math.cos(transforms_2[index])
                transform[i, 0, 0] = cy * cz
                transform[i, 0, 1] = sx * sy * cz - cx * sz
                transform[i, 0, 2] = cx * sy * cz + sx * sz
                transform[i, 0, 3] = transforms_3[index]
                transform[i, 1, 0] = cy * sz
                transform[i, 1, 1] = sx * sy * sz + cx * cz
                transform[i, 1, 2] = cx * sy * sz - sx * cz
                transform[i, 1, 3] = transforms_4[index]
                transform[i, 2, 0] = sy * NEGONE
                transform[i, 2, 1] = sx * cy
                transform[i, 2, 2] = cx * cy
                transform[i, 2, 3] = transforms_5[index]
                etot[i] = ZERO
            # print("transform\n", transform)

            with openmp("""barrier"""):
                pass

            for il in range(natlig):
                lfindex = ligand_molecule_type[il]
                l_params_rhe = local_forcefield_rhe[lfindex]
                l_params_hbtype = local_forcefield_hbtype[lfindex]
                lhphb_ltz = l_params_rhe[1] < ZERO
                lhphb_gtz = l_params_rhe[1] > ZERO
                linitpos = ligand_molecule_xyz[il]

                for i in range(NUM_TD_PER_THREAD):
                    lpos[i, 0] = (
                        transform[i, 0, 3]
                        + linitpos[0] * transform[i, 0, 0]
                        + linitpos[1] * transform[i, 0, 1]
                        + linitpos[2] * transform[i, 0, 2]
                    )
                    lpos[i, 1] = (
                        transform[i, 1, 3]
                        + linitpos[0] * transform[i, 1, 0]
                        + linitpos[1] * transform[i, 1, 1]
                        + linitpos[2] * transform[i, 1, 2]
                    )
                    lpos[i, 2] = (
                        transform[i, 2, 3]
                        + linitpos[0] * transform[i, 2, 0]
                        + linitpos[1] * transform[i, 2, 1]
                        + linitpos[2] * transform[i, 2, 2]
                    )
                # print("lpos\n", lpos)

                for ip in range(natpro):
                    p_atom_xyz = protein_molecule_xyz[ip]
                    p_atom_type = protein_molecule_type[ip]
                    p_params_rhe = local_forcefield_rhe[p_atom_type]
                    p_params_hbtype = local_forcefield_hbtype[p_atom_type]
                    radij = p_params_rhe[0] + l_params_rhe[0]
                    r_radij = ONE / radij
                    elcdst = (
                        FOUR if p_params_hbtype == 70 and l_params_hbtype == 70 else TWO
                    )
                    elcdst1 = (
                        QUARTER
                        if p_params_hbtype == 70 and l_params_hbtype == 70
                        else HALF
                    )
                    type_E = p_params_hbtype == 69 or l_params_hbtype == 69
                    phphb_ltz = p_params_rhe[1] < ZERO
                    phphb_gtz = p_params_rhe[1] > ZERO
                    phphb_nz = p_params_rhe[1] != ZERO
                    p_hphb = (
                        (p_params_rhe[1] * NEGONE)
                        if phphb_ltz and lhphb_gtz
                        else (p_params_rhe[1] * ONE)
                    )
                    l_hphb = (
                        (l_params_rhe[1] * NEGONE)
                        if phphb_gtz and lhphb_ltz
                        else (l_params_rhe[1] * ONE)
                    )
                    distdslv = (
                        (np.float32(NPNDIST) if lhphb_ltz else ONE)
                        if phphb_ltz
                        else (ONE if lhphb_ltz else (FLT_MAX * NEGONE))
                    )
                    r_distdslv = ONE / distdslv
                    chrg_init = l_params_rhe[2] * p_params_rhe[2]
                    dslv_init = p_hphb + l_hphb
                    # print("p_hphb:", p_hphb, l_hphb, p_params_rhe[1], phphb_ltz, lhphb_gtz, lhphb_ltz, l_params_rhe[1], distdslv)

                    # print("etot", il, ip, etot[0], end=" ")
                    for i in range(NUM_TD_PER_THREAD):
                        x = lpos[i, 0] - p_atom_xyz[0]
                        y = lpos[i, 1] - p_atom_xyz[1]
                        z = lpos[i, 2] - p_atom_xyz[2]
                        distij = math.sqrt(x * x + y * y + z * z)
                        distbb = distij - radij
                        zone1 = distbb < ZERO
                        etot[i] += (ONE - (distij * r_radij)) * (
                            TWO * HARDNESS if zone1 else ZERO
                        )
                        # print(etot[i], distij, r_radij, zone1, end=" ")
                        chrg_e = chrg_init * (
                            (ONE if zone1 else (ONE - distbb * elcdst1))
                            * (ONE if distbb < elcdst else ZERO)
                        )
                        neg_chrg_e = abs(chrg_e) * NEGONE
                        chrg_e = neg_chrg_e if type_E else chrg_e
                        etot[i] += chrg_e * CNSTNT
                        # print(etot[i], end=" ")
                        coeff = ONE - (distbb * r_distdslv)
                        dslv_e = dslv_init * (
                            ONE if ((distbb < distdslv) and phphb_nz) else ZERO
                        )
                        # print("dslv_init", dslv_init, dslv_e, distbb, distdslv, phphb_nz, end=" ")
                        dslv_e *= ONE if zone1 else coeff
                        etot[i] += dslv_e
                        # print(x, y, z, etot[i], chrg_init, coeff)

            td_base = gid * lrange * NUM_TD_PER_THREAD + lid
            if td_base < nposes:
                for i in range(NUM_TD_PER_THREAD):
                    etotals[td_base + i * lrange] = etot[i] * HALF
