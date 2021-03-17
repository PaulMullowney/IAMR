#include <iamr_mol.H>
#include <NS_util.H>
#include <iamr_mol_edge_state_K.H>

#if AMREX_USE_EB
#include <iamr_mol_eb_edge_state_K.H>
#endif


using namespace amrex;

namespace {
    std::pair<bool,bool> has_extdir_or_ho (BCRec const* bcrec, int ncomp, int dir)
    {
        std::pair<bool,bool> r{false,false};
        for (int n = 0; n < ncomp; ++n)
        {
            r.first = r.first or bcrec[n].lo(dir) == BCType::ext_dir
                              or bcrec[n].lo(dir) == BCType::hoextrap;
            r.second = r.second or bcrec[n].hi(dir) == BCType::ext_dir
                                or bcrec[n].hi(dir) == BCType::hoextrap;
        }
        return r;
    }
}
void
MOL::ComputeEdgeState (const Box& bx,
                       D_DECL( Array4<Real> const& xedge,
                               Array4<Real> const& yedge,
                               Array4<Real> const& zedge),
                       Array4<Real const> const& q,
                       const int ncomp,
                       D_DECL( Array4<Real const> const& umac,
                               Array4<Real const> const& vmac,
                               Array4<Real const> const& wmac),
                       const Box&       domain,
                       const Vector<BCRec>& bcs,
                       const        BCRec * d_bcrec )
{
    const int domain_ilo = domain.smallEnd(0);
    const int domain_ihi = domain.bigEnd(0);
    const int domain_jlo = domain.smallEnd(1);
    const int domain_jhi = domain.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
    const int domain_klo = domain.smallEnd(2);
    const int domain_khi = domain.bigEnd(2);
#endif

    AMREX_D_TERM(Box const& xbx = amrex::surroundingNodes(bx,0);,
                 Box const& ybx = amrex::surroundingNodes(bx,1);,
                 Box const& zbx = amrex::surroundingNodes(bx,2););

    const auto h_bcrec = bcs.dataPtr();

    // At an ext_dir or hoextrap boundary,
    //    the boundary value is on the face, not cell center.
    auto extdir_lohi = has_extdir_or_ho(h_bcrec, ncomp, static_cast<int>(Direction::x));
    bool has_extdir_or_ho_lo = extdir_lohi.first;
    bool has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_ilo >= xbx.smallEnd(0)-1) or
        (has_extdir_or_ho_hi and domain_ihi <= xbx.bigEnd(0)))
    {
        amrex::ParallelFor(xbx, ncomp, [d_bcrec,q,domain_ilo,domain_ihi,umac,xedge]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            bool extdir_or_ho_ilo = (d_bcrec[n].lo(0) == BCType::ext_dir) or
                                    (d_bcrec[n].lo(0) == BCType::hoextrap);
            bool extdir_or_ho_ihi = (d_bcrec[n].hi(0) == BCType::ext_dir) or
                                    (d_bcrec[n].hi(0) == BCType::hoextrap);
            Real qs;
            if (i <= domain_ilo && (d_bcrec[n].lo(0) == BCType::ext_dir)) {
                qs = q(domain_ilo-1,j,k,n);
            } else if (i >= domain_ihi+1 && (d_bcrec[n].hi(0) == BCType::ext_dir)) {
                qs = q(domain_ihi+1,j,k,n);
            } else {
                int order = 2;
                Real qpls = q(i,j,k,n) - 0.5 * amrex_calc_xslope_extdir
                    (i  ,j,k,n,order,q,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
                Real qmns = q(i-1,j,k,n) + 0.5 * amrex_calc_xslope_extdir
                    (i-1,j,k,n,order,q,extdir_or_ho_ilo,extdir_or_ho_ihi,domain_ilo,domain_ihi);
                if (umac(i,j,k) > small_vel) {
                    qs = qmns;
                } else if (umac(i,j,k) < -small_vel) {
                    qs = qpls;
                } else {
                    qs = 0.5*(qmns+qpls);
                }
            }

            xedge(i,j,k,n) = qs;
        });
    }
    else
    {
        amrex::ParallelFor(xbx, ncomp, [q,umac,xedge,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            int order = 2;
            Real qpls = q(i  ,j,k,n) - 0.5 * amrex_calc_xslope(i  ,j,k,n,order,q);
            Real qmns = q(i-1,j,k,n) + 0.5 * amrex_calc_xslope(i-1,j,k,n,order,q);
            Real qs;
            if (umac(i,j,k) > small_vel) {
                qs = qmns;
            } else if (umac(i,j,k) < -small_vel) {
                qs = qpls;
            } else {
                qs = 0.5*(qmns+qpls);
            }
            xedge(i,j,k,n) = qs;
        });
    }

    // At an ext_dir or hoextrap boundary,
    //    the boundary value is on the face, not cell center.
    extdir_lohi = has_extdir_or_ho(h_bcrec, ncomp,  static_cast<int>(Direction::y));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_jlo >= ybx.smallEnd(1)-1) or
        (has_extdir_or_ho_hi and domain_jhi <= ybx.bigEnd(1)))
    {
        amrex::ParallelFor(ybx, ncomp, [d_bcrec,q,domain_jlo,domain_jhi,vmac,yedge]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            bool extdir_or_ho_jlo = (d_bcrec[n].lo(1) == BCType::ext_dir) or
                                    (d_bcrec[n].lo(1) == BCType::hoextrap);
            bool extdir_or_ho_jhi = (d_bcrec[n].hi(1) == BCType::ext_dir) or
                                    (d_bcrec[n].hi(1) == BCType::hoextrap);
            Real qs;
            if (j <= domain_jlo && (d_bcrec[n].lo(1) == BCType::ext_dir)) {
                qs = q(i,domain_jlo-1,k,n);
            } else if (j >= domain_jhi+1 && (d_bcrec[n].hi(1) == BCType::ext_dir)) {
                qs = q(i,domain_jhi+1,k,n);
            } else {
                int order = 2;
                Real qpls = q(i,j  ,k,n) - 0.5 * amrex_calc_yslope_extdir(
                     i,j  ,k,n,order,q,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
                Real qmns = q(i,j-1,k,n) + 0.5 * amrex_calc_yslope_extdir(
                     i,j-1,k,n,order,q,extdir_or_ho_jlo,extdir_or_ho_jhi,domain_jlo,domain_jhi);
                if (vmac(i,j,k) > small_vel) {
                    qs = qmns;
                } else if (vmac(i,j,k) < -small_vel) {
                    qs = qpls;
                } else {
                    qs = 0.5*(qmns+qpls);
                }
            }

            yedge(i,j,k,n) = qs;
        });
    }
    else
    {
        amrex::ParallelFor(ybx, ncomp, [q,vmac,yedge,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            int order = 2;
            Real qpls = q(i,j  ,k,n) - 0.5 * amrex_calc_yslope(i,j  ,k,n,order,q);
            Real qmns = q(i,j-1,k,n) + 0.5 * amrex_calc_yslope(i,j-1,k,n,order,q);
            Real qs;
            if (vmac(i,j,k) > small_vel) {
                qs = qmns;
            } else if (vmac(i,j,k) < -small_vel) {
                qs = qpls;
            } else {
                qs = 0.5*(qmns+qpls);
            }

            yedge(i,j,k,n) = qs;
        });
    }

#if (AMREX_SPACEDIM == 3)
    // At an ext_dir or hoextrap boundary,
    //    the boundary value is on the face, not cell center.
    extdir_lohi = has_extdir_or_ho(h_bcrec, ncomp, static_cast<int>(Direction::z));
    has_extdir_or_ho_lo = extdir_lohi.first;
    has_extdir_or_ho_hi = extdir_lohi.second;

    if ((has_extdir_or_ho_lo and domain_klo >= zbx.smallEnd(2)-1) or
        (has_extdir_or_ho_hi and domain_khi <= zbx.bigEnd(2)))
    {
        amrex::ParallelFor(zbx, ncomp, [d_bcrec,q,domain_klo,domain_khi,wmac,zedge]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            bool extdir_or_ho_klo =   (d_bcrec[n].lo(2) == BCType::ext_dir) or
                                      (d_bcrec[n].lo(2) == BCType::hoextrap);
            bool extdir_or_ho_khi =   (d_bcrec[n].hi(2) == BCType::ext_dir) or
                                      (d_bcrec[n].hi(2) == BCType::hoextrap);
            Real qs;
            if (k <= domain_klo && (d_bcrec[n].lo(2) == BCType::ext_dir)) {
                qs = q(i,j,domain_klo-1,n);
            } else if (k >= domain_khi+1 && (d_bcrec[n].hi(2) == BCType::ext_dir)) {
                qs = q(i,j,domain_khi+1,n);
            } else {
                int order = 2;
                Real qpls = q(i,j,k,n) - 0.5 * amrex_calc_zslope_extdir(
                    i,j,k  ,n,order,q,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
                Real qmns = q(i,j,k-1,n) + 0.5 * amrex_calc_zslope_extdir(
                    i,j,k-1,n,order,q,extdir_or_ho_klo,extdir_or_ho_khi,domain_klo,domain_khi);
                if (wmac(i,j,k) > small_vel) {
                    qs = qmns;
                } else if (wmac(i,j,k) < -small_vel) {
                    qs = qpls;
                } else {
                    qs = 0.5*(qmns+qpls);
                }
            }
            zedge(i,j,k,n) = qs;
        });
    }
    else
    {
        amrex::ParallelFor(zbx, ncomp, [q,wmac,zedge,d_bcrec]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            int order = 2;
            Real qpls = q(i,j,k  ,n) - 0.5 * amrex_calc_zslope(i,j,k  ,n,order,q);
            Real qmns = q(i,j,k-1,n) + 0.5 * amrex_calc_zslope(i,j,k-1,n,order,q);
            Real qs;
            if (wmac(i,j,k) > small_vel) {
                qs = qmns;
            } else if (wmac(i,j,k) < -small_vel) {
                qs = qpls;
            } else {
                qs = 0.5*(qmns+qpls);
            }

            zedge(i,j,k,n) = qs;
        });
    }
#endif
}




//
// Compute edge state on REGULAR box
//
// void
// MOL::ComputeEdgeState (const Box& bx,
//                        D_DECL( Array4<Real> const& xedge,
//                                Array4<Real> const& yedge,
//                                Array4<Real> const& zedge),
//                        Array4<Real const> const& q,
//                        const int ncomp,
//                        D_DECL( Array4<Real const> const& umac,
//                                Array4<Real const> const& vmac,
//                                Array4<Real const> const& wmac),
//                        const Box&       domain,
//                        const Vector<BCRec>& bcs,
//                        const        BCRec * d_bcrec_ptr)
// {
//     const int domain_ilo = domain.smallEnd(0);
//     const int domain_ihi = domain.bigEnd(0);
//     const int domain_jlo = domain.smallEnd(1);
//     const int domain_jhi = domain.bigEnd(1);
// #if (AMREX_SPACEDIM==3)
//     const int domain_klo = domain.smallEnd(2);
//     const int domain_khi = domain.bigEnd(2);
// #endif

//     D_TERM( const Box& ubx = amrex::surroundingNodes(bx,0);,
//             const Box& vbx = amrex::surroundingNodes(bx,1);,
//             const Box& wbx = amrex::surroundingNodes(bx,2););

//     // At an ext_dir boundary, the boundary value is on the face, not cell center.
//     auto extdir_lohi = has_extdir_or_ho(bcs.dataPtr(), ncomp, 0);
//     bool has_extdir_or_ho_lo = extdir_lohi.first;
//     bool has_extdir_or_ho_hi = extdir_lohi.second;

//     if ((has_extdir_or_ho_lo and domain_ilo >= ubx.smallEnd(0)-1) or
//         (has_extdir_or_ho_hi and domain_ihi <= ubx.bigEnd(0)))
//     {
//         amrex::ParallelFor(ubx, ncomp, [d_bcrec_ptr,q,domain_ilo,domain_ihi,umac,xedge]
//         AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
//         {
//             xedge(i,j,k,n) = iamr_xedge_state_mol_extdir( i, j, k, n, q, umac,
//                                                           d_bcrec_ptr,
//                                                           domain_ilo, domain_ihi );
//         });
//     }
//     else
//     {
//         amrex::ParallelFor(ubx, ncomp, [q,umac,xedge]
//         AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
//         {
//             xedge(i,j,k,n) = iamr_xedge_state_mol( i, j, k, n, q, umac);
//         });
//     }


//     extdir_lohi = has_extdir_or_ho(bcs.dataPtr(), ncomp, 1);
//     has_extdir_or_ho_lo = extdir_lohi.first;
//     has_extdir_or_ho_hi = extdir_lohi.second;
//     if ((has_extdir_or_ho_lo and domain_jlo >= vbx.smallEnd(1)-1) or
//         (has_extdir_or_ho_hi and domain_jhi <= vbx.bigEnd(1)))
//     {
//         amrex::ParallelFor(vbx, ncomp, [d_bcrec_ptr,q,domain_jlo,domain_jhi,vmac,yedge]
//         AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
//         {
//             yedge(i,j,k,n) = iamr_yedge_state_mol_extdir( i, j, k, n, q, vmac,
//                                                           d_bcrec_ptr,
//                                                           domain_jlo, domain_jhi );
//         });
//     }
//     else
//     {
//         amrex::ParallelFor(vbx, ncomp, [q,vmac,yedge]
//         AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
//         {
//             yedge(i,j,k,n) = iamr_yedge_state_mol( i, j, k, n, q, vmac );
//         });
//     }


// #if ( AMREX_SPACEDIM ==3 )

//     extdir_lohi = has_extdir_or_ho(bcs.dataPtr(), ncomp, 2);
//     has_extdir_or_ho_lo = extdir_lohi.first;
//     has_extdir_or_ho_hi = extdir_lohi.second;
//     if ((has_extdir_or_ho_lo and domain_klo >= wbx.smallEnd(2)-1) or
//         (has_extdir_or_ho_hi and domain_khi <= wbx.bigEnd(2)))
//     {
//         amrex::ParallelFor(wbx, ncomp, [d_bcrec_ptr,q,domain_klo,domain_khi,wmac,zedge]
//         AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
//         {
//             zedge(i,j,k,n) = iamr_zedge_state_mol_extdir( i, j, k, n, q, wmac,
//                                                           d_bcrec_ptr,
//                                                           domain_klo, domain_khi );
//         });
//     }
//     else
//     {
//         amrex::ParallelFor(wbx, ncomp, [q,wmac,zedge]
//         AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
//         {
//             zedge(i,j,k,n) = iamr_zedge_state_mol( i, j, k, n, q, wmac );
//         });
//     }

// #endif
// }


// #ifdef AMREX_USE_EB
#if 0
//
// Compute edge state on EB box
//
void
MOL::EB_ComputeEdgeState ( Box const& bx,
                           D_DECL( Array4<Real> const& xedge,
                                   Array4<Real> const& yedge,
                                   Array4<Real> const& zedge),
                           Array4<Real const> const& q,
                           const int ncomp,
                           D_DECL( Array4<Real const> const& umac,
                                   Array4<Real const> const& vmac,
                                   Array4<Real const> const& wmac),
                           const Box&       domain,
                           const Vector<BCRec>& bcs,
			   const        BCRec * d_bcrec_ptr,
                           D_DECL( Array4<Real const> const& fcx,
                                   Array4<Real const> const& fcy,
                                   Array4<Real const> const& fcz),
                           Array4<Real const> const& ccc,
                      Array4<EBCellFlag const> const& flag)
{

    D_TERM( const Box& ubx = amrex::surroundingNodes(bx,0);,
            const Box& vbx = amrex::surroundingNodes(bx,1);,
            const Box& wbx = amrex::surroundingNodes(bx,2););

    // ****************************************************************************
    // Predict to x-faces
    // ****************************************************************************

    // At an ext_dir boundary, the boundary value is on the face, not cell center.
    auto extdir_lohi = has_extdir_or_ho(bcs.dataPtr(), ncomp, 0);
    if ((extdir_lohi.first  and domain.smallEnd(0) >= ubx.smallEnd(0)-1) or
        (extdir_lohi.second and domain.bigEnd(0)  <= ubx.bigEnd(0)))
    {
      amrex::ParallelFor(ubx, ncomp, [d_bcrec_ptr,q,ccc,AMREX_D_DECL(fcx,fcy,fcz),flag,umac, xedge, domain]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           if (flag(i,j,k).isConnected(-1,0,0))
           {
               xedge(i,j,k,n) = iamr_eb_xedge_state_mol_extdir( D_DECL(i, j, k), n, q, umac,
								D_DECL(fcx,fcy,fcz),
								ccc, flag, d_bcrec_ptr, domain );
           }
           else
           {
               xedge(i,j,k,n) = covered_val;
           }
        });
    }
    else
    {
        amrex::ParallelFor(ubx, ncomp, [q,ccc,fcx,flag,umac,xedge]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           if (flag(i,j,k).isConnected(-1,0,0))
           {
               xedge(i,j,k,n) = iamr_eb_xedge_state_mol( D_DECL(i, j, k), n, q, umac, fcx, ccc, flag );
           }
           else
           {
               xedge(i,j,k,n) = covered_val;
           }
        });
    }


    // ****************************************************************************
    // Predict to y-faces
    // ****************************************************************************
    extdir_lohi = has_extdir_or_ho(bcs.dataPtr(), ncomp, 1);
    if ((extdir_lohi.first  and domain.smallEnd(1) >= vbx.smallEnd(1)-1) or
        (extdir_lohi.second and domain.bigEnd(1)   <= vbx.bigEnd(1)))
    {
        amrex::ParallelFor(vbx, ncomp, [d_bcrec_ptr,q,ccc,AMREX_D_DECL(fcx,fcy,fcz),flag,vmac,yedge,domain]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (flag(i,j,k).isConnected(0,-1,0))
            {
                yedge(i,j,k,n) = iamr_eb_yedge_state_mol_extdir( D_DECL(i, j, k), n, q, vmac,
								 AMREX_D_DECL(fcx,fcy,fcz), ccc,
                                                                 flag, d_bcrec_ptr, domain );
            }
            else
            {
                yedge(i,j,k,n) = covered_val;
            }
        });
    }
    else
    {
        amrex::ParallelFor(vbx, ncomp, [q,ccc,fcy,flag,vmac,yedge]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (flag(i,j,k).isConnected(0,-1,0))
            {
                yedge(i,j,k,n) = iamr_eb_yedge_state_mol( D_DECL(i, j, k), n, q, vmac, fcy, ccc, flag );
            }
            else
            {
                yedge(i,j,k,n) = covered_val;
            }
        });
    }



#if ( AMREX_SPACEDIM == 3 )

    // ****************************************************************************
    // Predict to z-faces
    // ****************************************************************************
    extdir_lohi =  has_extdir_or_ho(bcs.dataPtr(), ncomp, 2);
    if ((extdir_lohi.first  and domain.smallEnd(2) >= wbx.smallEnd(2)-1) or
        (extdir_lohi.second and domain.bigEnd(2)   <= wbx.bigEnd(2)))
    {
        amrex::ParallelFor(wbx, ncomp, [d_bcrec_ptr,q,ccc,AMREX_D_DECL(fcx,fcy,fcz),flag,wmac,zedge,domain]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (flag(i,j,k).isConnected(0,0,-1))
            {
                zedge(i,j,k,n) = iamr_eb_zedge_state_mol_extdir( i, j, k, n, q, wmac,
								 AMREX_D_DECL(fcx,fcy,fcz), ccc,
                                                                 flag, d_bcrec_ptr, domain );
            }
            else
            {
                zedge(i,j,k,n) = covered_val;
            }
        });
    }
    else
    {
        amrex::ParallelFor(wbx, ncomp, [q,ccc,fcz,flag,wmac,zedge]
        AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            if (flag(i,j,k).isConnected(0,0,-1))
            {
                zedge(i,j,k,n) = iamr_eb_zedge_state_mol( i, j, k, n, q, wmac, fcz, ccc, flag );
            }
            else
            {
                zedge(i,j,k,n) = covered_val;
            }
        });
    }

#endif
}

#endif //End of AMREX_USE_EB
