#include "sparse_matrix.h"
#include "cvode_functions.h"
#include "flame_params.h"

int sign(const double x)
{
    return (x > 0) ? 1 :
           (x < 0) ? -1
                   : 0;
}

int sign(const int x)
{
    return (x > 0) ? 1 :
           (x < 0) ? -1
                   : 0;
}

static double FindMaximumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_max);

static double FindMinimumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_min);

// Main RHS function
int ConstPressureFlame(realtype t,
		       N_Vector y,
		       N_Vector ydot,
		       void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  long int Nlocal = num_local_points*num_states;

  // MPI calls are in Local, no need for Comm function
  ConstPressureFlameLocal(Nlocal, t, y, ydot, user_data);

  return 0;
}

// RHS function
int ConstPressureFlameLocal(long int nlocal,
			    realtype t,
			    N_Vector y,
			    N_Vector ydot,
			    void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;
  double *y_ptr    = NV_DATA_P(y);   // caution: assumes realtype == double
  double *ydot_ptr = NV_DATA_P(ydot); // caution: assumes realtype == double

  const int num_local_points = params->num_local_points_;
  const int num_total_points = params->z_.size();
  const int num_states  = params->reactor_->GetNumStates();
  const int num_species = params->reactor_->GetNumSpecies();
  const int soot_idx_start_ = params->reactor_->GetSootIdxStart();
  const int total_soot_vars = params->reactor_->GetNumSectionalTotal();
  const int num = params->reactor_->GetNumSpecies();
  const int num_local_states = num_local_points*num_states;
  const int convective_scheme_type = params->convective_scheme_type_;
  int my_pe = params->my_pe_;
  int npes  = params->npes_;
  int nover = params->nover_;

  const double ref_temperature = params->ref_temperature_;
  const double ref_momentum = params->ref_momentum_;
  const bool finite_separation = params->parser_->finite_separation();

  std::vector<double> enthalpies;
  enthalpies.assign(num_species,0.0);

  // Splitting RHS into chemistry, convection, and diffusion terms
  // for readability/future use
  std::vector<double> rhs_chem, rhs_conv, rhs_diff;
  rhs_chem.assign(num_local_points*num_states,0.0);
  rhs_conv.assign(num_local_points*num_states,0.0);
  rhs_diff.assign(num_local_points*num_states,0.0);

  double cp_flux_sum, mass_fraction_sum;
  double relative_volume_j;
  int transport_error;

  double local_sum;
  double local_max;

  double thermal_diffusivity, continuity_error;

  // set the derivative to zero
  for(int j=0; j<num_local_states; ++j) {
    ydot_ptr[j] = 0.0;
  }

  // compute the constant pressure reactor source term
  // using Zero-RK
  for(int j=0; j<num_local_points; ++j) {
    params->reactor_->SetViscosity(params->mixture_viscosity_[j]);
    params->reactor_->GetTimeDerivativeLimiter(t,
					       &y_ptr[j*num_states],
					       &params->step_limiter_[0],
					       &rhs_chem[j*num_states]);
  }

  //--------------------------------------------------------------------------
  // Perform parallel communications
  MPI_Comm comm = params->comm_;
  MPI_Status status;
  long int dsize = num_states*nover;

  std::vector<double> dz, dzm, inv_dz, inv_dzm;
  dz.assign( num_local_points+(2*nover), 0.0);
  dzm.assign( num_local_points+(2*nover), 0.0);
  inv_dz.assign( num_local_points+(2*nover), 0.0);
  inv_dzm.assign( num_local_points+(2*nover), 0.0);

  // Copy y_ptr data into larger arrays
  for (int j=0; j<num_states*num_local_points; ++j) {
    params->y_ext_[num_states*nover + j] = y_ptr[j];
  }
  for (int j=0; j<num_local_points; ++j) {
    params->mass_flux_ext_[nover+j] = params->mass_flux_[j];
  }

  for (int j=0; j<num_local_points+2*nover; ++j) {
    dz[j] = params->dz_local_[j];
    dzm[j] = params->dzm_local_[j];
    inv_dz[j] = params->inv_dz_local_[j];
    inv_dzm[j] = params->inv_dzm_local_[j];
  }

  // Update ghost cells with send/receive
  int nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  int nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&params->y_ext_[nover*num_states], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->y_ext_[num_states*(num_local_points+nover)], dsize, PVEC_REAL_MPI_TYPE,
               nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&params->y_ext_[num_states*num_local_points], dsize,
               PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->y_ext_[0], dsize, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  // Apply boundary conditions
  // First proc: fuel conditions in ghost cells
  if (my_pe ==0) {
    for(int j=0; j<nover; ++j) {
      if(params->flame_type_ == 0) {
        for(int k=0; k<num_species; ++k) {
          params->y_ext_[j*num_states + k] = params->fuel_mass_fractions_[k];
        }
        params->y_ext_[j*num_states+num_species]   = params->fuel_relative_volume_;
      } else if (params->flame_type_ == 1 || params->flame_type_ == 2) {
        for(int k=0; k<num_species; ++k) {
          params->y_ext_[j*num_states + k] = params->inlet_mass_fractions_[k];
        }
        params->y_ext_[j*num_states+num_species]   = params->inlet_relative_volume_;
      }
      params->y_ext_[j*num_states+num_species+1] = params->fuel_temperature_;
      if(finite_separation) {
        params->y_ext_[j*num_states+num_species+2] = 0.0;
        params->P_left_ = params->y_ext_[nover*num_states+num_species+3];
        params->y_ext_[j*num_states+num_species+3] = params->P_left_;
      } else {
        params->y_ext_[j*num_states+num_species+2] =
          params->y_ext_[nover*num_states+num_species+2];//zero gradient
      }

      // Zero soot at left BC
      for(int k=0; k<total_soot_vars; ++k) {
        params->y_ext_[j*num_states + soot_idx_start_ + k] = 0.0;  
      }

      params->mass_flux_ext_[j] = params->mass_flux_fuel_;
    }
  }
  MPI_Bcast(&params->P_left_, 1, MPI_DOUBLE, 0, comm);

  // Last proc: oxidizer conditions in ghost cells
  if (my_pe == npes-1) {
    for(int j=num_local_points+nover; j<num_local_points+2*nover; ++j) {
      if(params->flame_type_ == 1) { //zero gradient on Y, 1/rho, T, G
        for(int k=0; k<num_species; ++k) {
          params->oxidizer_mass_fractions_[k] =
            params->y_ext_[(num_local_points+nover-1)*num_states+k];
        }
        params->oxidizer_relative_volume_ =
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species];
        params->oxidizer_temperature_ =
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species+1];
      }
      for(int k=0; k<num_species; ++k) {
        params->y_ext_[j*num_states + k] = params->oxidizer_mass_fractions_[k];
      }
      params->y_ext_[j*num_states+num_species] = params->oxidizer_relative_volume_;
      params->y_ext_[j*num_states+num_species+1] = params->oxidizer_temperature_;
      if(finite_separation) {
        if(params->flame_type_ == 0 || params->flame_type_ == 2) {
          params->y_ext_[j*num_states+num_species+2] = 0.0;
        } else if (params->flame_type_ == 1) {
          params->G_right_ = params->y_ext_[(num_local_points+nover-1)*num_states+num_species+2];
          params->y_ext_[j*num_states+num_species+2] = params->G_right_;
        }
        params->P_right_ = params->y_ext_[(num_local_points+nover-1)*num_states+num_species+3];
        params->y_ext_[j*num_states+num_species+3] = params->P_right_;

      } else {
        params->y_ext_[j*num_states+num_species+2] =
          params->y_ext_[(num_local_points+nover-1)*num_states+num_species+2];//dG/dx=0
      }
      // Zero soot at right BC
      for(int k=0; k<total_soot_vars; ++k) {
        params->y_ext_[j*num_states + soot_idx_start_ + k] = 0.0;
      }
      params->mass_flux_ext_[j] = params->mass_flux_oxidizer_;
    }
  }
  MPI_Bcast(&params->P_right_, 1, MPI_DOUBLE, npes-1, comm);
  if(params->flame_type_ == 1) {
    MPI_Bcast(&params->G_right_, 1, MPI_DOUBLE, npes-1, comm);
    MPI_Bcast(&params->oxidizer_temperature_, 1, MPI_DOUBLE, npes-1, comm);
    MPI_Bcast(&params->oxidizer_relative_volume_, 1, MPI_DOUBLE, npes-1, comm);
    MPI_Bcast(&params->oxidizer_mass_fractions_[0], num_species, MPI_DOUBLE, npes-1, comm);
  }

  //--------------------------------------------------------------------------
  // Compute mass flux -- Integrate continuity equation
  // dmdot/dx = -beta*rho*G -> mdot = int(-beta*rho*G, dx)
  const double RuTref_p = params->reactor_->GetGasConstant()*
    params->ref_temperature_/params->parser_->pressure();

  // Compute Stagnation plane location
  // ONLY WORKS IN SERIAL FOR NOW
  double *gbuf, *vbuf, *mbuf, *vloc, *gloc;
  gloc = (double *)malloc(num_local_points*sizeof(double));
  vloc = (double *)malloc(num_local_points*sizeof(double));
  for(int j=0; j<num_local_points; j++) {
    gloc[j] = y_ptr[j*num_states + num_species + 2]*ref_momentum;
    vloc[j] = y_ptr[j*num_states + num_species];
    //printf("j: %d, gloc: %5.3e, vloc: %5.3e\n", my_pe*num_local_points+j, gloc[j], vloc[j]);
  }
  //if(my_pe == 0) {
    gbuf = (double *)malloc(num_total_points*sizeof(double));
    vbuf = (double *)malloc(num_total_points*sizeof(double));
    mbuf = (double *)malloc(num_total_points*sizeof(double));
    //}
  // Gather global G, relative_volume, and mass flux on root
  dsize = num_local_points;
  // TODO: Replace MPI_Allgather with MPI_Gather!
  MPI_Allgather(gloc, dsize, PVEC_REAL_MPI_TYPE,
             gbuf, dsize, PVEC_REAL_MPI_TYPE, comm);
  MPI_Allgather(vloc, dsize, PVEC_REAL_MPI_TYPE,
             vbuf, dsize, PVEC_REAL_MPI_TYPE, comm);
  MPI_Allgather(&params->mass_flux_[0], dsize, PVEC_REAL_MPI_TYPE,
             mbuf, dsize, PVEC_REAL_MPI_TYPE, comm);

  if(my_pe == 0) {
    int jContBC = 0;
    if(params->flame_type_ == 0 || params->flame_type_ == 2) {
      // Find grid point closest to old stagnation point location
      int jStart = -1;
      for(int j=0; j<num_total_points; j++) {
        if(params->z_[j] > params->stagnation_plane_) {
          jStart = j;
          break;
        }
      }
      if(jStart == -1) jStart = num_total_points-1;
      jContBC = jStart;
      for(int i=1; jStart+i < num_total_points || jStart >= i; i++) {
        if(jStart+i < num_total_points && sign(mbuf[jStart+i]) != sign(mbuf[jStart])) {
          jContBC = jStart + i;
          break;
        } else if (jStart>=i && sign(mbuf[jStart-i]) != sign(mbuf[jStart-i+1])) {
          jContBC = jStart-i+1;
          break;
        }
      }
      if(jContBC == 0) {
        assert(mbuf[jContBC] <=0);
        params->stagnation_plane_ = params->z_[0] - mbuf[0]*params->dz_[0]/(mbuf[1]-mbuf[0]);
      } else {
        assert(mbuf[jContBC]*mbuf[jContBC-1] <=0 || mbuf[num_total_points-1] >=0); // test opposite sign
        params->stagnation_plane_ = params->z_[jContBC] - mbuf[jContBC]*params->dz_[jContBC]/
          (mbuf[jContBC]-mbuf[jContBC-1]);
      }
    } else if (params->flame_type_ == 1) {
      params->stagnation_plane_ = params->length_;
      jContBC = params->num_points_;
    }

    // Integrate continuity to compute mass flux
    double dVdx0;
    double beta = 1.0 + params->simulation_type_;

    if(finite_separation) {
      // Integrate left to right
      // First point uses fuel BC
      // use midpoint rhoG?
      int jj = 0;
      mbuf[jj] = params->mass_flux_fuel_ - (
        params->y_ext_[(jj+nover)*num_states+num_species+2]/
        params->y_ext_[(jj+nover)*num_states+num_species] +
        params->y_ext_[(jj+nover-1)*num_states+num_species+2]/
        params->y_ext_[(jj+nover-1)*num_states+num_species])
        *ref_momentum*params->dz_[jj];
      // Rest of domain
      for(int j=0; j<num_total_points-1; j++) {
        mbuf[j+1] = mbuf[j] - (gbuf[j]/vbuf[j]+gbuf[j+1]/vbuf[j+1])*params->dz_[j+1];
        /*
        printf("j: %d, mbuf: %5.3e, gbuf: %5.3e, vbuf: %5.3e, dz: %5.3e\n",
                   j,  mbuf[j+1],   gbuf[j+1],   vbuf[j+1],   params->dz_[j+1]);
        */
      }

    } else {
      // Integrate from both sides of stagnation plane and update BC values
      // Stagnation plane
      int jj = jContBC;
      dVdx0 = - beta*gbuf[jj]/vbuf[jj];
      if(jContBC != 0) {
        dVdx0 = 0.5*dVdx0 - 0.5*beta*gbuf[jj-1]/vbuf[jj-1];
      }
      mbuf[jj] = (params->z_[jj]-params->stagnation_plane_) * dVdx0;

      // Right of stagnation plane
      for(int j=jContBC; j<num_total_points-1; j++) {
        mbuf[j+1] = mbuf[j] - beta*gbuf[j]/vbuf[j]*params->dz_[j+1];
      }
      // Right BC
      jj = num_total_points-1;
      params->mass_flux_oxidizer_ = mbuf[jj] - beta*gbuf[jj]/vbuf[jj]*params->dz_[jj+1];

      // Left of stagnation plane
      if(jContBC != 0) {
        jj = jContBC-1;
        mbuf[jj] = (params->z_[jj] - params->stagnation_plane_) * dVdx0;
        for(int j=jContBC-1; j>0; j--) {
          mbuf[j-1] = mbuf[j] + beta*gbuf[j-1]/vbuf[j-1]*params->dz_[j];
        }
      }
      jj = 0;
      params->mass_flux_fuel_ = mbuf[jj] +
        beta*params->y_ext_[(jj+nover-1)*num_states+num_species+2]*ref_momentum/
        params->y_ext_[(jj+nover-1)*num_states+num_species]*params->dz_[jj];
    }
  } // End mass flux integration on root

  // Scatter mass flux to all procs
  MPI_Scatter(mbuf,
              dsize,
              PVEC_REAL_MPI_TYPE,
              &params->mass_flux_[0],
              dsize,
              PVEC_REAL_MPI_TYPE,
              0,
              comm);

  // Broadcast mass flux BCs and stagnation plane
  MPI_Bcast(&params->mass_flux_fuel_, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&params->mass_flux_oxidizer_, 1, MPI_DOUBLE, 0, comm);
  MPI_Bcast(&params->stagnation_plane_, 1, MPI_DOUBLE, 0, comm);

  //  Copy into extended vector
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    params->mass_flux_ext_[jext] = params->mass_flux_[j];
  }

  // Parallel communication of mass flux
  dsize = nover;
  nodeDest = my_pe-1;
  if (nodeDest < 0) nodeDest = npes-1;
  nodeFrom = my_pe+1;
  if (nodeFrom > npes-1) nodeFrom = 0;
  MPI_Sendrecv(&params->mass_flux_ext_[nover], dsize, PVEC_REAL_MPI_TYPE, nodeDest, 0,
               &params->mass_flux_ext_[num_local_points+nover], dsize,
               PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  nodeDest = my_pe+1;
  if (nodeDest > npes-1) nodeDest = 0;
  nodeFrom = my_pe-1;
  if (nodeFrom < 0) nodeFrom = npes-1;
  MPI_Sendrecv(&params->mass_flux_ext_[num_local_points], dsize,
               PVEC_REAL_MPI_TYPE, nodeDest, 0, &params->mass_flux_ext_[0],
               dsize, PVEC_REAL_MPI_TYPE, nodeFrom, 0, comm, &status);

  // Add BCs to mass_flux_ext
  // First proc: fuel conditions in ghost cells
  if (my_pe ==0) {
    for(int j=0; j<nover; ++j) {
      params->mass_flux_ext_[j] = params->mass_flux_fuel_;
    }
  }

  // Last proc: oxidizer conditions in ghost cells
  if (my_pe == npes-1) {
    for(int j=num_local_points+nover; j<num_local_points+2*nover; ++j) {
        params->mass_flux_ext_[j] = params->mass_flux_oxidizer_;
    }
  }

  //--------------------------------------------------------------------------
  // Compute the interior heat capacity, conductivity, and species mass fluxes.
  //--------------------------------------------------------------------------
  for(int j=0; j<num_local_points+1; ++j) {
    int jext = j + nover;

    // compute the upstream mid point state for the transport calculations
    for(int k=0; k<num_species; ++k) {

      // mid point mass fractions
      params->transport_input_.mass_fraction_[k] =
        0.5*(params->y_ext_[jext*num_states+k] + params->y_ext_[(jext-1)*num_states+k]);

      // mid point mass fraction gradient
      params->transport_input_.grad_mass_fraction_[k] = inv_dz[jext]*
	(params->y_ext_[jext*num_states+k] - params->y_ext_[(jext-1)*num_states+k]);
    }

    // mid point temperature
    params->transport_input_.temperature_ = 0.5*ref_temperature*
      (params->y_ext_[jext*num_states+num_species+1] +
       params->y_ext_[(jext-1)*num_states+num_species+1]);

    // mid point temperature gradient
    params->transport_input_.grad_temperature_[0] = inv_dz[jext]*ref_temperature*
      (params->y_ext_[jext*num_states+num_species+1] -
       params->y_ext_[(jext-1)*num_states+num_species+1]);

    // mixture specific heat at mid point. Species cp will be overwritten
    // for diffusion jacobian only
    params->mixture_specific_heat_mid_[j] =
      params->reactor_->GetMixtureSpecificHeat_Cp(
        params->transport_input_.temperature_,
        &params->transport_input_.mass_fraction_[0],
        &params->species_specific_heats_[num_species*j]);

    // Reset species cp
    for(int k=0; k<num_species; k++) {
      params->species_specific_heats_[num_species*j+k] = 0.0;
    }

    // specific heat at grid point j
    params->mixture_specific_heat_[j] =
      params->reactor_->GetMixtureSpecificHeat_Cp(
        ref_temperature*params->y_ext_[jext*num_states+num_species+1],
        &params->y_ext_[jext*num_states],
        &params->species_specific_heats_[num_species*j]);

    //mixture molecular mass at mid point
    // for frozen thermo only
    double mass_fraction_weight_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      mass_fraction_weight_sum +=
        params->inv_molecular_mass_[k]*params->transport_input_.mass_fraction_[k];
    }
    params->molecular_mass_mix_mid_[j] = 1.0/mass_fraction_weight_sum;

    // compute the conductivity at the upstream mid point (j-1/2)
    transport_error = params->trans_->GetMixtureConductivity(
      params->transport_input_,
      &params->thermal_conductivity_[j]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

    // compute the viscosity at the upstream mid point (j-1/2)
    transport_error = params->trans_->GetMixtureViscosity(
      params->transport_input_,
      &params->mixture_viscosity_[j]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

    // compute the species diffusion mass flux at the upstream mid point
    // always use corrected diffusion flux in unsteady solver
    transport_error = params->trans_->GetSpeciesMassFlux(
      params->transport_input_,
      num_species,
      &params->thermal_conductivity_[j],
      &params->mixture_specific_heat_mid_[j],
      &params->species_mass_flux_[j*num_species],
      &params->species_lewis_numbers_[j*num_species]);
    if(transport_error != transport::NO_ERROR) {
      return transport_error;
    }

    // Section-independent thermophoretic velocity
    params->soot_thermophoretic_coefficients_[j] =
      -params->thermophoretic_const_*params->mixture_viscosity_[j]
      / params->transport_input_.temperature_; // Cth*mu/T  (densities cancel out)

    // ------------------------------------------------------------------
    // SESC Brownian diffusion coefficients at face j, per soot bin.
    //   D_soot,k = kB*T*Cc(Kn_k)/(3*pi*mu*d_p,k)                 (Stokes-Einstein)
    //   Cc(Kn)  = 1 + Kn*[alpha + beta*exp(-gamma/Kn)]           (Cunningham slip)
    //   Kn_k    = 2*lambda / d_p,k
    //   lambda  = (mu/p)*sqrt(pi*R_u*T/(2*W))                    (Chapman-Enskog)
    // Cunningham coefficients are from ISO 15900:2009.
    // We store rho_face*D_soot,k so the interior diffusive flux is simply
    //   J = -(rho*D)_face * dY/dx.
    // ------------------------------------------------------------------
    if(total_soot_vars > 0) {
      const std::vector<double>& soot_bin_diameters =
        params->reactor_->GetSectionsDiameter();
      const double kB       = 1.380649e-23;                 // [J/K]
      const double R_u      = params->reactor_->GetGasConstant();
      const double pi_local = 4.0*atan(1.0);
      // ISO 15900:2009 Cunningham slip-correction coefficients
      const double cc_alpha = 1.165;
      const double cc_beta  = 0.483;
      const double cc_gamma = 0.997;
      const double T_face   = params->transport_input_.temperature_;
      const double p_face   = params->transport_input_.pressure_;
      const double mu_face  = params->mixture_viscosity_[j];
      const double W_face   = params->molecular_mass_mix_mid_[j];
      const double rho_face = p_face*W_face/(R_u*T_face);
      const double mfp      =
        (mu_face/p_face)*sqrt(pi_local*R_u*T_face/(2.0*W_face));
      const double D_prefactor = kB*T_face/(3.0*pi_local*mu_face);
      for(int k=0; k<total_soot_vars; ++k) {
        const double dp  = soot_bin_diameters[k];
        const double Kn  = 2.0*mfp/dp;
        const double Cc  = 1.0 + Kn*(cc_alpha + cc_beta*exp(-cc_gamma/Kn));
        const double D_k = D_prefactor*Cc/dp;
        params->soot_diffusion_coefficients_[j*total_soot_vars + k] =
          rho_face*D_k;
      }
    }

  } // for j<num_local_points+1

  //--------------------------------------------------------------------------
  // Compute convective and diffusive terms for species, temperature, and momentum
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    int jglobal = j + my_pe*num_local_points;

    relative_volume_j  = params->y_ext_[jext*num_states+num_species];

    double a=0,b=0,c=0,d=0,e=0;; //coefficients of j+2, j+1, j, j-1, j-2 terms
    if(convective_scheme_type == 0) {
      // First order upwind
      if(params->mass_flux_ext_[jext]*relative_volume_j > 0) {
        a = 0;
        b = 0;
        c =  inv_dz[jext];
        d = -inv_dz[jext];
        e = 0;
      } else {
        a = 0;
        b =  inv_dz[jext+1];
        c = -inv_dz[jext+1];
        d = 0;
        e = 0;
      }
    } else if(convective_scheme_type == 1) {
      // Second order upwind
      if (params->mass_flux_ext_[jext]*relative_volume_j > 0) {
        // Use points upstream
        a = 0;
        b = 0;
        c = inv_dz[jext] + 1.0/(dz[jext]+dz[jext-1]);
	d = -(dz[jext]+dz[jext-1])/(dz[jext]*dz[jext-1]);
        e = dz[jext]/dz[jext-1]/(dz[jext]+dz[jext-1]);
      } else {
        // Use points downstream
        a = -dz[jext+1]/dz[jext+2]/(dz[jext+1]+dz[jext+2]);
        b = inv_dz[jext+1] + inv_dz[jext+2];
        c = -inv_dz[jext+1] - 1.0/(dz[jext+1]+dz[jext+2]);
        d = 0;
        e = 0;
      }
    } else if(convective_scheme_type == 2) {
      // Centered
      a = 0;
      b = dz[jext]/dz[jext+1]/(dz[jext]+dz[jext+1]);
      c = (dz[jext+1]-dz[jext])/dz[jext+1]/dz[jext];
      d = -dz[jext+1]/dz[jext]/(dz[jext]+dz[jext+1]);
      e = 0;
    } else {
      cerr << "Undefined convective scheme \n";
      MPI_Finalize();
      exit(0);
    }

    // compute the species mass fraction advection and diffusion
    for(int k=0; k<num_species; ++k) {

      rhs_conv[j*num_states+k] -= relative_volume_j*
	(a*params->y_ext_[(jext+2)*num_states+k] +
	 b*params->y_ext_[(jext+1)*num_states+k] +
	 c*params->y_ext_[ jext  *num_states+k] +
	 d*params->y_ext_[(jext-1)*num_states+k] +
	 e*params->y_ext_[(jext-2)*num_states+k]);

      rhs_diff[j*num_states+k] -= relative_volume_j*inv_dzm[jext]*
	( params->species_mass_flux_[num_species*(j+1)+k]
	 -params->species_mass_flux_[num_species*j+k]);
    }


    // compute the species specific heat diffusive flux sum
    cp_flux_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      cp_flux_sum += params->species_specific_heats_[num_species*j+k]*
	0.5*(params->species_mass_flux_[num_species*j+k]+
	     params->species_mass_flux_[num_species*(j+1)+k]);
    }

    // Compute the temperature advection (will be multiplied my mass flux)
    rhs_conv[j*num_states+num_species+1] -= relative_volume_j*
	(a*params->y_ext_[(jext+2)*num_states+num_species+1] +
	 b*params->y_ext_[(jext+1)*num_states+num_species+1] +
	 c*params->y_ext_[ jext   *num_states+num_species+1] +
	 d*params->y_ext_[(jext-1)*num_states+num_species+1] +
	 e*params->y_ext_[(jext-2)*num_states+num_species+1]);

    rhs_diff[j*num_states+num_species+1] -= relative_volume_j*
      cp_flux_sum/params->mixture_specific_heat_[j]*
	(a*params->y_ext_[(jext+2)*num_states+num_species+1] +
	 b*params->y_ext_[(jext+1)*num_states+num_species+1] +
	 c*params->y_ext_[ jext   *num_states+num_species+1] +
	 d*params->y_ext_[(jext-1)*num_states+num_species+1] +
	 e*params->y_ext_[(jext-2)*num_states+num_species+1]);

    // Add the thermal conductivity contribution to dT[j]/dt
    rhs_diff[j*num_states+num_species+1] +=
      (relative_volume_j*inv_dzm[jext]/params->mixture_specific_heat_[j])*
      (params->thermal_conductivity_[j+1]*inv_dz[jext+1]*
       (params->y_ext_[(jext+1)*num_states+num_species+1] -
        params->y_ext_[jext*num_states+num_species+1])
       -params->thermal_conductivity_[j]*inv_dz[jext]*
       (params->y_ext_[jext*num_states+num_species+1] -
        params->y_ext_[(jext-1)*num_states+num_species+1]));

    // Momentum equation
    // Compute the momentum advection term (will be multiplied by mass flux)
    rhs_conv[j*num_states+num_species+2] -= relative_volume_j*
      (a*params->y_ext_[(jext+2)*num_states+num_species+2] +
       b*params->y_ext_[(jext+1)*num_states+num_species+2] +
       c*params->y_ext_[ jext   *num_states+num_species+2] +
       d*params->y_ext_[(jext-1)*num_states+num_species+2] +
       e*params->y_ext_[(jext-2)*num_states+num_species+2]);

    // Compute momentum strain term P
    if(finite_separation) {
      rhs_diff[j*num_states+num_species+2] -= params->y_ext_[jext*num_states+num_species+3]
        *relative_volume_j;
    } else {
      //rhoInf*a^2/beta^2
      rhs_diff[j*num_states+num_species+2] += params->strain_rate_*params->strain_rate_/
        (1.0+params->simulation_type_)/(1.0+params->simulation_type_)*
        relative_volume_j/params->oxidizer_relative_volume_/ref_momentum;
    }

    // G*G
    rhs_diff[j*num_states+num_species+2] -= params->y_ext_[jext*num_states+num_species+2]*
      params->y_ext_[jext*num_states+num_species+2]*ref_momentum;

    // Compute momentum diffusion term
    rhs_diff[j*num_states+num_species+2] +=
      (inv_dzm[jext]*relative_volume_j)*
      (params->mixture_viscosity_[j+1]*inv_dz[jext+1]*
       (params->y_ext_[(jext+1)*num_states+num_species+2] -
        params->y_ext_[ jext   *num_states+num_species+2])
       -params->mixture_viscosity_[j]*inv_dz[jext]*
       (params->y_ext_[ jext   *num_states+num_species+2] -
        params->y_ext_[(jext-1)*num_states+num_species+2]));

    // Pstrain equation
    // Pstrain is imposed for infinite separation
    if(finite_separation) {
      if (jglobal == (npes*num_local_points-1)) { //lastpoint: dV/dx+beta*rho*G
        rhs_diff[j*num_states+num_species+3] =
          (params->mass_flux_ext_[jext+1]-params->mass_flux_ext_[jext])*inv_dz[jext+1] +
          (params->y_ext_[(jext+1)*num_states+num_species+2]/
           params->y_ext_[(jext+1)*num_states+num_species] +
           params->y_ext_[jext*num_states+num_species+2]/
           params->y_ext_[(jext+1)*num_states+num_species])*ref_momentum;
      } else { // dP/dx
        rhs_diff[j*num_states+num_species+3] =
          (params->y_ext_[(jext+1)*num_states+num_species+3] -
           params->y_ext_[jext*num_states+num_species+3])*inv_dz[jext+1]*100;
        // P adjusts faster with *100 but maybe it makes the ODEs stiffer/harder to solve?
      }
    }

    for(int k=0; k<total_soot_vars; ++k) {
      // Soot values at the midpoints
      double soot_jp1 = 0.5*(params->y_ext_[(jext+1)*num_states+soot_idx_start_+k]
			     + params->y_ext_[(jext)*num_states+soot_idx_start_+k]);
      double soot_j   = 0.5*(params->y_ext_[jext*num_states+soot_idx_start_+k]
			     + params->y_ext_[(jext-1)*num_states+soot_idx_start_+k]);

      double flux_jp1 = soot_jp1 * params->soot_thermophoretic_coefficients_[j+1] // Y_{s,l}*rho*-Cth*mu/rho/T
	* inv_dz[jext+1] * ref_temperature
	* (params->y_ext_[(jext+1)*num_states+num_species+1] -
	   params->y_ext_[jext*num_states+num_species+1]); // dT/dx

      double flux_j   = soot_j * params->soot_thermophoretic_coefficients_[j] // Y_{s,l}*rho*-Cth*mu/rho/T
	* inv_dz[jext] * ref_temperature
	* (params->y_ext_[jext*num_states+num_species+1] -
	   params->y_ext_[(jext-1)*num_states+num_species+1]); // dT/dx

      // Soot convection term
      rhs_conv[j*num_states + soot_idx_start_ + k] -= relative_volume_j*
	(a*params->y_ext_[(jext+2)*num_states + soot_idx_start_ + k] +
	 b*params->y_ext_[(jext+1)*num_states + soot_idx_start_ + k] +
	 c*params->y_ext_[ jext  *num_states + soot_idx_start_ + k] +
	 d*params->y_ext_[(jext-1)*num_states + soot_idx_start_ + k] +
	 e*params->y_ext_[(jext-2)*num_states + soot_idx_start_ + k]);
      // Soot thermophoretic term
      rhs_diff[j*num_states + soot_idx_start_ + k] -= (relative_volume_j*inv_dzm[jext])*
	(flux_jp1-flux_j);

      // ------------------------------------------------------------------
      // SESC Brownian diffusion (ISO 15900:2009 slip correction).
      // soot_diffusion_coefficients_ stores rho_face*D_soot,k at each face.
      // Fickian mass flux at face:  J = -rho*D * dY/dx.
      // dY/dt contribution: rhs_diff -= rel_vol * inv_dzm * (J_right - J_left)
      //                   =  rel_vol * div(rho*D*grad Y)
      //                   =  (1/rho) * div(rho*D*grad Y)
      // (same sign convention as species diffusion, cvode_functions.cpp:566-568)
      // ------------------------------------------------------------------
      const double rhoD_right =
        params->soot_diffusion_coefficients_[(j+1)*total_soot_vars + k];
      const double rhoD_left =
        params->soot_diffusion_coefficients_[ j   *total_soot_vars + k];
      const double diff_flux_right = -rhoD_right*inv_dz[jext+1]*
        (params->y_ext_[(jext+1)*num_states + soot_idx_start_ + k]
        -params->y_ext_[ jext   *num_states + soot_idx_start_ + k]);
      const double diff_flux_left  = -rhoD_left*inv_dz[jext]*
        (params->y_ext_[ jext   *num_states + soot_idx_start_ + k]
        -params->y_ext_[(jext-1)*num_states + soot_idx_start_ + k]);
      rhs_diff[j*num_states + soot_idx_start_ + k] -=
        relative_volume_j*inv_dzm[jext]*(diff_flux_right - diff_flux_left);
    }
  } // for(int j=0; j<num_local_points; ++j) // loop computing rhs

  // -------------------------------------------------------------------
  // Store residual breakdown if requested by monitor
  // -------------------------------------------------------------------
  if(params->compute_residual_breakdown_) {
    for(int j=0; j<num_local_points*num_states; ++j) {
      params->monitor_rhs_chem_[j] = rhs_chem[j];
      params->monitor_rhs_conv_[j] = rhs_conv[j];
      params->monitor_rhs_diff_[j] = rhs_diff[j];
    }
    // Note: rhs_conv will be multiplied by mass_flux below,
    // so store the final convective contribution instead
    for(int j=0; j<num_local_points; ++j) {
      for(int k=0; k<num_states; ++k) {
	params->monitor_rhs_conv_[j*num_states+k] *= params->mass_flux_[j];
      }
    }
  }

  // -------------------------------------------------------------------------
  // Compute the rate of change of the relative volume using the ideal
  // gas equation of state, and the FINAL derivatives of temperature and
  // mass fractions.
  //
  // dv/dt = v/T * dT/dt + RuT/p * \sum_i (1/mw[i] * dy[i]/dt)
  // dv/dt   current units [m^3/kg/s]
  for(int j=0; j<num_local_points; ++j) {
    int rvol_id  = j*num_states+num_species; // relative volume index of pt j
    int temp_id  = rvol_id+1;                // temperature index of pt j
    int mom_id   = rvol_id+2;                // momentum index of pt j

    mass_fraction_sum = 0.0;
    for(int k=0; k<num_species; ++k) {
      ydot_ptr[j*num_states+k] = rhs_conv[j*num_states+k]*params->mass_flux_[j]
        + rhs_chem[j*num_states+k] + rhs_diff[j*num_states+k];

      mass_fraction_sum += params->inv_molecular_mass_[k]*ydot_ptr[j*num_states+k];
    }

    ydot_ptr[temp_id] = rhs_conv[temp_id]*params->mass_flux_[j]
      + rhs_chem[temp_id] + rhs_diff[temp_id];

    ydot_ptr[rvol_id] = y_ptr[rvol_id]*ydot_ptr[temp_id]/y_ptr[temp_id] +
      RuTref_p*y_ptr[temp_id]*mass_fraction_sum;

    ydot_ptr[mom_id] = rhs_conv[mom_id]*params->mass_flux_[j] + rhs_diff[mom_id];

    if(finite_separation) {
      int strain_id = rvol_id+3;               // strain index of pt j
      ydot_ptr[strain_id] = rhs_diff[strain_id];
    }

    for(int k=0; k<total_soot_vars; ++k) {
      ydot_ptr[j*num_states + soot_idx_start_ + k] =
	rhs_conv[j*num_states + soot_idx_start_ + k]*params->mass_flux_[j]
	+ rhs_chem[j*num_states + soot_idx_start_ + k]
	+ rhs_diff[j*num_states + soot_idx_start_ + k];
    }

  }

  // -------------------------------------------------------------------------
  // For output to the screen/logfile
  params->mass_change_ = 0.0;

  // Compute fuel burning rate/laminar flame speed = int(omega_F)/rho_u/YF_u
  double sum_omega_F = 0.0;
  int num_fuel_species = params->fuel_species_id_.size();
  local_sum = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    for(int k=0; k<num_fuel_species; ++k) {
      local_sum -= rhs_chem[j*num_states + params->fuel_species_id_[k]]*
        dzm[jext]/y_ptr[j*num_states + num_species];
    }
  }
  MPI_Allreduce(&local_sum,&sum_omega_F,1,PVEC_REAL_MPI_TYPE,MPI_SUM,comm);
  double sum_inlet_fuel_mass_fractions = 0.0;
  if(params->flame_type_ == 1 || params->flame_type_ == 2) {
    // premixed flame
    for(int k=0; k<num_fuel_species; ++k) {
      sum_inlet_fuel_mass_fractions += params->inlet_mass_fractions_[params->fuel_species_id_[k]];
    }
    sum_omega_F /= sum_inlet_fuel_mass_fractions/params->inlet_relative_volume_;
  } else {
    //diffusion flame
    sum_inlet_fuel_mass_fractions = 1.0;
    sum_omega_F /= sum_inlet_fuel_mass_fractions/params->fuel_relative_volume_;
  }
  params->flame_speed_ = sum_omega_F;


  // Compute characteristic strain rate
  // Compute normal strain rate (dv/dz)
  std::vector<double> strain_rate_abs, velocity;
  strain_rate_abs.assign(num_local_points, 0.0);
  velocity.assign(num_local_points, 0.0);
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    velocity[j] = params->y_ext_[jext*num_states+num_species]*params->mass_flux_ext_[jext];
    strain_rate_abs[j] = fabs(
      (params->y_ext_[(jext+1)*num_states+num_species]*params->mass_flux_ext_[jext+1] -
       velocity[j])*inv_dz[jext]);
  }

  if(finite_separation) {
    // Method 1: Sometimes fails at high strain rates
    /*
    // Find the max normal strain rate location
    double max_strain;
    int jglobal_max_strain;
    max_strain = FindMaximumParallel(num_local_points, &strain_rate_abs[0], &jglobal_max_strain);

    // Find the minimum velocity ahead of the max strain location
    struct { double value; int index;} in, out;
    in.value = 10000;
    in.index = 0;
    for(int j=0; j<num_local_points; ++j) {
      int jglobal = j + my_pe*num_local_points;
      if(in.value > velocity[j] and jglobal < jglobal_max_strain) {
        in.value = velocity[j];
        in.index = jglobal;
      }
    }
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
    int jglobal_min_vel = out.index;

    // Find the max strain rate ahead of the minimum velocity point
    in.value = -10000;
    in.index = 0;
    for(int j=0; j<num_local_points; ++j) {
      int jglobal = j + my_pe*num_local_points;
      if(in.value < strain_rate_abs[j] and jglobal < jglobal_min_vel) {
        in.value = strain_rate_abs[j];
        in.index = jglobal;
      }
    }
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
    params->strain_rate_ = out.value;
    */

    // Method 2: highest absolute value before "turnaround" point
    // ONLY WORKS IN SERIAL FOR NOW
    long int dsize;
    double *sbuf;
    if(my_pe == 0)
      sbuf = (double *)malloc(num_local_points*npes*sizeof(double));

    // Gather strain rate on root
    dsize = num_local_points;
    MPI_Gather(&strain_rate_abs[0],
               dsize,
               PVEC_REAL_MPI_TYPE,
               sbuf,
               dsize,
               PVEC_REAL_MPI_TYPE,
               0,
               comm);

    if(my_pe == 0) {
      params->strain_rate_ = -100000;
      for(int j=0; j<num_local_points*npes; ++j) {
        if(sbuf[j] > params->strain_rate_) {
          params->strain_rate_ = sbuf[j];
        } else {
          break;
        }
      }
    }
    MPI_Bcast(&params->strain_rate_, 1, MPI_DOUBLE, 0, comm);
  }

  // initialize the max variables used for explicit time step information
  // compute the max velocity from the mass flux and relative volume stored
  // in the state vector
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    if(velocity[j] > local_max) {
      local_max = velocity[j];
    }
  }
  MPI_Allreduce(&local_max,&params->max_velocity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

  double local_temperature;
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    local_temperature = ref_temperature*params->y_ext_[jext*num_states+num_species+1];
    if(local_temperature > local_max) {
      local_max = local_temperature;
    }
  }
  MPI_Allreduce(&local_max,&params->max_temperature_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

  double gradT;
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    int jext = j + nover;
    gradT = fabs(inv_dz[jext]*ref_temperature*
		 (params->y_ext_[ jext   *num_states+num_species+1] -
                  params->y_ext_[(jext-1)*num_states+num_species+1]));
    if (gradT > local_max) {
      local_max = gradT;
    }
  }
  MPI_Allreduce(&local_max,&params->flame_thickness_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);
  params->flame_thickness_ = (params->max_temperature_-params->fuel_temperature_)/
    params->flame_thickness_;

  // compute the max thermal diffusivity using the average value of the
  // conductivity and the up and downstream interfaces
  local_max = 0.0;
  for(int j=0; j<num_local_points; ++j) {
    thermal_diffusivity =
      fabs(0.5*(params->thermal_conductivity_[j]+
                params->thermal_conductivity_[j+1])*
           y_ptr[j*num_states+num_species]/params->mixture_specific_heat_[j]);
    if(thermal_diffusivity > local_max) {
      local_max = thermal_diffusivity;
    }
  }
  MPI_Allreduce(&local_max,&params->max_thermal_diffusivity_,1,PVEC_REAL_MPI_TYPE,MPI_MAX,comm);

  return 0;
}


int FlameMonitorFunction(void *cvode_mem, void *user_data)
{
  FlameParams *params = (FlameParams *)user_data;

  if(params->residual_verbosity_ < 1) return -1;

  const int num_local_points = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  const int num_species = params->reactor_->GetNumSpecies();
  const int soot_idx_start = params->reactor_->GetSootIdxStart();
  const int total_soot_vars = params->reactor_->GetNumSectionalTotal();
  const int my_pe = params->my_pe_;
  const long int num_local_states = num_local_points*num_states;
  const long int total_states = (long int)params->z_.size()*num_states;
  MPI_Comm comm = params->comm_;

  N_Vector monitor_ydot_;

  monitor_ydot_ = N_VNew_Parallel(comm, num_local_states, total_states);

  // -------------------------------------------------------------------
  // 1. Get CVODE integrator stats
  // -------------------------------------------------------------------
  long int nsteps, nfevals, nlinsetups, netfails;
  int qlast, qcur;
  realtype hinused, hlast, hcur, tcur;
  CVodeGetIntegratorStats(cvode_mem, &nsteps, &nfevals, &nlinsetups,
                          &netfails, &qlast, &qcur,
                          &hinused, &hlast, &hcur, &tcur);

  long int nniters, nncfails;
  CVodeGetNonlinSolvStats(cvode_mem, &nniters, &nncfails);

  // Compute deltas since last monitor call
  long int delta_steps  = nsteps  - params->monitor_nsteps_prev_;
  long int delta_fevals = nfevals - params->monitor_nfevals_prev_;
  long int delta_nni    = nniters - params->monitor_nniters_prev_;
  params->monitor_nsteps_prev_  = nsteps;
  params->monitor_nfevals_prev_ = nfevals;
  params->monitor_nniters_prev_ = nniters;

  // -------------------------------------------------------------------
  // 2. Get current state and evaluate RHS
  // -------------------------------------------------------------------
  N_Vector y_cur;
  CVodeGetCurrentState(cvode_mem, &y_cur);

  // Set flag so RHS stores chem/conv/diff breakkdown
  params->compute_residual_breakdown_ = true;
  ConstPressureFlame(tcur, y_cur, monitor_ydot_, user_data);
  params->compute_residual_breakdown_ = false;

  double *ydot_ptr = NV_DATA_P(monitor_ydot_);

  // -------------------------------------------------------------------
  // 3. Compute grouped residual norms (local, then MPI reduce)
  // -------------------------------------------------------------------

  // Local L_inf for total ydot
  double local_species_Linf = 0.0;
  double local_soot_Linf    = 0.0;
  double local_temp_Linf    = 0.0;
  double local_rvol_Linf    = 0.0;
  double local_mom_Linf     = 0.0;

  // Local L_inf for sub-terms (chem, conv, diff)
  double local_species_chem_Linf = 0.0, local_species_conv_Linf = 0.0, local_species_diff_Linf = 0.0;
  double local_soot_chem_Linf = 0.0, local_soot_conv_Linf = 0.0, local_soot_diff_Linf = 0.0;

  // Track which variable has the max residual
  struct { double value; int index; } species_max_local, soot_max_local;
  species_max_local.value = 0.0; species_max_local.index = 0;
  soot_max_local.value    = 0.0; soot_max_local.index    = 0;

  for(int j=0; j<num_local_points; ++j) {
    // --- Species ---
    for(int k=0; k<num_species; ++k) {
      int idx = j*num_states + k;
      double val = fabs(ydot_ptr[idx]);
      if(val > local_species_Linf) local_species_Linf = val;
      if(val > species_max_local.value) {
        species_max_local.value = val;
        species_max_local.index = k;  // species id (same across grid)
      }
      // Sub-terms
      double vc = fabs(params->monitor_rhs_chem_[idx]);
      double vv = fabs(params->monitor_rhs_conv_[idx]);
      double vd = fabs(params->monitor_rhs_diff_[idx]);
      if(vc > local_species_chem_Linf) local_species_chem_Linf = vc;
      if(vv > local_species_conv_Linf) local_species_conv_Linf = vv;
      if(vd > local_species_diff_Linf) local_species_diff_Linf = vd;
    }

    // --- Thermo/flow ---
    double val_rvol = fabs(ydot_ptr[j*num_states + num_species]);
    double val_temp = fabs(ydot_ptr[j*num_states + num_species + 1]);
    double val_mom  = fabs(ydot_ptr[j*num_states + num_species + 2]);
    if(val_rvol > local_rvol_Linf) local_rvol_Linf = val_rvol;
    if(val_temp > local_temp_Linf) local_temp_Linf = val_temp;
    if(val_mom  > local_mom_Linf)  local_mom_Linf  = val_mom;

    // --- Soot sections ---
    for(int k=0; k<total_soot_vars; ++k) {
      int idx = j*num_states + soot_idx_start + k;
      double val = fabs(ydot_ptr[idx]);
      if(val > local_soot_Linf) local_soot_Linf = val;
      if(val > soot_max_local.value) {
        soot_max_local.value = val;
        soot_max_local.index = k;
      }
      double vc = fabs(params->monitor_rhs_chem_[idx]);
      double vv = fabs(params->monitor_rhs_conv_[idx]);
      double vd = fabs(params->monitor_rhs_diff_[idx]);
      if(vc > local_soot_chem_Linf) local_soot_chem_Linf = vc;
      if(vv > local_soot_conv_Linf) local_soot_conv_Linf = vv;
      if(vd > local_soot_diff_Linf) local_soot_diff_Linf = vd;
    }
  }

  
  // Quick check on soot values
  double *y_ptr;
  CVodeGetCurrentState(cvode_mem, &y_cur);
  y_ptr = NV_DATA_P(y_cur);

  // Check soot state values
    double local_soot_min =  1.0e+300;
    double local_soot_max = -1.0e+300;
    double local_soot_absmax = 0.0;
    int local_min_bin = 0, local_max_bin = 0;
    
    for(int j=0; j<num_local_points; ++j) {
      for(int k=0; k<total_soot_vars; ++k) {
        double val = y_ptr[j*num_states + soot_idx_start + k];
        if(val < local_soot_min) { local_soot_min = val; local_min_bin = k; }
        if(val > local_soot_max) { local_soot_max = val; local_max_bin = k; }
        if(fabs(val) > local_soot_absmax) local_soot_absmax = fabs(val);
      }
    }

    double global_soot_min, global_soot_max, global_soot_absmax;
    MPI_Allreduce(&local_soot_min, &global_soot_min, 1,
                  MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_soot_max, &global_soot_max, 1,
                  MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_soot_absmax, &global_soot_absmax, 1,
                  MPI_DOUBLE, MPI_MAX, comm);
  if(total_soot_vars == 0) {
    global_soot_min = 0.0; global_soot_max = 0.0; global_soot_absmax = 0.0;
  }

  // Get soot process at peak soot location
  // -------------------------------------------------------------------
  // Soot process rate breakdown at peak soot grid point
  // -------------------------------------------------------------------
  if(total_soot_vars > 0) {
    // Find local grid point with maximum soot
    int j_max_soot = 0;
    double max_soot_val = 0.0;
    for(int j=0; j<num_local_points; ++j) {
      for(int k=0; k<total_soot_vars; ++k) {
        double val = fabs(y_ptr[j*num_states + soot_idx_start + k]);
        if(val > max_soot_val) {
          max_soot_val = val;
          j_max_soot = j;
        }
      }
    }

    // Find global max and which rank owns it
    struct { double value; int rank; } local_in, global_out;
    local_in.value = max_soot_val;
    local_in.rank  = my_pe;
    MPI_Allreduce(&local_in, &global_out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);

    // The rank that owns the peak soot point does the diagnostic
    if(my_pe == global_out.rank) {
      // Call GetTimeDerivativeLimiter on this specific grid point
      // to populate the Fortran last_* arrays for this point
      std::vector<double> temp_deriv(num_states, 0.0);
      params->reactor_->GetTimeDerivativeLimiter(
          0.0,  // time doesn't matter for chemistry
          &y_ptr[j_max_soot * num_states],
          &params->step_limiter_[0],
          &temp_deriv[0]);

      // Now retrieve the stored process rates from Fortran
      int n_bins, n_spec;
      std::vector<double> coag_rates(total_soot_vars);
      std::vector<double> sg_rates(total_soot_vars);
      std::vector<double> ox_rates(total_soot_vars);
      std::vector<double> cond_rates(total_soot_vars);
      double nuc_rate;
      int num_soot_species = params->reactor_->GetNumSpecies();
      std::vector<double> nuc_gas(num_soot_species);
      std::vector<double> sg_gas(num_soot_species);
      std::vector<double> ox_gas(num_soot_species);
      std::vector<double> cond_gas(num_soot_species);

      params->reactor_->GetLastSootRates(&coag_rates[0], &sg_rates[0], &ox_rates[0],
					 &cond_rates[0], &nuc_rate,
					 &nuc_gas[0], &sg_gas[0], &ox_gas[0], &cond_gas[0]);

      // Send to rank 0 for printing if needed
      // For simplicity, if this rank is also rank 0, print directly
      // Otherwise, send to rank 0
      // (For now, assume serial or that rank 0 has peak soot — 
      //  add MPI_Send/Recv if needed for multi-rank)
      if(my_pe == 0 && params->monitor_file_ != NULL) {
        FILE *mf = params->monitor_file_;
	    // DEBUG: check soot chemistry at one point


        // Grid point info
        fprintf(mf, "# SOOT PROCESS RATES at grid %d (peak soot Y=%10.3e):\n",
                j_max_soot, max_soot_val);

	fprintf(mf,"# SOOT CHEM DEBUG grid %d: bin0=%e bin1=%e\n",
	       j_max_soot, params->monitor_rhs_chem_[j_max_soot*num_states + soot_idx_start + 0],
	       params->monitor_rhs_chem_[j_max_soot*num_states + soot_idx_start + 1]);

        // Nucleation (only bin0)
        fprintf(mf, "#   Nucleation rate: %10.3e [#/m^3/s]\n", nuc_rate);

        // Per-bin process rates (first 10 bins)
        fprintf(mf, "#   bin   coagulation    surf_growth    oxidation      condensation   total\n");
        for(int k=0; k<std::min(10, (int)total_soot_vars); ++k) {
          double total = coag_rates[k] + sg_rates[k] + ox_rates[k] + cond_rates[k];
          if(k == 0) total += nuc_rate;
          fprintf(mf, "#   %3d   %12.4e   %12.4e   %12.4e   %12.4e   %12.4e\n",
                  k, coag_rates[k], sg_rates[k], ox_rates[k], cond_rates[k], total);
        }

        // Gas-phase feedback (find largest contributors)
        fprintf(mf, "#   Gas-phase feedback [kmol/m^3/s]:\n");
        fprintf(mf, "#   species   nucleation     surf_growth    oxidation      condensation   total\n");

        // Find top 5 species by total absolute feedback
        std::vector<std::pair<double,int>> gas_ranked(num_soot_species);
        for(int k=0; k<num_soot_species; ++k) {
          double total_abs = fabs(nuc_gas[k]) + fabs(sg_gas[k])
                           + fabs(ox_gas[k]) + fabs(cond_gas[k]);
          gas_ranked[k] = std::make_pair(total_abs, k);
        }
        std::partial_sort(gas_ranked.begin(),
                          gas_ranked.begin() + std::min(10, num_soot_species),
                          gas_ranked.end(),
                          std::greater<std::pair<double,int>>());

        for(int i=0; i<std::min(10, num_soot_species); ++i) {
          int k = gas_ranked[i].second;
          double total = nuc_gas[k] + sg_gas[k] + ox_gas[k] + cond_gas[k];
          fprintf(mf, "#   %-20s %12.4e   %12.4e   %12.4e   %12.4e   %12.4e\n",
                  params->reactor_->GetNameOfStateId(k),
                  nuc_gas[k], sg_gas[k], ox_gas[k], cond_gas[k], total);
        }

	long int nli;  // linear iterations
	CVodeGetNumLinIters(cvode_mem, &nli);
	long int delta_nli = nli - params->monitor_nli_prev_;
	params->monitor_nli_prev_ = nli;

	fprintf(mf, "# LINEAR: nli=%ld(+%ld)  nli/nni=%.1f\n",
		nli, delta_nli, 
		(double)delta_nli / (double)std::max(1L, delta_nni));

        // Also print the gas state at this point for reference
	fprintf(mf, "# DEBUG: total_soot_vars=%d  soot_idx_start=%d  num_states=%d\n",
	       total_soot_vars, soot_idx_start, num_states);
        fprintf(mf, "#   Gas state at peak soot: T=%10.3e K\n",
                params->ref_temperature_ * y_ptr[j_max_soot*num_states + num_species + 1]);

        // Print key species mass fractions
        const char* key_species[] = {"MassFraction_C2H2", "MassFraction_A4",
                                     "MassFraction_O2", "MassFraction_OH",
                                     "MassFraction_H", "MassFraction_H2O"};
        for(int i=0; i<6; ++i) {
          int id = params->reactor_->GetIdOfState(key_species[i]);
          if(id >= 0 && id < num_species) {
            fprintf(mf, "#     %s = %10.3e\n",
                    key_species[i], y_ptr[j_max_soot*num_states + id]);
          }
        }

        fflush(mf);
      }
    } // if this rank owns peak soot
  } // if total_soot_vars > 0

  // -------------------------------------------------------------------
  // SOOT PROCESS RATES: Write per-grid-point data for time-series plotting
  // -------------------------------------------------------------------
  if(total_soot_vars > 0) {

    // Open file on first call (rank 0 only)
    static FILE *soot_rates_file = NULL;
    static bool soot_rates_header_written = false;

    if(my_pe == 0 && soot_rates_file == NULL) {
      soot_rates_file = fopen("soot_process_rates.dat", "w");
      if(soot_rates_file == NULL) {
        fprintf(stderr, "ERROR: Cannot open soot_process_rates.dat\n");
      }
    }

    if(my_pe == 0 && soot_rates_file != NULL && !soot_rates_header_written) {
      fprintf(soot_rates_file, "# Soot process rates per grid point per monitor call\n");
      fprintf(soot_rates_file, "# Units: rates in [#/m^3/s]\n");
      fprintf(soot_rates_file, "# NT = %d soot bins\n", total_soot_vars);
      fprintf(soot_rates_file, "# num_grid_points = %d\n", (int)params->z_.size());
      fprintf(soot_rates_file, "#\n");
      fprintf(soot_rates_file, "# Column layout:\n");
      fprintf(soot_rates_file, "#   1: time [s]\n");
      fprintf(soot_rates_file, "#   2: grid index (global)\n");
      fprintf(soot_rates_file, "#   3: z position [m]\n");
      fprintf(soot_rates_file, "#   4: peak_svf_grid (global grid index of peak soot)\n");
      fprintf(soot_rates_file, "#   5: nuc_rate [#/m^3/s]\n");
      int col = 6;
      fprintf(soot_rates_file, "#   %d-%d: coag_rates[0..%d]\n",
              col, col + total_soot_vars - 1, total_soot_vars - 1);
      col += total_soot_vars;
      fprintf(soot_rates_file, "#   %d-%d: sg_rates[0..%d]\n",
              col, col + total_soot_vars - 1, total_soot_vars - 1);
      col += total_soot_vars;
      fprintf(soot_rates_file, "#   %d-%d: ox_rates[0..%d]\n",
              col, col + total_soot_vars - 1, total_soot_vars - 1);
      col += total_soot_vars;
      fprintf(soot_rates_file, "#   %d-%d: cond_rates[0..%d]\n",
              col, col + total_soot_vars - 1, total_soot_vars - 1);
      fprintf(soot_rates_file, "#   Total columns: %d\n", 5 + 4 * total_soot_vars);
      fprintf(soot_rates_file, "#\n");
      soot_rates_header_written = true;
    }

    // ------------------------------------------------------------------
    // Each rank computes process rates for its local grid points
    // ------------------------------------------------------------------
    // Per grid point, we store: nuc_rate + 4*NT values = 1 + 4*NT doubles
    const int vals_per_point = 1 + 4 * total_soot_vars;
    std::vector<double> local_rates(num_local_points * vals_per_point, 0.0);

    // Also find local peak SVF
    int local_peak_grid = 0;
    double local_peak_soot = 0.0;

    {
      std::vector<double> temp_deriv(num_states, 0.0);
      std::vector<double> coag_rates(total_soot_vars);
      std::vector<double> sg_rates(total_soot_vars);
      std::vector<double> ox_rates(total_soot_vars);
      std::vector<double> cond_rates(total_soot_vars);
      double nuc_rate_local;
      std::vector<double> nuc_gas(num_species);
      std::vector<double> sg_gas(num_species);
      std::vector<double> ox_gas(num_species);
      std::vector<double> cond_gas(num_species);

      for(int j = 0; j < num_local_points; ++j) {
        // Check for peak SVF at this point
        double soot_sum = 0.0;
        for(int k = 0; k < total_soot_vars; ++k) {
          soot_sum += y_ptr[j * num_states + soot_idx_start + k];
        }
        if(soot_sum > local_peak_soot) {
          local_peak_soot = soot_sum;
          local_peak_grid = params->npes_*0 + j;  // need global index — see below
        }

        // Evaluate chemistry to populate Fortran state
        params->reactor_->GetTimeDerivativeLimiter(
            0.0,
            &y_ptr[j * num_states],
            &params->step_limiter_[0],
            &temp_deriv[0]);

        // Retrieve process rates
        params->reactor_->GetLastSootRates(
            &coag_rates[0], &sg_rates[0], &ox_rates[0],
            &cond_rates[0], &nuc_rate_local,
            &nuc_gas[0], &sg_gas[0], &ox_gas[0], &cond_gas[0]);

        // Pack into local_rates buffer
        int offset = j * vals_per_point;
        local_rates[offset] = nuc_rate_local;
        for(int k = 0; k < total_soot_vars; ++k)
          local_rates[offset + 1 + k] = coag_rates[k];
        for(int k = 0; k < total_soot_vars; ++k)
          local_rates[offset + 1 + total_soot_vars + k] = sg_rates[k];
        for(int k = 0; k < total_soot_vars; ++k)
          local_rates[offset + 1 + 2*total_soot_vars + k] = ox_rates[k];
        for(int k = 0; k < total_soot_vars; ++k)
          local_rates[offset + 1 + 3*total_soot_vars + k] = cond_rates[k];
      }
    }

    // ------------------------------------------------------------------
    // Find global peak SVF grid index
    // ------------------------------------------------------------------
    // Convert local_peak_grid to global index
    int global_grid_offset = 0;  // need cumulative sum of local points before this rank
    int npes;
    MPI_Comm_size(comm, &npes);
    
    // Compute global offset for this rank's grid points
    std::vector<int> all_num_local(npes);
    MPI_Allgather(&num_local_points, 1, MPI_INT,
                  &all_num_local[0], 1, MPI_INT, comm);
    for(int r = 0; r < my_pe; ++r) {
      global_grid_offset += all_num_local[r];
    }
    int local_peak_global_idx = global_grid_offset + local_peak_grid;

    // MPI_MAXLOC to find which rank has the global peak
    struct { double value; int index; } peak_local, peak_global;
    peak_local.value = local_peak_soot;
    peak_local.index = local_peak_global_idx;
    MPI_Allreduce(&peak_local, &peak_global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
    int j_peak_svf_global = peak_global.index;

    // ------------------------------------------------------------------
    // Gather all rates to rank 0
    // ------------------------------------------------------------------
    // Gather counts and displacements
    std::vector<int> recvcounts(npes), displs(npes);
    int sendcount = num_local_points * vals_per_point;
    MPI_Gather(&sendcount, 1, MPI_INT, &recvcounts[0], 1, MPI_INT, 0, comm);

    if(my_pe == 0) {
      displs[0] = 0;
      for(int r = 1; r < npes; ++r)
        displs[r] = displs[r-1] + recvcounts[r-1];
    }

    int total_points = (int)params->z_.size();
    std::vector<double> global_rates;
    if(my_pe == 0) {
      global_rates.resize(total_points * vals_per_point);
    }

    MPI_Gatherv(&local_rates[0], sendcount, MPI_DOUBLE,
                my_pe == 0 ? &global_rates[0] : NULL,
                &recvcounts[0], &displs[0], MPI_DOUBLE,
                0, comm);

    // ------------------------------------------------------------------
    // Rank 0 writes all grid points
    // ------------------------------------------------------------------
    if(my_pe == 0 && soot_rates_file != NULL) {
      for(int j = 0; j < total_points; ++j) {
        int offset = j * vals_per_point;
        double nuc_rate_j = global_rates[offset];

        fprintf(soot_rates_file, "%16.8e %4d %12.6e %4d %14.6e",
                tcur, j, params->z_[j], j_peak_svf_global, nuc_rate_j);

        for(int k = 0; k < total_soot_vars; ++k)
          fprintf(soot_rates_file, " %14.6e", global_rates[offset + 1 + k]);
        for(int k = 0; k < total_soot_vars; ++k)
          fprintf(soot_rates_file, " %14.6e", global_rates[offset + 1 + total_soot_vars + k]);
        for(int k = 0; k < total_soot_vars; ++k)
          fprintf(soot_rates_file, " %14.6e", global_rates[offset + 1 + 2*total_soot_vars + k]);
        for(int k = 0; k < total_soot_vars; ++k)
          fprintf(soot_rates_file, " %14.6e", global_rates[offset + 1 + 3*total_soot_vars + k]);

        fprintf(soot_rates_file, "\n");
      }

      fprintf(soot_rates_file, "\n");
      fflush(soot_rates_file);
    }
  } // soot process rates output
  
  // -------------------------------------------------------------------
  // 4. MPI reductions for global norms
  // -------------------------------------------------------------------
  double global_species_Linf, global_soot_Linf;
  double global_temp_Linf, global_rvol_Linf, global_mom_Linf;
  double global_species_chem, global_species_conv, global_species_diff;
  double global_soot_chem, global_soot_conv, global_soot_diff;

  MPI_Allreduce(&local_species_Linf, &global_species_Linf, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_soot_Linf, &global_soot_Linf, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_temp_Linf, &global_temp_Linf, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_rvol_Linf, &global_rvol_Linf, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_mom_Linf, &global_mom_Linf, 1,
                MPI_DOUBLE, MPI_MAX, comm);

  MPI_Allreduce(&local_species_chem_Linf, &global_species_chem, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_species_conv_Linf, &global_species_conv, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_species_diff_Linf, &global_species_diff, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_soot_chem_Linf, &global_soot_chem, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_soot_conv_Linf, &global_soot_conv, 1,
                MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(&local_soot_diff_Linf, &global_soot_diff, 1,
                MPI_DOUBLE, MPI_MAX, comm);

  // Global max variable IDs (use MAXLOC)
  struct { double value; int index; } species_max_global, soot_max_global;
  MPI_Allreduce(&species_max_local, &species_max_global, 1,
                MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  MPI_Allreduce(&soot_max_local, &soot_max_global, 1,
                MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  // -------------------------------------------------------------------
  // 5. Print (rank 0 only)
  // -------------------------------------------------------------------
  if(my_pe == 0 && params->monitor_file_ != NULL) {
    FILE *mf = params->monitor_file_;
    
    // Solver stats
    fprintf(mf, "# MONITOR t=%12.5e  steps=%ld(+%ld)  hcur=%10.3e  q=%d  "
	    "fevals=%ld(+%ld)  nni=%ld(+%ld)  netf=%ld  nncf=%ld\n",
	    tcur, nsteps, delta_steps, hcur, qcur,
	    nfevals, delta_fevals, nniters, delta_nni,
	    netfails, nncfails);

    // Grouped residual norms
    fprintf(mf, "# RESIDUAL Linf:  species=%10.3e (%s)"
	    "  soot=%10.3e (bin%d)"
	    "  T=%10.3e  rvol=%10.3e  mom=%10.3e\n",
	    global_species_Linf,
	    params->reactor_->GetNameOfStateId(species_max_global.index),
	    global_soot_Linf,
	    soot_max_global.index,
	    global_temp_Linf,
	    global_rvol_Linf,
	    global_mom_Linf);

    // Sub-term breakdown for species
    fprintf(mf, "# SPECIES breakdown:  chem=%10.3e  conv=%10.3e  diff=%10.3e\n",
	    global_species_chem,
	    global_species_conv,
	    global_species_diff);

    // Sub-term breakdown for soot
    if(total_soot_vars > 0) {
      fprintf(mf, "# SOOT    breakdown:  chem=%10.3e  conv=%10.3e  diff=%10.3e\n",
	      global_soot_chem,
	      global_soot_conv,
	      global_soot_diff);

    if(my_pe == 0 && params->monitor_file_ != NULL) {
      fprintf(mf, "# SOOT STATE:  min=%10.3e  max=%10.3e  absmax=%10.3e\n",
              global_soot_min, global_soot_max, global_soot_absmax);
    }

      // Ratio of soot to species residuals
      double ratio = (global_species_Linf > 1.0e-300) ?
	global_soot_Linf / global_species_Linf : 0.0;
      fprintf(mf,"# STIFFNESS soot/species ratio=%10.3e\n", ratio);
    }

    // -------------------------------------------------------------------
    // Verbosity level 2: per-variable detail
    // -------------------------------------------------------------------
    if(params->residual_verbosity_ >= 2) {

      fprintf(mf, "# --- Per-species residual Linf ---\n");

      // Need to gather per-species Linf across MPI ranks
      // Allocate temporary arrays
      std::vector<double> local_per_species(num_species, 0.0);
      std::vector<double> global_per_species(num_species, 0.0);
      std::vector<double> local_per_soot(total_soot_vars, 0.0);
      std::vector<double> global_per_soot(total_soot_vars, 0.0);

      for(int j=0; j<num_local_points; ++j) {
        for(int k=0; k<num_species; ++k) {
          double val = fabs(ydot_ptr[j*num_states + k]);
          if(val > local_per_species[k]) local_per_species[k] = val;
        }
        for(int k=0; k<total_soot_vars; ++k) {
          double val = fabs(ydot_ptr[j*num_states + soot_idx_start + k]);
          if(val > local_per_soot[k]) local_per_soot[k] = val;
        }
      }

      MPI_Reduce(&local_per_species[0], &global_per_species[0],
                 num_species, MPI_DOUBLE, MPI_MAX, 0, comm);

      if(total_soot_vars > 0) {
        MPI_Reduce(&local_per_soot[0], &global_per_soot[0],
                   total_soot_vars, MPI_DOUBLE, MPI_MAX, 0, comm);
      }

      // Print top 10 species by residual magnitude
      // Build index-value pairs and partial sort
      std::vector<std::pair<double,int>> species_ranked(num_species);
      for(int k=0; k<num_species; ++k) {
        species_ranked[k] = std::make_pair(global_per_species[k], k);
      }
      std::partial_sort(species_ranked.begin(),
                        species_ranked.begin() + std::min(10, num_species),
                        species_ranked.end(),
                        std::greater<std::pair<double,int>>());

      fprintf(mf,"# Top 10 species residuals:\n");
      for(int i=0; i<std::min(10, num_species); ++i) {
	fprintf(mf,"#   %3d  %10.3e  %s\n",
		i+1,
		species_ranked[i].first,
		params->reactor_->GetNameOfStateId(species_ranked[i].second));
      }

      // Print top 10 soot sections by residual magnitude
      if(total_soot_vars > 0) {
        std::vector<std::pair<double,int>> soot_ranked(total_soot_vars);
        for(int k=0; k<total_soot_vars; ++k) {
          soot_ranked[k] = std::make_pair(global_per_soot[k], k);
        }
        std::partial_sort(soot_ranked.begin(),
                          soot_ranked.begin() + std::min(10, total_soot_vars),
                          soot_ranked.end(),
                          std::greater<std::pair<double,int>>());

	fprintf(mf, "# Top 10 soot section residuals:\n");
	for(int i=0; i<std::min(10, total_soot_vars); ++i) {
	  fprintf(mf, "#   %3d  %10.3e  %s\n",
		  i+1,
		  soot_ranked[i].first,
		  params->reactor_->GetNameOfStateId(soot_idx_start + soot_ranked[i].second));
	}
      }

    } // verbosity >= 2

    fflush(mf);

  } // my_pe == 0

  // -------------------------------------------------------------------
  // 6. Store norms on params for external access (all ranks)
  // -------------------------------------------------------------------
  params->species_residual_Linf_ = global_species_Linf;
  params->soot_residual_Linf_    = global_soot_Linf;
  params->thermo_residual_Linf_  = global_temp_Linf;
  params->species_chem_Linf_     = global_species_chem;
  params->species_conv_Linf_     = global_species_conv;
  params->species_diff_Linf_     = global_species_diff;
  params->soot_chem_Linf_        = global_soot_chem;
  params->soot_conv_Linf_        = global_soot_conv;
  params->soot_diff_Linf_        = global_soot_diff;

  // Clean up the vector we initialized above
  if(monitor_ydot_ != NULL) {
    N_VDestroy_Parallel(monitor_ydot_);
  }
  return 0;
} // FlameMonitorFunction


#if defined SUNDIALS2
int ReactorPreconditionerChemistrySetup(realtype t,      // [in] ODE system time
                                        N_Vector y,      // [in] ODE state vector
                                        N_Vector ydot,   // [in] ODE state derivative
                                        booleantype jok,
                                        booleantype *new_j,
                                        realtype gamma,
                                        void *user_data, // [in/out]
                                        N_Vector tmp1,
                                        N_Vector tmp2,
                                        N_Vector tmp3)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorPreconditionerChemistrySetup(realtype t,      // [in] ODE system time
                               N_Vector y,      // [in] ODE state vector
                               N_Vector ydot,   // [in] ODE state derivative
                               booleantype jok,
			       booleantype *new_j,
			       realtype gamma,
		               void *user_data) // [in/out]
{
#endif
  FlameParams *params    = (FlameParams *)user_data;
  const int num_local_points   = params->num_local_points_;
  const int num_states   = params->reactor_->GetNumStates();
  const int num_nonzeros = params->reactor_->GetJacobianSize();
  double *y_ptr          = NV_DATA_P(y); //_S // caution: assumes realtype == double
  int error_flag = 0;

  if(params->store_jacobian_) {

    if(!jok) {
      // The Jacobian is not okay, need to recompute
      params->saved_jacobian_.assign(num_nonzeros*num_local_points, 0.0);
      for(int j=0; j<num_local_points; ++j) {
	params->reactor_->SetViscosity(params->mixture_viscosity_[j]);
        params->reactor_->GetJacobianLimiter(t,
					     &y_ptr[j*num_states],
					     &params->step_limiter_[0],
                                             &params->saved_jacobian_[j*num_nonzeros]);
      } // for j<num_local_points
      (*new_j) = true;
    } else {
      (*new_j) = false;
    } // if/else jok

    for(int j=0; j<num_local_points; ++j) {
      // compute I - gamma*J
      // this could be updated with blas routines
      for(int k=0; k<num_nonzeros; ++k) {
        params->reactor_jacobian_[k] =
          -gamma*params->saved_jacobian_[j*num_nonzeros+k];
      }
      for(int k=0; k<num_states; ++k) {
        params->reactor_jacobian_[params->diagonal_id_[k]] += 1.0;
      }
      // factor the numerical jacobian
      if(params->sparse_matrix_[j]->IsFirstFactor()) {
        error_flag =
          params->sparse_matrix_[j]->FactorNewPatternCCS(num_nonzeros,
                                                         &params->row_id_[0],
                                                         &params->column_sum_[0],
                                                         &params->reactor_jacobian_[0]);
      } else {
        error_flag =
          params->sparse_matrix_[j]->FactorSamePattern(
                                                 &params->reactor_jacobian_[0]);
      }
      if(error_flag != 0) {
        params->logger_->PrintF(
          "# DEBUG: At t = %.18g [s],\n"
          "#        grid point %d (z = %.18g [m]) reactor produced a\n"
          "#        sparse matrix error flag = %d\n", t, j, params->z_[j],  error_flag);
        return error_flag;
      }

     } // for(int j=0; j<num_local_points; ++j)

  } else {
    // recompute and factor the Jacobian, there is no saved data
    // TODO: offer option for the fake update
    for(int j=0; j<num_local_points; ++j) {
      params->reactor_->SetViscosity(params->mixture_viscosity_[j]);
      params->reactor_->GetJacobianLimiter(t,
					   &y_ptr[j*num_states],
					   &params->step_limiter_[0],
					   &params->reactor_jacobian_[0]);

      // compute I - gamma*J
      // this could be updated with blas routines
      for(int k=0; k<num_nonzeros; ++k) {
        params->reactor_jacobian_[k] *= -gamma;
      }
      for(int k=0; k<num_states; ++k) {
        params->reactor_jacobian_[params->diagonal_id_[k]] += 1.0;
      }
      // factor the numerical jacobian
      if(params->sparse_matrix_[j]->IsFirstFactor()) {
        error_flag =
          params->sparse_matrix_[j]->FactorNewPatternCCS(num_nonzeros,
						 &params->row_id_[0],
						 &params->column_sum_[0],
						 &params->reactor_jacobian_[0]);
      } else {
        error_flag =
          params->sparse_matrix_[j]->FactorSamePattern(
                                                 &params->reactor_jacobian_[0]);
      }
      if(error_flag != 0) {
        params->logger_->PrintF(
          "# DEBUG: At t = %.18g [s],\n"
          "#        grid point %d (z = %.18g [m]) reactor produced a\n"
          "#        sparse matrix error flag = %d\n", t, j, params->z_[j], error_flag);
        return error_flag;
      }

     } // for(int j=0; j<num_points; ++j)

    (*new_j) = true; // without saving, it is always a new Jacobian

  } // if(params->store_jacobian_) else

  return 0;
}

#if defined SUNDIALS2
int ReactorPreconditionerChemistrySolve(realtype t,      // [in] ODE system time
                                        N_Vector y,      // [in] ODE state vector
                                        N_Vector ydot,   // [in] ODE state derivative
                                        N_Vector r,      // [in] jacobian rhs
                                        N_Vector z,      // [out]
                                        realtype gamma,
                                        realtype delta,
                                        int lr,
                                        void *user_data, // [in/out]
                                        N_Vector tmp)
{
#elif defined SUNDIALS3 || defined SUNDIALS4
int ReactorPreconditionerChemistrySolve(realtype t,      // [in] ODE system time
                               N_Vector y,      // [in] ODE state vector
                               N_Vector ydot,   // [in] ODE state derivative
                               N_Vector r,      // [in] jacobian rhs
                               N_Vector z,      // [out]
			       realtype gamma,
			       realtype delta,
			       int lr,
		               void *user_data)    // [in/out]
{
#endif
  FlameParams *params = (FlameParams *)user_data;
  const int num_local_points  = params->num_local_points_;
  const int num_states  = params->reactor_->GetNumStates();
  double *rhs         = NV_DATA_P(r);  // pointers to data array for N_Vector
  double *solution    = NV_DATA_P(z);  // pointers to data array for N_Vector
  int error_flag = 0;
  int start_id=0;

  for(int j=0; j<num_local_points; ++j) {
    error_flag = params->sparse_matrix_[j]->Solve(&rhs[start_id],
                                                  &solution[start_id]);
    start_id += num_states;
    if(error_flag != 0) {
      return error_flag;
    }
  }

  return error_flag;
}

static double FindMinimumAbsParallel(const size_t num_points,
                                     const double x[],
                                     const size_t x_stride,
                                     const double f[],
                                     const size_t f_stride,
                                     const bool use_quadratic,
                                     double *x_at_min,
                                     int *j_at_min)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local minimum of |f|
  in.value = fabs(f[0]);
  in.index = 0;
  for(int j=1; j<(int)num_points; ++j) {
    if(in.value > fabs(f[j*f_stride]) ) {
      in.value = fabs(f[j*f_stride]);
      in.index = j;
    }
  }

  // Compute global minimum
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  *x_at_min = x[out.index];
  *j_at_min = out.index;

  return out.value;
}

static double FindMaximumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_max)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local maximum
  in.value = f[0];
  in.index = 0;
  for(int j=1; j<num_points; ++j) {
    if(in.value < f[j]) {
      in.value = f[j];
      in.index = j;
    }
  }

  // Compute global maximum
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  *j_at_max = out.index;
  return out.value;
}

static double FindMinimumParallel(const int num_points,
                                  const double f[],
                                  int *j_at_min)
{
  int myrank;
  struct {
    double value;
    int index;
  } in, out;

  // Compute local minimum
  in.value = f[0];
  in.index = 0;
  for(int j=1; j<num_points; ++j) {
    if(in.value > f[j]) {
      in.value = f[j];
      in.index = j;
    }
  }

  // Compute global maximum
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  in.index += myrank*num_points;

  MPI_Allreduce(&in,&out,1,MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

  *j_at_min = out.index;

  return out.value;
}
