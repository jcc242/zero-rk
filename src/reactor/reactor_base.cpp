#include "reactor_base.h"

#include "section_soot.h"

ReactorError ReactorBase::BuildMechanism(const char mechanism_name[],
                                         const char thermodynamics_name[],
                                         const char parser_log_name[])
{
  mechanism_name_      = std::string(mechanism_name);
  thermodynamics_name_ = std::string(thermodynamics_name);
  parser_log_name_     = std::string(parser_log_name);

  // TODO: add more robust file/validity checks
  mechanism_ = new zerork::mechanism(mechanism_name,
                                     thermodynamics_name,
                                     parser_log_name);
  if(mechanism_ == NULL) {
    return INVALID_MECHANISM;
  }
  num_species_ = mechanism_->getNumSpecies();
  if(num_species_ < 1) {
    return INVALID_MECHANISM;
  }
  num_reactions_ = mechanism_->getNumReactions();
  if(num_reactions_ < 1) {
    return INVALID_MECHANISM;
  }
  num_steps_ = mechanism_->getNumSteps();
  if(num_steps_ < 1) {
    return INVALID_MECHANISM;
  }
  // Initialize all A-Factors to one
  a_multipliers_.assign(num_steps_,1.0);
  return NONE;
}

void ReactorBase::DestroyMechanism()
{
  if(mechanism_ != NULL) {
    delete mechanism_;
  }
}

ReactorError ReactorBase::SetJacobianSize(const int jacobian_size)
{
  if(jacobian_size < 1) {
    return INDEX_OUT_OF_RANGE;
  }
  jacobian_size_ = jacobian_size;
  return NONE;
}



// Needs to set the following private data members
//  num_states_
//  state_names_
//  state_names_map_
int
  ReactorBase::BuildStateNamesMap(const std::vector<std::string> &state_names)
{
  num_states_ = static_cast<int>(state_names.size());
  printf("num_states_: %d\n",num_states_);
  fflush(stdout);
  state_names_.clear();
  state_names_map_.clear();
  for(int j=0; j<num_states_; ++j) {
    state_names_.push_back(state_names[j]);
    state_names_map_[state_names[j]] = j;
  }
  return num_states_;
}


int ReactorBase::GetIdOfState(const char *state_name) const
{
  std::string search_name = std::string(state_name);
  std::map<std::string,int>::const_iterator iter;

  iter = state_names_map_.find(search_name);
  if(iter == state_names_map_.end()) {
    // state_name not found
    return -1;
  }
  return iter->second;
}
const char * ReactorBase::GetNameOfStateId(const int state_id) const
{
  if(state_id < 0 || state_id >= num_states_) {
    return "invalid state index";
  }
  return state_names_[state_id].c_str();
}



double 
  ReactorBase::GetAMultiplierOfForwardReactionId(const int reaction_id) const
{
  if(reaction_id < 0 || reaction_id >= num_reactions_) {
    // reaction index does not exist
    return 0.0;
  }
  const int step_id = mechanism_->getStepIdxOfRxn(reaction_id,1);
  if(step_id < 0 || step_id >= num_steps_) {
    // step index does not exist
    return 0.0;
  }
  return a_multipliers_[step_id];
}
ReactorError 
  ReactorBase::SetAMultiplierOfForwardReactionId(const int reaction_id, 
                                                 const double a_multiplier)
{
  if(reaction_id < 0 || reaction_id >= num_reactions_) {
    // reaction index does not exist to be set
    return INDEX_OUT_OF_RANGE;
  }
  const int step_id = mechanism_->getStepIdxOfRxn(reaction_id,1);
  if(step_id < 0 || step_id >= num_steps_) {
    // step index does not exist to be set
    return INDEX_OUT_OF_RANGE;
  }
  a_multipliers_[step_id] = a_multiplier;
  return NONE;
}

double 
  ReactorBase::GetAMultiplierOfReverseReactionId(const int reaction_id) const
{
  if(reaction_id < 0 || reaction_id >= num_reactions_) {
    // reaction index does not exist
    return 0.0;
  }
  const int step_id = mechanism_->getStepIdxOfRxn(reaction_id,-1);
  if(step_id < 0 || step_id >= num_steps_) {
    // step index does not exist
    return 0.0;
  }
  return a_multipliers_[step_id];
}

ReactorError 
  ReactorBase::SetAMultiplierOfReverseReactionId(const int reaction_id, 
                                                 const double a_multiplier)
{
  if(reaction_id < 0 || reaction_id >= num_reactions_) {
    // reaction index does not exist to be set
    return INDEX_OUT_OF_RANGE;
  }
  const int step_id = mechanism_->getStepIdxOfRxn(reaction_id,-1);
  if(step_id < 0 || step_id >= num_steps_) {
    // step index does not exist to be set
    return INDEX_OUT_OF_RANGE;
  }
  a_multipliers_[step_id] = a_multiplier;
  return NONE;
}

double 
  ReactorBase::GetAMultiplierOfStepId(const int step_id) const
{
  if(step_id < 0 || step_id >= num_steps_) {
    // step index does not exist
    return 0.0;
  }
  return a_multipliers_[step_id];
}
ReactorError 
  ReactorBase::SetAMultiplierOfStepId(const int step_id, 
                                      const double a_multiplier)
{
  if(step_id < 0 || step_id >= num_steps_) {
    // step index does not exist to be set
    return INDEX_OUT_OF_RANGE;
  }
  a_multipliers_[step_id] = a_multiplier;
  return NONE;
}

ReactorError ReactorBase::InitializeSectionalSoot(const bool use_sectional)
{
  num_soot_secs_ = 0;
  num_soot_psd_ = 0;
  use_sectional_ = use_sectional;
  if (!use_sectional_) return NONE;
  std::vector<int> sp_idx(20);

  for (int j=0; j<num_species_; ++j) {
    std::string current_name = std::string(mechanism_->getSpeciesName(j));

    if (current_name == "C2H2") {
      sp_idx[0] = j;
    } else if (current_name == "H2") {
      sp_idx[1] = j;
    } else if (current_name == "O2") {
      sp_idx[2] = j;
    } else if (current_name == "OH") {
      sp_idx[3] = j;
    } else if (current_name == "H") {
      sp_idx[4] = j;
    } else if (current_name == "H2O") {
      sp_idx[5] = j;
    // } else if (current_name == "PYRENE") {
    //   sp_idx[6] = j; // Not used at the moment!
    } else if (current_name == "CO") {
      sp_idx[7] = j;
    } else if (current_name == "O") {
      sp_idx[8] = j;
    } else if (current_name == "CO2") {
      sp_idx[9] = j;
    } else if (current_name == "C4H2") {
      sp_idx[10] = j;
    } else if (current_name == "A4R5") {
      sp_idx[11] = j;
    } else if (current_name == "C6H6") {
      sp_idx[12] = j;
    } else if (current_name == "C6H5C2H") {
      sp_idx[13] = j;
    } else if (current_name == "NAPH") {
      sp_idx[14] = j;
    } else if (current_name == "A2R5") {
      sp_idx[15] = j;
    } else if (current_name == "ANTHRACENE") {
      sp_idx[16] = j;
    } else if (current_name == "PHNTHRN") {
      sp_idx[17] = j;
    } else if (current_name == "A3R5") {
      sp_idx[18] = j;
    } else if (current_name == "PYRENE") {
      sp_idx[19] = j;
    }
  }
  sootsec_initialize_sp_idx(num_species_, sp_idx.data(), &num_soot_secs_, &num_soot_psd_);
  soot_masses_.assign(num_soot_secs_*num_soot_psd_, 0.0);
  sootsec_mass_si(soot_masses_.data());

  return NONE;
}

ReactorError ReactorBase::FinalizeSectionalSoot() {
  sootsec_finalize();
  return NONE;
}

ReactorError ReactorBase::ComputeSootResidual(const std::vector<double>& species_conc,
					      const std::vector<double>& current_soot_values,
					      const double temperature,
					      const double pressure,
					      const double density,
					      const double viscosity,
					      std::vector<double>& species_residual,
					      std::vector<double>& soot_residual) {
  sootsec_compute_residual_si(species_conc.data(), current_soot_values.data(),
			      temperature, pressure, density, viscosity,
			      species_residual.data(), soot_residual.data());

  // // ================================================================
  // // MASS CONSERVATION CHECK
  // //
  // // Gas side:  species_residual[i] in [kmol/(m^3*s)]
  // //            mass rate = species_residual[i] * MW[i]  → [kg/(m^3*s)]
  // //
  // // Soot side: soot_residual[k] in [#/(m^3*s)]
  // //            mass rate = soot_residual[k] * m_bin[k]  → [kg/(m^3*s)]
  // //
  // // Conservation: gas_mass_rate + soot_mass_rate = 0
  // //   (mass leaving gas = mass entering soot)
  // // ================================================================
  // {
  //   const int num_species = GetNumSpecies();
  //   const int total_soot_vars = GetNumSectionalTotal();
  //   const std::vector<double>& soot_sec_mass = GetSectionsMass(); // [kg/particle]

  //   zerork::mechanism *mech = GetMechanism();
  //   std::vector<double> mol_wt(num_species);
  //   mech->getMolWtSpc(&mol_wt[0]);  // [kg/kmol]

  //   // Gas-phase mass rate: Σ species_residual[i] * MW[i]  [kg/(m^3*s)]
  //   double gas_mass_rate = 0.0;
  //   for(int i = 0; i < num_species; ++i) {
  //     gas_mass_rate += species_residual[i] * mol_wt[i];
  //   }

  //   // Soot mass rate: Σ soot_residual[k] * m_bin[k]  [kg/(m^3*s)]
  //   double soot_mass_rate = 0.0;
  //   for(int k = 0; k < total_soot_vars; ++k) {
  //     soot_mass_rate += soot_residual[k] * soot_sec_mass[k];
  //   }

  //   double net_mass_rate = gas_mass_rate + soot_mass_rate;

  //   // Compute scale for relative error
  //   double scale = std::max(fabs(gas_mass_rate), fabs(soot_mass_rate));
  //   double relative_error = (scale > 0.0) ? fabs(net_mass_rate) / scale : 0.0;

  //   // Print if error exceeds threshold
  //   if(relative_error > 1.0e-6 || fabs(net_mass_rate) > 1.0e-10) {
  //     printf("# SOOT_MASS_CONSERV: gas_mass_rate=%12.4e  soot_mass_rate=%12.4e"
  //            "  net=%12.4e  rel_err=%8.2e  T=%.1f K\n",
  //            gas_mass_rate, soot_mass_rate, net_mass_rate, relative_error, temperature);

  //     // Break down gas side by species to find where mass is leaking
  //     if(relative_error > 1.0e-4) {
  //       printf("# SOOT_MASS_CONSERV:   Gas breakdown (top contributors):\n");
  //       for(int i = 0; i < num_species; ++i) {
  //         double contribution = species_residual[i] * mol_wt[i];
  //         if(fabs(contribution) > 0.01 * scale) {
  //           printf("# SOOT_MASS_CONSERV:     species %3d: resid=%12.4e kmol/m3/s"
  //                  "  mass_rate=%12.4e kg/m3/s  (MW=%.2f)\n",
  //                  i, species_residual[i], contribution, mol_wt[i]);
  //         }
  //       }
  //       printf("# SOOT_MASS_CONSERV:   Soot breakdown (top contributors):\n");
  //       for(int k = 0; k < std::min(total_soot_vars, 10); ++k) {
  //         double contribution = soot_residual[k] * soot_sec_mass[k];
  //         if(fabs(contribution) > 0.01 * scale) {
  //           printf("# SOOT_MASS_CONSERV:     bin %2d: resid=%12.4e #/m3/s"
  //                  "  mass_rate=%12.4e kg/m3/s  (m_bin=%.4e kg)\n",
  //                  k, soot_residual[k], contribution, soot_sec_mass[k]);
  //         }
  //       }
  //     }
  //   }
  // }
  
  return NONE;
}

ReactorError ReactorBase::GetLastSootRates(double *coag, double *sg, double *ox,
					   double *cond, double *nuc,
					   double *nuc_gas, double *sg_gas,
					   double *ox_gas, double *cond_gas) const {

  sootsec_last_rates(&coag[0], &sg[0], &ox[0],
		     &cond[0], nuc,
		     &nuc_gas[0], &sg_gas[0], &ox_gas[0], &cond_gas[0]);
    
  return NONE;
}
