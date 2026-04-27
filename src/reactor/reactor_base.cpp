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
  std::vector<int> sp_idx(10);

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
    } else if (current_name == "PYRENE") {
      sp_idx[6] = j;
    } else if (current_name == "CO") {
      sp_idx[7] = j;
    } else if (current_name == "O") {
      sp_idx[8] = j;
    } else if (current_name == "CO2") {
      sp_idx[9] = j;
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
