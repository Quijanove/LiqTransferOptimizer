# %%
from init import BO_LiqTransfer
ROOT = 'C:\\Users\\amdm_\\OneDrive\\Documents\\GitHub\\viscosity_liquid_transfer_Pablo\\Opentrons_experiments\\BOTorch_optimization\\VS_code_csv\\'

# Change according to experiment
liquid_name = 'Viscosity_std_1275' 

# Do not change
liq = BO_LiqTransfer(liquid_name)
liq.data_from_csv(ROOT + 'Viscosity_std_1275_3_vol_opt_more_trials_final_changed_bounds_correct.csv')
liq.data

# %%
liq.optimized_suggestions()

# %%
volume = 300
liq.update_data(error=-0.840723, volume=volume)

# %%
#save after each standard-experiment iteration
liq.data.to_csv(ROOT + f'{liquid_name}_duplicate_unused_exp3.csv', index=False)

# %%
liq.data

# %%
