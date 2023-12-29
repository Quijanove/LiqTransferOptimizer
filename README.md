# LiqTransferOptimizer: Multi-Objective Bayesian Optimization for Viscous Liquid Transfer Parameters
This repository contains the code to execute the optimzation of the apsiration and dispense rates of robotic air displacement pipettes for the accurate and fast transfer of challenging liquids. This GitHub repository is part of the Electronic Supporting Information (ESI) for an undergoing publication where we described a method to use MOBO to obtain aspiration and dispense rates that minimize transfer error and time to transfer 1000 µL of viscous liquids.

## How to use this resource?
The code required to perform the MOBO of the liquid handling parameters of viscous liquids is define the BO_LiqTransfer class in bo_liquid_transfer.py. The class can be used to either perform the optimization of liquid handling parameters (aspiration and dispense rates) of a electronic pipette through a semi-automated method where a human experimenterer assist with the optimization or in full automated fashion using a liquid handling platform able to be controlled from a kernel with torch packages installed and that can contains an automated mass balance. The former method is preferred when the optimization of a commercial liquid handling platform such as the Opentrons OT2 platform that does not allow the user to install the required python packages to perfomr BO or that it does not contain an autoamted mass balance. The latter method was used in a in-house assembled liquid handling platform controlled through controllalby a lab equipment automation package (https://pypi.org/project/control-lab-ly/).









