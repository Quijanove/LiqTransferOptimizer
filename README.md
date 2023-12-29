# LiqTransferOptimizer: Multi-Objective Bayesian Optimization for Viscous Liquid Transfer Parameters
This repository contains the code to execute the optimzation of the apsiration and dispense rates of robotic air displacement pipettes for the accurate and fast transfer of challenging liquids. This GitHub repository is part of the Electronic Supporting Information (ESI) for an undergoing publication where we described a method to use MOBO to obtain aspiration and dispense rates that minimize transfer error and time to transfer 1000 µL of viscous liquids.

## The protocol
The experimental protocol to perform the optimization of liquid handling parameters of viscous liquids is summarized in the figure below. The protocol comprises 3 steps: initialization, exploration and optimization. The in the initialization step the flow rate of the liquid within the pipette tip (a1) is estimated by recording the time required to aspirate 1000 µL at a default rate. The second step is the exploration phase where the boundries of the parametric space of aspiration and dispense rates are tested. In this step five gravimetric transfers are recorded using the following combinations  aspiration and dispense rate respectively: (a1,a1), (1.25xa1,1.25xa1), (1.25xa1,0.1xa1), (0.1xa1,1.25xa1)  and (0.1xa1,0.1xa1). Finally a MOBO algorithm is used to suggest new combination of aspiration and dispense rates that minimize transfer error and time to transfer 1000 µL.
![Protocol_github](https://github.com/Quijanove/LiqTransferOptimizer/assets/99941287/562e66f6-a8bf-4bb9-b2d3-807bfe863fa8)

## How to use this resource?
The code required to perform the MOBO of the liquid handling parameters of viscous liquids is define the BO_LiqTransfer class in bo_liquid_transfer.py. The class can be used to either perform the optimization of liquid handling parameters (aspiration and dispense rates) of a electronic pipette through a semi-automated method where a human experimenterer assist with the optimization or in full automated fashion using a liquid handling platform able to be controlled from a kernel with torch packages installed and that can contains an automated mass balance. The former method is preferred when the optimization of a commercial liquid handling platform such as the Opentrons OT2 platform that does not allow the user to install the required python packages to perfomr BO or that it does not contain an autoamted mass balance. The latter method was used in a in-house assembled liquid handling platform controlled through controllalby a lab equipment automation package (https://pypi.org/project/control-lab-ly/).

The notebook MOBO_liquid_handling_paramters.ipynb has two examples of how the BO_LiqTransfer can be implemented with 
1) An Opentrons OT2 liquid handling platform through a semi automated method. The method requires an experimenterer to run the code and to measure the weight changes after each tranfer. The MOBO_liquid_handling_paramters.ipynb notebook needs to be run in parallel with that controls the actions of the Opentrons OT2 platform.
2) Fully automated protocol using an in-house assembled liquid handling platform. The method requires a liquid handling platform with an automated mass balance and with a computer with botorch and controllably packages isntalled. The code to control the platform can be reused by integrating the code control the automated platform with the controllably framework.








