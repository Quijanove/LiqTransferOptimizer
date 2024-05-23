# %%
#Import robot related packages and run setup
import pandas as pd
import time
from matplotlib import pyplot as plt
from init import ClosedLoop_BO_LiqTransfer

from configs.Platform import SETUP, LAYOUT_FILE

from controllably import load_deck      # optional
load_deck(SETUP.setup, LAYOUT_FILE)     # optional

platform = SETUP
platform.mover.verbose = False #askpablo

# %%
#Initialization of variables for platform objects
pipette= platform.setup
deck = platform.setup.deck
balance = platform.balance
balance_deck = deck.slots['1']
source = deck.slots['2']
tip_rack = deck.slots['3']
bin = deck.slots['4']
pipette.mover.setSpeed(50)
print(balance_deck)
print(source)
print(tip_rack)
print(bin)

#Check if balance is connected
balance.zero() #to tare
balance.toggleRecord(True) # turn on and record weight
time.sleep(5) # do previous action for 5s
print(balance.buffer_df.iloc[-1]) #iloc can take -1, loc needs to be 839 loc[839,["Time","Value","Factor","Baseline","Mass"]]. -1 is last line. to find number of last line, print(balance.buffer_df)
balance.toggleRecord(False) #turn off

# %%

liq = ClosedLoop_BO_LiqTransfer('BPAEDMA',1.12)
liq.platform = platform

# %%
liq.obtain_aproximate_rate(7.5,balance_deck.wells['A1'],file_name='BPAEDMA_flow_rate.csv')

liq.clean_tip(source.wells['A1'])

liq.explore_boundaries(42,source.wells['A1'],balance_deck.wells['A1'])

source_iquid_level = liq.data['liquid_level'].iloc[-1]

# liq.optimize_parameters(initial_liquid_level_source=source_iquid_level,source_well=source.wells['A1'],balance_well=balance_deck.wells['A1'])

# %%
liq.optimize_parameters(initial_liquid_level_source=39,source_well=source.wells['A1'],balance_well=balance_deck.wells['A1'])

# %%
liq.data

# %%
balance.clearCache() #to tare
balance.zero()
balance.toggleRecord(True) # turn on and record weight
time.sleep(10) # do previous action for 5s
print(balance.buffer_df.iloc[-1]) #iloc can take -1, loc needs to be 839 loc[839,["Time","Value","Factor","Baseline","Mass"]]. -1 is last line. to find number of last line, print(balance.buffer_df)
balance.toggleRecord(False) #turn off

# %%
df = liq.verify_parameters(49.5,source_well=source.wells['A1'],balance_well=balance_deck.wells['A1'],file_name='full_auto_505_verify_inverse.csv')  #BUG method not defined

# %%
platform.mover.disconnect()