# %% basic dependencies
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import time
from typing import Optional, Iterable
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# Platform configs
from pathlib import Path
import sys
REPO = 'LiqTransferOptimizer'
ROOT = str(Path().absolute()).split(REPO)[0]
sys.path.append(f'{ROOT}{REPO}')

# %% torch dependencies
import torch
torch.set_printoptions(precision=3)
tkwargs = {
    "dtype": torch.double,                                                      # set as double to minimize zero error for cholesky decomposition error
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # set tensors to GPU, if multiple GPUs please set cuda:x properly
}

# %% botorch dependencies
# data related
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize

# surrogate model specific
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_model

# qNEHVI specific
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

# utilities
from botorch.sampling import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

# %%
import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# %% plotting dependencies
from matplotlib import pyplot as plt
# %matplotlib inline

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
FEATURES = ('aspiration_rate','dispense_rate')
OBJECTIVES = (r'%error','time_asp_1000')
CLOSE_LOOP_COLS = (
    'liquid', 
    'pipette',
    'volume',
    'aspiration_rate',
    'dispense_rate',
    'blow_out_rate',
    'delay_aspirate',
    'delay_dispense',
    'delay_blow_out',
    'touch_tip_aspirate',
    'touch_tip_dispense',
    'density',
    r'%error',
    'time_asp_1000',
    'acq_value',
    'iteration',
    'liquid_level'
)
OT_SELECTED_COLS = [
    'liquid',
    'pipette',
    'volume',
    'aspiration_rate',
    'dispense_rate',
    'blow_out_rate',
    'delay_aspirate',
    'delay_dispense',
    'delay_blow_out',
    r'%error'
]

# %%
class BO_LiqTransfer:
    def __init__(self, 
        liquid_name:str, 
        columns: Optional[list[str]] = None,
        features: Iterable = FEATURES,
        objectives: Iterable = OBJECTIVES,
    ):
        self.liquid_name = liquid_name
        self.data = pd.DataFrame(columns=columns)
        self.features = features
        self.objectives = objectives
        self.mean_volumes = [1000,500,300]
        
        # Exploration bound factors
        self.bmax = 1.25
        self.bmin = 0.1
        
        self._latest_acq_value = None
        self._latest_suggestion = None
        self._latest_volume = None
        return
    
    def data_from_csv(self, file_name:str, select_columns:Optional[list[str]] = None):
        data = pd.read_csv(file_name)
        if select_columns is not None:
            data = data.loc[:, select_columns]
        self.set_data(data)
        return
    
    def fit_surrogate(self) -> tuple[ModelListGP, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_train, y_train = self.xy_split()
        x_train = torch.tensor(x_train.to_numpy(dtype=float), **tkwargs)
        y_train = torch.tensor(y_train.to_numpy(dtype=float), **tkwargs)
        y_train[:,0] = -torch.absolute(y_train[:,0])
        y_train[:,1] = -torch.absolute(y_train[:,1])

        problem_bounds = self.set_bounds(x_train)
        time_upper = 1000/problem_bounds[0][0] +1000/problem_bounds[0][1] + 10
        error_upper = y_train[:,0].abs().min()*1.25
        ref_point = torch.tensor([-error_upper,-time_upper], **tkwargs)

        train_x_gp = normalize(x_train, problem_bounds)
        models = []
        for i in range(y_train.shape[-1]):
            models.append(SingleTaskGP(train_x_gp, y_train[..., i : i + 1], outcome_transform=Standardize(m=1)))
        model1 = ModelListGP(*models)
        mll1 = SumMarginalLogLikelihood(model1.likelihood, model1)
        fit_gpytorch_model(mll1)
        return model1, ref_point, train_x_gp, problem_bounds
    
    def optimized_suggestions(self, random_state:int = 42) -> pd.DataFrame:
        if random_state is not None:
            torch.manual_seed(random_state) 
        standard_bounds = torch.zeros(2, len(self.features), **tkwargs)
        standard_bounds[1] = 1
        model1, ref_point, train_x_gp, problem_bounds = self.fit_surrogate()
        acq_func1 = qNoisyExpectedHypervolumeImprovement(
            model=model1,
            ref_point=ref_point,                                                                            # for computing HV, must flip for BoTorch
            X_baseline=train_x_gp,                                                                          # feed total list of train_x for this current iteration
            sampler=SobolQMCNormalSampler(sample_shape=512),                                                # determines how candidates are randomly proposed before selection
            objective=IdentityMCMultiOutputObjective(outcomes=np.arange(len(self.objectives)).tolist()),    # optimize first n_obj col 
            prune_baseline=True, cache_pending=True                                                         # options for improving qNEHVI, keep these on
        )
        sobol1 = draw_sobol_samples(bounds=standard_bounds,n=512, q=1).squeeze(1)
        sobol2 = draw_sobol_samples(bounds=standard_bounds,n=512, q=1).squeeze(1)
        sobol_all = torch.vstack([sobol1, sobol2])
            
        acq_value_list = []
        for i in range(0, sobol_all.shape[0]):
            with torch.no_grad():
                acq_value = acq_func1(sobol_all[i].unsqueeze(dim=0))
                acq_value_list.append(acq_value.item())
                
        # filter the best 12 QMC candidates first
        sorted_x = sobol_all.cpu().numpy()[np.argsort((acq_value_list))]
        qnehvi_x = torch.tensor(sorted_x[-12:], **tkwargs)  
        # unnormalize our training inputs back to original problem bounds
        new_x = unnormalize(qnehvi_x, bounds=problem_bounds)
        new_x = pd.DataFrame(new_x.numpy(),columns=['aspiration_rate','dispense_rate'])
        new_x['acq_value'] = sorted(acq_value_list, reverse=True)[:12]
        self._latest_suggestion = new_x[['aspiration_rate','dispense_rate']].iloc[0]
        self._latest_acq_value = new_x['acq_value'].iloc[0]                                                 # NOTE: only for satorius 
        return new_x
    
    def set_bounds(self, x_train:Iterable) -> torch.Tensor:
        return torch.vstack([x_train[0]*self.bmin, x_train[0]*self.bmax])

    def set_data(self, df:pd.DataFrame, **kwargs):
        df['time_asp_1000'] = 1000/df['aspiration_rate'] + 1000/df['dispense_rate'] + df['delay_aspirate'] + df['delay_dispense']
        if 'acq_value' not in df.columns:
            df['acq_value'] = None

        if df.loc[:,self.features].duplicated().sum()==0:
            df_mean = df
        else:
            df_duplicates = df.where(df.duplicated(self.features,keep=False)==True).dropna(how='all')
            df_incomplete = df.where(df.duplicated(self.features,keep=False)==False).dropna(how='all')
            df_mean = pd.DataFrame(columns= df.columns)
            for index,_ in df_duplicates.drop_duplicates(self.features).iterrows():
                df_mean,df_incomplete = self._set_data_by_row(df, df_duplicates, index, **kwargs)
            df_mean = pd.concat([df_mean,df_incomplete])
            df_mean = df_mean.reset_index(drop=True)    
        self.data = df_mean
        return

    def update_data(self, df = None, error = None, volume=1000): 
        if df is not None and type(df) == pd.DataFrame:                     # sartorius version
            self._latest_volume = df['volume'].iloc[-1]
            updated_data = pd.concat([self.data,df],ignore_index=True)
        else:                                                               # opentrons version
            self._latest_volume = volume
            updated_data = pd.concat([self.data,self.data.iloc[[-1]]],ignore_index=True)
            updated_data.loc[updated_data.last_valid_index(),'volume'] = self._latest_volume
            updated_data.loc[updated_data.last_valid_index(),'aspiration_rate']  = self._latest_suggestion['aspiration_rate']
            updated_data.loc[updated_data.last_valid_index(),'dispense_rate']  = self._latest_suggestion['dispense_rate']
            updated_data.loc[updated_data.last_valid_index(),'%error'] = error
        
        self.set_data(updated_data)
        return self.data
    
    def xy_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train = self.data.where(self.data['volume']=='mean'+str(self.mean_volumes)).dropna(how='all')
        x_train = df_train[self.features]
        y_train = df_train[self.objectives]
        return x_train,y_train

    # Protected methods
    def _set_data_by_row(self, df, df_duplicates, index, *args, **kwargs):
        if len(df_duplicates.loc[index:index+2]) == len(self.mean_volumes):
            mean_error =df_duplicates.loc[index:index+2,r'%error'].abs().mean()
            df_duplicates.loc[index,r'%error'] = -mean_error
            df_duplicates.loc[index, 'volume'] ='mean'+str(self.mean_volumes)
            df_mean = pd.concat([df_mean,df.loc[index:index+2],df_duplicates.loc[[index]]])
        else:
            df_incomplete = pd.concat([df_incomplete,df_duplicates.loc[index:index+2]]).drop_duplicates()
        return
    


class ClosedLoop_BO_LiqTransfer(BO_LiqTransfer):
    def __init__(self, 
        liquid_name: str, 
        columns: list[str] | None = None, 
        features: Iterable = FEATURES, 
        objectives: Iterable = OBJECTIVES,
        density:Optional[float] = None
    ):
        super().__init__(liquid_name, columns, features, objectives)
        self.density = density
        self._first_approximation = None
        self.platform = None
        return

    @staticmethod
    def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:

        if 'volume_transfered' and 'volume_error' and 'time_asp_1000' not in df.columns:
            df['volume_transfered'] = (df['m']/df['density'])*1000
            df['volume_error'] = df['volume_transfered'] - df['volume']
            df['time_asp_1000']=1000/df['aspiration_rate'] + 1000/df['dispense_rate'] + df['delay_aspirate'] + df['delay_dispense']             

        df_summary_all = pd.DataFrame()

        for volume in df['volume'].unique():
            df_experiment_v = df.where(df['volume'] == volume).dropna(how='all')
            df_summary = pd.DataFrame(columns = (f'Mean transfer volume for {volume} µL [µL]', f'Mean transfer volume error of {volume} µL [µL]', f'Mean relative error for transfer of {volume} µL [%]', f'Standard deviation for transfer of {volume} µL [µL]', f'Relative standard deviation for transfer of {volume} µL [%]') )
            data = [df_experiment_v['volume_transfered'].mean(), df_experiment_v['volume_error'].mean(), df_experiment_v[r'%error'].mean(), df_experiment_v['volume_transfered'].std(), (df_experiment_v['volume_transfered'].std() / df_experiment_v['volume_transfered'].mean() * 100)]
            df_summary.loc[df['liquid'].iloc[0]] = data
            df_summary_all = pd.concat([df_summary_all, df_summary], axis = 1)
        return df_summary_all
    
    def calibrate_parameters(self, 
        initial_liquid_level_source: float,
        source_well,
        balance_well,
        iterations: int = 10, 
        file_name: Optional[str] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        self.platform.mover.setSpeed(50)
        self.platform.mover.setHandedness(False)
        liquid_level = initial_liquid_level_source

        # Check if new tip is required
        if self.platform.liquid.isTipOn()== False:
            self.platform.setup.attachTip()

        volumes_list = self.mean_volumes
        
        #NOT TO BE CHANGED
        mean_average_data = self.data.where(self.data.volume == 'mean'+str(self.mean_volumes))
        mean_average_data = mean_average_data.where(mean_average_data.iteration>5).dropna()
        best_parameter_index = mean_average_data[mean_average_data[r'%error']==mean_average_data[r'%error'].max()].index

        aspiration_rate = self.data.loc[best_parameter_index,'aspiration_rate'].values[0]
        dispense_rate = self.data.loc[best_parameter_index,'dispense_rate'].values[0]
        
        calibration_df = pd.DataFrame(columns = ['liquid', 'pipette', 'volume', 'aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense','liquid_level','density', 'm', r'%error'])
        
        #for loop
        for volume in volumes_list:
            counter = 1
            # #while loop
            while counter <= iterations:
                #liquid transfer
                liquid_level,df = self.gravimetric_transfer(volume,liquid_level,source_well,balance_well,aspiration_rate,dispense_rate)
                calibration_df = pd.concat([calibration_df,df[calibration_df.columns]]).reset_index(drop=True)
                m = df.m.iloc[-1]
                
                #printing checks
                print("Mass: "+str(m)+"LIQUID LEVEL: " +str(liquid_level) + "   LIQUID CHANGE: " +str(1.2*m/self.density) + "   ITERATION: " + str(counter) + ", " + "VOLUME: " + str(volume))    

                #liquid level checks
                if (1.2*m/self.density > 1.2) or (1.2*m/self.density < 0):
                    break
                if (liquid_level > initial_liquid_level_source) or (liquid_level < 6):
                    break
                counter += 1
            #liquid level checks
            if (1.2*m/self.density > 1.2) or (1.2*m/self.density < 0):
                break
            if (liquid_level > initial_liquid_level_source) or (liquid_level < 6): 
                break
        
        calibration_df['volume_transfered'] = calibration_df['m']/calibration_df['density']*1000
        calibration_df['volume_error'] = calibration_df['volume_transfered'] - calibration_df['volume']
        calibration_df['time_asp_1000'] = 1000/calibration_df['aspiration_rate'] + 1000/calibration_df['dispense_rate'] + calibration_df['delay_aspirate'] + calibration_df['delay_dispense']       
        
        calibration_summary_df= self.calibration_summary(calibration_df)

        if file_name != None:
            calibration_df.to_csv(file_name, index=False)
            calibration_summary_df.to_csv(file_name[:-4]+'_summary.csv')
        return calibration_df, calibration_summary_df
    
    def clean_tip(self, source_well, repetitions:int = 2):
        self.platform.mover.safeMoveTo(source_well.top)
        for i in range(repetitions):
            self.platform.liquid.blowout(home=False) 
            time.sleep(5)
            self.platform.liquid.home()
            self.platform.setup.touchTip(source_well)
            time.sleep(5)

            self.platform.liquid.blowout(home=False) 
            time.sleep(5)
            self.platform.liquid.home()
            self.platform.setup.touchTip(source_well)
            time.sleep(5)

            self.platform.liquid.blowout(home=False) 
            time.sleep(5)
            self.platform.liquid.home()
            self.platform.setup.touchTip(source_well)
            time.sleep(5)
        return

    def explore_boundaries(self, 
        initial_liquid_level_source: float,
        source_well,
        balance_well
    ):
        self.platform.mover.setSpeed(50)
        self.platform.mover.setHandedness(False)
        liquid_level = initial_liquid_level_source

        if type(self.data) == None:
            df = pd.DataFrame(columns = ['liquid', 'pipette', 'volume', 'aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense','iteration','liquid_level','density', 'time', 'm', r'%error','time_asp_1000','acq_value'])
            df = df.astype({'liquid':str,'pipette':str})
            self.set_data(df)
        
        #Check if new tip is required
        if self.platform.liquid.isTipOn()== False:
            self.platform.setup.attachTip()
        volumes_list = self.mean_volumes
        
        #NOT TO BE CHANGED
        counter = 1
        iterations = 5
        #while loop
        while counter <= iterations:
            #hardcoding aspirate and dispense rates:
            if counter == 1:
                aspiration_rate = self._first_approximation
                dispense_rate = self._first_approximation
            if counter == 2:
                aspiration_rate = self._first_approximation*self.bmax
                dispense_rate = self._first_approximation*self.bmax
            if counter == 3:
                aspiration_rate = self._first_approximation*self.bmax
                dispense_rate = self._first_approximation*self.bmin
            if counter == 4:
                aspiration_rate = self._first_approximation*self.bmin
                dispense_rate = self._first_approximation*self.bmax
            if counter == 5:
                aspiration_rate = self._first_approximation*self.bmin
                dispense_rate = self._first_approximation*self.bmin

            #for loop
            for volume in volumes_list:
                #liquid transfer
                liquid_level,df = self.gravimetric_transfer(volume,liquid_level,source_well,balance_well,aspiration_rate,dispense_rate)
                m = df.m.iloc[-1]
                self.set_data(pd.concat([self.data,df]).reset_index(drop=True))
                
                #printing checks
                print(f"LIQUID LEVEL: {liquid_level}   LIQUID CHANGE: {1.2*m/self.density}   ITERATION: {counter}, VOLUME: {volume}")
                
                #liquid level checks
                if (1.2*m/self.density > 1.2) or (1.2*m/self.density < 0):
                    break
                if (liquid_level > initial_liquid_level_source) or (liquid_level < 6):
                    break
            counter += 1
        return

    def gravimetric_transfer(self, 
        volume: float, 
        liquid_level: float, 
        source_well,
        balance_well, 
        aspiration_rate: float, 
        dispense_rate: float
    ) -> tuple[float, pd.DataFrame]:
        #liquid transfer
        #transfer start
        start = time.time() 

        #aspirate step
        self.platform.mover.safeMoveTo(source_well.from_bottom((0,0,liquid_level-5))) 
        self.platform.liquid.aspirate(volume, speed=aspiration_rate)
        time.sleep(10)
        self.platform.setup.touchTip(source_well) 

        #dispense step
        self.platform.mover.safeMoveTo(balance_well.from_top((0,0,-5))) 
        self.platform.balance.tare() 
        self.platform.balance.clearCache() 
        self.platform.balance.toggleRecord(True) 
        time.sleep(5)
        self.platform.liquid.dispense(volume,speed=dispense_rate)
        time.sleep(10)

        #transfer termination
        finish = time.time() 
        time_m = finish - start

        self.platform.mover.safeMoveTo(source_well.top) 
        time.sleep(5)
        self.platform.balance.toggleRecord(False)

        #do blowout
        self.cleanTip(source_well)

        #record transfer values 
        #calculating mass error functions
        m = (self.platform.balance.buffer_df.iloc[-10:,-1].mean()-self.platform.balance.buffer_df.iloc[:10,-1].mean())/1000 
        error = (m-self.density*volume/1000)/(self.density/1000*volume)*100
        
        #change liquid levels
        liquid_level = liquid_level - 2*m/self.density   
        
        #making new dataframe + filling it in
        df = pd.DataFrame(columns=self.data.columns)            
        df = pd.concat([df, pd.DataFrame({
                "liquid": self.liquid_name,
                'pipette': 'rLine1000',
                "volume": volume,
                "aspiration_rate": aspiration_rate,
                "dispense_rate": dispense_rate, 
                "delay_aspirate" : 10,  
                "delay_dispense" : 10,
                'iteration': 'NaN',
                "liquid_level" : liquid_level,
                "density" : self.density,
                "time" : time_m,
                "m": m,
                r"%error": error,
                "time_asp_1000" : 'NaN',
                "acq_value": self._latest_acq_value
            }, index=[0])],ignore_index=True)
        return liquid_level, df

    def obtain_aproximate_rate(self, 
        initial_liquid_level_balance: float,
        balance_well,
        speed: float = 265, 
        file_name: Optional[str] = None
    ):
        liquid_level = initial_liquid_level_balance
        if self.platform.liquid.isTipOn()== False:
            self.platform.setup.attachTip()
        self.platform.mover.safeMoveTo(balance_well.from_bottom((0,0,liquid_level-5)),descent_speed_fraction=0.25)
        
        #Starting balance measurement
        time.sleep(5)
        self.platform.balance.zero(wait=5)
        self.platform.balance.clearCache()
        self.platform.balance.toggleRecord(on=True)
        time.sleep(15)
        self.platform.liquid.aspirate(1000, speed=speed)

        #Switching the balance off after change in mass is less than 0.05
        while True:
            data = self.platform.balance.buffer_df
            data['Mass_smooth']= signal.savgol_filter(data['Mass'],91,1)
            data['Mass_derivative_smooth']=data['Mass_smooth'].diff()
            condition=data['Mass_derivative_smooth'].rolling(30).mean().iloc[-1]
            if condition>-0.05:
                break
        print('loop stopped')
        self.platform.balance.toggleRecord(on=False)
        self.platform.mover.moveTo(balance_well.from_top((0,0,-5)))

        def sigmoid(x, K ,x0, B,v,A):
            y = (K-A) / (1 + np.exp(B*(x-x0)))**(1/v) + A
            return y

        #using data from balance buffer_df above, calculate time in seconds and mass derivatives
        data['ts'] = data['Time'].astype('datetime64[ns]').values.astype('float') / 1E9
        data['ts']= data['ts']-data['ts'][0]
        data_fit = data.where(data['ts']>10).dropna()
        data_fit['Mass']=data_fit['Mass']-data_fit['Mass'].iloc[0]
        data_fit['Mass_smooth'] = data_fit['Mass_smooth']-data_fit['Mass_smooth'].iloc[0]

        p0 = [min(data_fit['Mass']), np.median(data_fit['ts']),1,1,max(data_fit['Mass'])+30]
        
        popt, _ = curve_fit(sigmoid, data_fit['ts'], data_fit['Mass'],p0)
        mass_sigmoid = sigmoid(data_fit['ts'],popt[0],popt[1],popt[2],popt[3],popt[4])
        data_fit.loc[data_fit.index[0]:,'Mass_sigmoid'] = mass_sigmoid
        flow_rate = mass_sigmoid.diff()/data_fit.loc[data_fit.index[0]:,'ts'].diff()
        data_fit.loc[data_fit.index[0]:,'Flow_rate'] = flow_rate

        flow_rate = mass_sigmoid.diff()/data_fit.loc[data_fit.index[0]:,'ts'].diff()
        flow_rate_max = flow_rate.min()
        flow_rate_98 = data_fit.where(data_fit['Flow_rate']<(0.05*flow_rate_max)).dropna()

        time_start, time_final = flow_rate_98.iloc[0].loc['ts'],flow_rate_98.iloc[-1].loc['ts']
        initial_flow_rate_aspirate = 1000/(time_final-time_start)
        self._first_approximation = initial_flow_rate_aspirate 
        
        #switching balance off and saving csv
        if file_name != None:
            data_fit.to_csv(file_name, index=False)

        self.platform.liquid.dispense(1000,speed= self._first_approximation)
        return

    def optimize_parameters(self, 
        initial_liquid_level_source: float,
        source_well,
        balance_well,
        iterations: int =5,
        file_name: Optional[str] = None
    ):
        self.platform.mover.setSpeed(50)
        self.platform.mover.setHandedness(False)
        liquid_level = initial_liquid_level_source

        #Check if new tip is required
        if self.platform.liquid.isTipOn()== False:
            self.platform.setup.attachTip()
        volumes_list = self.mean_volumes
        
        #NOT TO BE CHANGED
        counter = 1
        #while loop
        while counter <= iterations:
            #getting botorch suggestions + implementing it in liquids_dict
            self.optimized_suggestions()
            aspiration_rate = self._latest_suggestion['aspiration_rate']
            dispense_rate = self._latest_suggestion['dispense_rate']
            #for loop
            for volume in volumes_list:
                #liquid transfer
                liquid_level,df = self.gravimetric_transfer(volume,liquid_level,source_well,balance_well,aspiration_rate,dispense_rate)
                m = df.m.iloc[-1]
                self.set_data(pd.concat([self.data,df]).reset_index(drop=True))
                
                #printing checks   
                print(f"LIQUID LEVEL: {liquid_level}   LIQUID CHANGE: {1.2*m/self.density}   ITERATION: {counter}, VOLUME: {volume}")
                
                #liquid level checks
                if (1.2*m/self.density > 1.2) or (1.2*m/self.density < 0):
                    break
                if (liquid_level > initial_liquid_level_source) or (liquid_level < 6):
                    break
            counter += 1
        if file_name != None:
            self.data.to_csv(file_name, index=False)
        return

    def set_data(self, df:pd.DataFrame):
        iteration = 1
        nan_columns = df.columns.to_list()
        nan_columns = [e for e in nan_columns if e not in CLOSE_LOOP_COLS]
        super().set_data(df, iteration=iteration, nan_columns=nan_columns)
        return
        
    # Protected methods
    def _set_data_by_row(self, df, df_duplicates, index, iteration, nan_columns) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(df_duplicates.loc[index:index+2]) == len(self.mean_volumes):
            mean_error = df_duplicates.loc[index:index+2,r'%error'].abs().mean()
            df_duplicates.loc[index,r'%error'] = -mean_error
            df_duplicates.loc[index, 'volume'] ='mean'+str(self.mean_volumes)
            df_duplicates.loc[index, 'iteration'] = iteration
            df_duplicates.loc[index, 'liquid_level'] = df.loc[index+2,'liquid_level']
            df.loc[index:index+2, 'iteration'] = iteration
            df_duplicates.loc[index, nan_columns] = 'NaN'
            df_mean = pd.concat([df_mean,df.loc[index:index+2],df_duplicates.loc[[index]]])
            iteration += 1
        else:
            df_incomplete = pd.concat([df_incomplete,df_duplicates.loc[index:index+2]]).drop_duplicates()
        return df_mean,df_incomplete
    