# %%
# basic dependencies
from __future__ import annotations
import numpy as np
import pandas as pd
import time
from typing import Optional

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

## To process mass data and fit sigmoid curves in automated initialization experiments
from scipy import signal
from scipy.optimize import curve_fit

# torch dependencies
import torch

tkwargs = {"dtype": torch.double, # set as double to minimize zero error for cholesky decomposition error
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")} # set tensors to GPU, if multiple GPUs please set cuda:x properly

torch.set_printoptions(precision=3)

###########

# botorch dependencies

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

import warnings

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

###########

# Platform configs
from pathlib import Path
import sys
REPO = 'LiqTransferOptimizer'
ROOT = str(Path().absolute()).split(REPO)[0]
sys.path.append(f'{ROOT}{REPO}')

# %%
class BO_LiqTransfer:
    """
    BO_LiqTransfer provides methods to perform Bayesian Optimization of liquid handling parameters of pipetting robots to transfer viscous liquids 

    ### Constructor
    Args:
        `liquid_name` (str): Name of the target liquid that requires liquid handling parameter optimization 
        `density` (float, Optional): Density of the target liquid.
    
    ### Attributes
    - 'features' (list): Column names from _data that will be used as features for the optimization
    - 'objectives' (list): Column names from _data that will be used as objectives for the optimization
    - 'bmax' (float): Factor that scales initial flow rate rate to maximum value 
    - 'bmin' (float): Factor that scales initial flow rate rate to minimum value 
    - 'mean_volumes' (list): List of volumes that will be used for optimization 
    
    
    ### Properties (with setter)
    - '_data' (DataFrame): DataFrame containing the data relevant to the experiments
    - '_param_dict' (dict): Dictionary containing the liquid handling parameters of the automated pipette
    - '_first_approximation' (float): Float value containing the approximated flow rate obtained in the first step of the optimization
    - 'platform' (device): Object pointing to automated platform used during optimization, should contain objects to control mass balance and 
    pipetting robot.
    - 'pipetteRobot' (device): Object pointing to pipetting Robot equipment used during optimization.
    - 'massBalance' (device): Object pointing to mass balance  equipment used during optimization.

    ### Properties (no setter)
    - '_latest_suggestion' (DataFrame): DataFrame containing the latest feature suggestions by the BO algorithm 
    - '_latest_volume' (int): Value of the latest volume transferred during the optimization
    - '_latest_acq_value' (float): Value of the latest suggestion acquisition value
    
    ### Methods
    ## Miscellaneous 
    - set_data: Takes a DataFrame calculates mean values of transfers and time to aspirate a 1000, and updates  property _data
    - update_data: Concatenates the DataFrame obtained during the last measurement with the previous data, then it sets to property _data 
    - data_from_csv: Loads data from a csv file path and sets to property _data.
    - df_last_measurement: Creates a DataFrame with the values from the last measurement. 
    - calibration_summary: Calculates statistics from calibration experiment
    - sigmoid: Defines sigmoid function used during mass balance controlled approximation of liquid flow in pipette tip.
    
    ## Bayesian Optimization
    - xy_split: Splits data into x and y values depending of the value of the attributes features and objectives
    - set_bounds: Set bounds of parametric space from attributes bmin and bmax
    - fit_surrogate: Fits GPR surrogate to training data
    - optimized_suggestions: Surrogate functions are sampled and likely gain is calculated by acquisition function

    ## Robotic platform control
    - cleanTip: Executes commands for the pipetting platform to perform 3 series of pipette blow out into a well.
    - gravimetric_transfer: Executes commands for the pipetting platform to perform 1 gravimetric test of a liquid transfer
    - obtainAproximateRate: Executes commands for the pipetting platform to estimate the liquid flow rate within the pipette tip
    - exploreBoundaries: Executes commands for the pipetting platform to perform five gravimetric transfers that use aspiration and dispense rates 
                        found at the boundary of the parametric space.
    - optimizeParameters: Executes commands for the pipetting platform to perform gravimetric transfers using aspiration
                        and dispense rates obtained through Bayesian optimization
    - calibrateParameters: Executes commands for the pipetting platform to perform gravimetric transfers using the 
                            aspiration and dispense rates with the best accuracy. 
    """

    def __init__(self, liquid_name:str, density:Optional[float] = None, pipette_brand:str = 'rline'):
        """
        Instantiate the class
        Args:
            - liquid_name (str): Name of the target liquid that require liquid handling 
            optimization
            - density (float): Density of the target liquid
        """

        self.liquid_name = liquid_name
        self.density = density
        self.features = ['aspiration_rate','dispense_rate']
        self.objectives = ['%error','time_asp_1000']
        self.bmax = 1.25
        self.bmin = 0.1
        self.mean_volumes = [1000,500,300]
        self._platform = None
        self.massBalance = None
        self.pipetteRobot = None

        self._data = pd.DataFrame(columns = ['liquid', 
                                             'pipette', 
                                             'volume', 
                                             'aspiration_rate', 
                                             'dispense_rate', 
                                             'delay_aspirate',
                                             'delay_dispense',
                                             'iteration',
                                             'liquid_level',
                                             'density', 
                                             'time', 
                                             'm', 
                                             '%error',
                                             'time_asp_1000',
                                             'acq_value'])
        self._first_approximation = None
        self._latest_acq_value = None
        self._latest_suggestion = None
        self._latest_volume = None
        self._param_dict = {
                            'aspiration_rate': None, 
                            'dispense_rate': None, 
                            'delay_aspirate': 10,  
                            'delay_dispense': 10
                            }
        self._pipette_brand = pipette_brand

     
    
    @property
    def platform(self):
        return self._platform
    
    @platform.setter
    def platform(self, setup):
        if str(type(setup)) != "<class 'controllably.misc.misc_utils.Setup'>":
            raise Exception('This is not a controllably.misc.misc_utils.Setup object.')

        for object in list(setup._asdict().values()):
            if object.__class__.__name__ == "LiquidMoverSetup":
                self.pipetteRobot = object

            if object.__class__.__name__ == "MassBalance":
                self.massBalance = object

        if self.pipetteRobot == None:
            print("""Setup Object assigned to self._platform does not contain a LiquidMoverSetup object.
                Please load a LiquidMoverSetup object using the self.pipetteRobot before starting optimization""")
            
        elif self.massBalance == None:
            print("""Setup Object assigned to self._platform does not contain a MassBalance object.
                Please load a MassBalance object using the self.massBalance  before starting optimization""")

        elif (self.massBalance == None and self.pipetteRobot == None): 
            print("""Setup Object assigned to self._platform does not contain a LiquidMoverSetup or a MassBalance object.
                Please load a a LiquidMoverSetup and MassBalance object using the self.pipetteRobot and self.massBalance before starting optimization""")    
        else:
            print(self.pipetteRobot)
            print(self.massBalance)

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, df):
        self.set_data(df)
        
    @property
    def first_approximation(self):
        return self._first_approximation

    @first_approximation.setter
    def first_approximation(self,flow_rate):
        self._first_approximation = flow_rate

    @property
    def latest_acq_value(self):
        return self._latest_acq_value
    
    @property
    def latest_suggestion(self):
        return self._latest_suggestion

    @property
    def latest_volume(self):
        return self._latest_volume

    @property
    def param_dict(self):
        return self._param_dict
    
    @param_dict.setter
    def param_dict(self,dictionary):
        for key,value in dictionary.items():
            try:
                self._param_dict[key] = value
            except:
                print('parameter is not in param_dict')
    
    
    ##Miscellaneous functions

    def set_data(self, df:pd.DataFrame):
        """ 
        Selects columns in a DataFrame that are also in property _data, calculates iteration, 
        time to aspirate 1000 µL and mean values for transfers using the same parameters that 
        have been performed for all values in attribute mean_volumes 

        Args:
            - df (pandas.DataFrame) : DataFrame containing the data of the gravimetric tests
            performed during an optimization
        """
       
        df = df.loc[:,self._data.columns].copy()
        nan_columns = df.columns.to_list()
        nan_columns = [column for column in nan_columns if column not in ('liquid',
        'pipette',
        'volume',
        'aspiration_rate',
        'dispense_rate',
        'delay_aspirate',
        'delay_dispense',
        'density',
        '%error',
        'time_asp_1000',
        'acq_value',
        'iteration',
        'liquid_level')]

        df['time_asp_1000'] = 1000/df['aspiration_rate'] + 1000/df['dispense_rate'] + df['delay_aspirate'] + df['delay_dispense']

        iteration = 1
        
        if 'acq_value' not in df.columns:
            df['acq_value'] = None

        if df.loc[:,self.features].duplicated().sum()==0:
            df_mean = df
        else:
            df_duplicates = df.where(df.duplicated(self.features,keep=False)==True).dropna(how='all')
            df_incomplete = df.where(df.duplicated(self.features,keep=False)==False).dropna(how='all')
            df_mean = pd.DataFrame(columns= df.columns)
            for index,values in df_duplicates.drop_duplicates(self.features).iterrows():
                if len(df_duplicates.loc[index:index+2]) == len(self.mean_volumes):
                    mean_error =df_duplicates.loc[index:index+2,'%error'].abs().mean()
                    df_duplicates.loc[index,'%error'] = -mean_error
                    df_duplicates.loc[index, 'volume'] ='mean'+str(self.mean_volumes)
                    df_duplicates.loc[index, 'iteration']= iteration
                    df_duplicates.loc[index, 'liquid_level']= df.loc[index+2,'liquid_level']
                    df.loc[index:index+2, 'iteration'] = iteration
                    df_duplicates.loc[index, nan_columns]= 'NaN'
                    df_mean = pd.concat([df_mean,df.loc[index:index+2],df_duplicates.loc[[index]]])
                    iteration +=1 
                else:
                    df_incomplete = pd.concat([df_incomplete,df_duplicates.loc[index:index+2]]).drop_duplicates()
            df_mean = pd.concat([df_mean,df_incomplete])
            df_mean = df_mean.reset_index(drop=True)    
        self._data = df_mean
 

    def update_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """ 
        Concatenates DataFrame from one gravimetric test with existing data in _data and
        sets the concatenated DataFrame to _data
        Args:
            - df (pandas.DataFrame) : DataFrame containing the data of one gravimetric test
        """
        self._latest_volume = df['volume'].iloc[-1]
        updated_data = pd.concat([self._data,df],ignore_index=True)
        self.set_data(updated_data)
        return self._data


    def data_from_csv(self, file_name:str):
        """ 
        Loads a csv file and sets to _data
        Args:
            - file_name (str) : Path of csv file
        """
        data = pd.read_csv(file_name)
        if self._pipette_brand != 'rline':
            data = data.loc[:,['liquid','pipette','volume','aspiration_rate','dispense_rate','blow_out_rate','delay_aspirate','delay_dispense','delay_blow_out','%error']]
        self.set_data(data)

    
    def df_last_measurement(self, error:float, volume:float = 1000) -> pd.DataFrame:
        """ 
        Returns a DataFrame object containing the latest measured error for a gravimetric test
        Args:
            - error (float) : value of relative error of transfer from gravimetric test
            - volume (int) : Volume transferred in gravimetric test
        """
        self._latest_volume = volume
        updated_data = pd.concat([self._data,self._data.iloc[[-1]]],ignore_index=True)
        last_measurement_data = self._data.iloc[-1].copy()
        last_measurement_data.loc[updated_data.last_valid_index(),'volume'] = self._latest_volume
        last_measurement_data.loc[updated_data.last_valid_index(),'aspiration_rate']  = self._latest_suggestion['aspiration_rate']
        last_measurement_data.loc[updated_data.last_valid_index(),'dispense_rate']  = self._latest_suggestion['dispense_rate']
        last_measurement_data.loc[updated_data.last_valid_index(),'%error'] = error
        return last_measurement_data
    
    def calibration_summary(self, df:pd.DataFrame) -> pd.DataFrame:
        """ 
        Returns a DataFrame object containing the summary of mean transfer errors, standard deviations,
        time to transfer 1000 µL and iteration of the tested parameters in the calibration procedure
        Args:
            - df (pandas.DataFrame) : DataFrame containing values of the gravimetric tests performed 
            during calibration test of 
        """
        if 'volume_transferred' and 'volume_error' and 'time_asp_1000' not in df.columns:
            df['volume_transferred'] = (df['m']/df['density'])*1000
            df['volume_error'] = df['volume_transferred'] - df['volume']
            df['time_asp_1000']=1000/df['aspiration_rate'] + 1000/df['dispense_rate'] + df['delay_aspirate'] + df['delay_dispense']             

        df_summary_all = pd.DataFrame()

        for volume in self.mean_volumes:
            df_experiment_v = df.where(df['volume'] == volume).dropna(how='all')
            df_summary = pd.DataFrame(columns = (f'Mean transfer volume for {volume} µL [µL]', f'Mean transfer volume error of {volume} µL [µL]', f'Mean relative error for transfer of {volume} µL [%]', f'Standard deviation for transfer of {volume} µL [µL]', f'Relative standard deviation for transfer of {volume} µL [%]') )
            data = [df_experiment_v['volume_transferred'].mean(), df_experiment_v['volume_error'].mean(), df_experiment_v['%error'].mean(), df_experiment_v['volume_transferred'].std(), (df_experiment_v['volume_transferred'].std() / df_experiment_v['volume_transferred'].mean() * 100)]
            df_summary.loc[df['liquid'].iloc[0]] = data
            df_summary_all = pd.concat([df_summary_all, df_summary], axis = 1)
        return df_summary_all 

    @staticmethod
    def sigmoid(x:np.ndarray, K:float, x0:float, B:float, v:float, A:float) -> np.ndarray:
        """ 
        Returns a value based on the evaluation of a generalized logistic function
        Args:
            - x (float) : Value to be evaluated
            - K (float) : Upper asymptote
            - A (float) : Lower asymptote
            - x0 (float) : Starting value for a series of x
            - B (float) : Growth constant
            - v (float) : Asymmetry factor
        """
        y = (K-A) / (1 + np.exp(B*(x-x0)))**(1/v) + A
        return y
    
    #BO relevant functions                            

    def xy_split(self) -> tuple[np.ndarray, np.ndarray]:
        """ 
        Returns pandas.DataFrames for features (x) and objectives (y) from _data to 
        train a ML algorithm  
        """
        df_train = self._data.where(self._data['volume']=='mean'+str(self.mean_volumes)).dropna(how='all')
        x_train = df_train[self.features]
        y_train = df_train[self.objectives]
        return x_train,y_train


    def set_bounds(self, x_train:np.ndarray) -> torch.Tensor:
        """
        Set the bounds for the parametric space in terms 
        of attributes bmin and bmax
        """
        return torch.vstack([x_train[0]*self.bmin, x_train[0]*self.bmax])


    def fit_surrogate(self) -> tuple[ModelListGP, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a GPR model for each value in attribute objectives using each value
        of attribute features, a reference point for each objective, normalized 
        training data and the bounds of the parametric space 
        """
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
        """
        Returns a DataFrame with the ten suggestions of features
        that the DataFrame function computed for most likely gain. 
        Sets _latest_acq_value and _latest_suggestion using the top suggestions
        of the DataFrame
        """
        if random_state != None:
            torch.manual_seed(random_state) 
        standard_bounds = torch.zeros(2, len(self.features), **tkwargs)
        standard_bounds[1] = 1
        model1, ref_point, train_x_gp, problem_bounds = self.fit_surrogate()
        acq_func1 = qNoisyExpectedHypervolumeImprovement(
        model=model1,
        ref_point=ref_point, # for computing HV, must flip for BoTorch
        X_baseline=train_x_gp, # feed total list of train_x for this current iteration
        sampler=SobolQMCNormalSampler(sample_shape=512),  # determines how candidates are randomly proposed before selection
        objective=IdentityMCMultiOutputObjective(outcomes=np.arange(len(self.objectives)).tolist()), # optimize first n_obj col 
        prune_baseline=True, cache_pending=True)  # options for improving qNEHVI, keep these on
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
        # unormalize our training inputs back to original problem bounds
        new_x =  unnormalize(qnehvi_x, bounds=problem_bounds)
        new_x = pd.DataFrame(new_x.numpy(),columns= self.features)
        new_x['acq_value'] = sorted(acq_value_list, reverse=True)[:12]
        self._latest_suggestion = new_x[self.features].iloc[0]
        self._latest_acq_value = new_x['acq_value'].iloc[0]
        return new_x


    
    ### Methods for controlling robotic platform

    def cleanTip(self, well, speed_factor:float = 0.2, repetitions:int = 9):
        """
        Executes commands to clean pipette tip using a cycles of blowouts
        and plunger homing
        Args:
            - well (labware.well) : Well of labware where the clean tip procedure 
            will be performed
            - 
            - repetitions (int) : Number of repetitions for the clean tip procedure
        """
        
        self.pipetteRobot.mover.safeMoveTo(well.top)
        
        for i in range(repetitions):
            self.pipetteRobot.liquid.blowout(home=False) 
            time.sleep(5)
            self.pipetteRobot.liquid.home()
            self.pipetteRobot.touchTip(well,speed_factor = speed_factor)
            time.sleep(5)

    
    def gravimetric_transfer(self, volume:int, liquid_level:float, source_well, balance_well) -> tuple[float, pd.DataFrame]:
        """
        Executes commands for one gravimetric test of a combination of liquid 
        handling parameters defined by the property param_dict. 
        Returns updated liquid level and a DataFrame containing data from transfer
        Args:
            - volume (int) : Target volume for transfer
            - liquid_level (float) : Height of liquid column in source well
            - source_well (labware.well) : Well of labware where the target liquid 
            is stored
            - balance_well (labware.well) : Well of labware placed on top of automated mass balance
        """
        #liquid transfer
        #transfer start
        start = time.time() 

        #aspirate step
        self.pipetteRobot.mover.safeMoveTo(source_well.from_bottom((0,0,liquid_level-5))) 
        self.pipetteRobot.liquid.aspirate(volume, speed=self._param_dict['aspiration_rate'] )
        time.sleep(self._param_dict['delay_aspirate'])

        self.pipetteRobot.touchTip(source_well) 

        #dispense step
        self.pipetteRobot.mover.safeMoveTo(balance_well.from_top((0,0,-5))) 
        self.massBalance.tare() 
        self.massBalance.clearCache() 
        self.massBalance.toggleRecord(True) 
        time.sleep(5)
        self.pipetteRobot.liquid.dispense(volume,speed=self._param_dict['dispense_rate'] )
        time.sleep(self._param_dict['delay_dispense'])

        #transfer termination
        finish = time.time() 
        time_m = finish - start

        self.pipetteRobot.mover.safeMoveTo(source_well.top) 
        time.sleep(5)
        self.massBalance.toggleRecord(False) 

        #do blowout
        
        self.cleanTip(source_well)

        #record transfer values 
        #calculating mass error functions
        m = (self.massBalance.buffer_df.iloc[-10:,-1].mean()-self.massBalance.buffer_df.iloc[:10,-1].mean())/1000 
        error = (m-self.density*volume/1000)/(self.density/1000*volume)*100
        
        #change liquid levels
        liquid_level = liquid_level - 2*m/self.density   
        
        #making new DataFrame + filling it in
        df = pd.DataFrame(columns=self._data.columns)            
        
        df = pd.concat([df,pd.DataFrame({
            "liquid": self.liquid_name,
            'pipette': 'rLine1000',
            "volume": volume,
            "aspiration_rate": self._param_dict['aspiration_rate'],
            "dispense_rate": self._param_dict['dispense_rate'], 
            "delay_aspirate" : self._param_dict['delay_aspirate'],  
            "delay_dispense" : self._param_dict['delay_dispense'],
            'iteration': 'NaN',
            "liquid_level" : liquid_level,
            "density" : self.density,
            "time" : time_m,
            "m": m,
            "%error": error,
            "time_asp_1000" : 'NaN',
            "acq_value": self._latest_acq_value
            },index=[0])],ignore_index=True)
        
        return liquid_level, df

    def obtainAproximateRate(self, initial_liquid_level_balance:float, balance_well, speed:float = 265, file_name:Optional[str]=None):
        """
        Executes commands to obtain first approximation of flow rate for optimization protocol.
        Sets _first_approximation property and saves file with the recorded mass/time data
        Args:
            - initial_liquid_level_balance (float) : Height of liquid column in well on top of balance
            - balance_well (labware.well): Well of labware placed on top of automated mass balance
            - speed (float) : Speed of plunger movement defined as flow rate [µL/s] 
            - file_name (str, optional) : Path to save the recorded mass/time data
        """
        liquid_level = initial_liquid_level_balance
        
        if self.pipetteRobot.liquid.isTipOn()== False:
            self.pipetteRobot.attachTip()
        
        self.pipetteRobot.mover.safeMoveTo(balance_well.from_bottom((0,0,liquid_level-5)),descent_speed_fraction=0.25)
        #Starting balance measurement
        time.sleep(5)
        self.massBalance.zero(wait=5)
        self.massBalance.clearCache()
        self.massBalance.toggleRecord(on=True)
        time.sleep(15)

        self.pipetteRobot.liquid.aspirate(1000, speed=speed)

        #Switching the balance off after change in mass is less than 0.05
        while True:
            data = self.massBalance.buffer_df.copy()
            data['Mass_smooth']= signal.savgol_filter(data['Mass'],91,1)
            data['Mass_derivative_smooth']=data['Mass_smooth'].diff()
            condition=data['Mass_derivative_smooth'].rolling(30).mean().iloc[-1]
            if condition>-0.05:
                break
        print('loop stopped')
        self.massBalance.toggleRecord(on=False)

        self.pipetteRobot.mover.moveTo(balance_well.from_top((0,0,-5)))


        #using data from balance buffer_df above, calculate time in seconds and mass derivatives
        data['ts'] = data['Time'].astype('datetime64[ns]').values.astype('float') / 10 ** 9
        data['ts']= data['ts']-data['ts'][0]
        data_fit = data.where(data['ts']>10).dropna()
        data_fit['Mass']=data_fit['Mass']-data_fit['Mass'].iloc[0]
        data_fit['Mass_smooth'] = data_fit['Mass_smooth']-data_fit['Mass_smooth'].iloc[0]

        p0 = [min(data_fit['Mass']), np.median(data_fit['ts']),1,1,max(data_fit['Mass'])+30]
        
        popt, pcov = curve_fit(self.sigmoid, data_fit['ts'], data_fit['Mass'],p0)

        mass_sigmoid = self.sigmoid(data_fit['ts'],popt[0],popt[1],popt[2],popt[3],popt[4])

        data_fit.loc[data_fit.index[0]:,'Mass_sigmoid'] = mass_sigmoid

        flow_rate = mass_sigmoid.diff()/data_fit.loc[data_fit.index[0]:,'ts'].diff()

        data_fit.loc[data_fit.index[0]:,'Flow_rate']=flow_rate

        flow_rate = mass_sigmoid.diff()/data_fit.loc[data_fit.index[0]:,'ts'].diff()

        flow_rate_max = flow_rate.min()

        flow_rate_98 = data_fit.where(data_fit['Flow_rate']<(0.05*flow_rate_max)).dropna()

        time_start, time_final = flow_rate_98.iloc[0].loc['ts'],flow_rate_98.iloc[-1].loc['ts']

        initial_flow_rate_aspirate = 1000/(time_final-time_start)
        
        self._first_approximation = initial_flow_rate_aspirate 
        
        #switching balance off and saving csv
        if file_name != None:
            data_fit.to_csv(file_name, index=False)

        self.pipetteRobot.liquid.dispense(1000,speed= self._first_approximation)


    def exploreBoundaries(self, initial_liquid_level_source:float, source_well, balance_well):
        """
        Executes commands for 5 gravimetric test using a combination of liquid 
        handling parameters derived from the property first_approximation and 
        attributes bmin, bmax. Each combination of liquid handling parameters gravimetric 
        test is repeated for eac value of attribute mean volumes
        Args
            - initial_liquid_level_source (float) : Height of liquid column in source well
            - source_well (labware.well) : Well of labware where the target liquid 
            is stored
            - balance_well (labware.well) : Well of labware placed on top of automated mass balance
        """


        if type(self._data) == None:
            df = pd.DataFrame(columns = ['liquid', 'pipette', 'volume', 'aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense','iteration','liquid_level','density', 'time', 'm', '%error','time_asp_1000','acq_value'])
            df = df.astype({'liquid':str,'pipette':str})
            self.set_data(df)
        
        liquid_level = initial_liquid_level_source

        #Check if new tip is required
        if self.pipetteRobot.liquid.isTipOn()== False:
            self.pipetteRobot.attachTip()

        volumes_list = self.mean_volumes
        
        #NOT TO BE CHANGED
        counter = 1
        iterations = 5
        #while loop
        while counter <= iterations:
            #hardcoding aspirate and dispense rates:
            if counter == 1:
                self._param_dict['aspiration_rate'] = self._first_approximation
                self._param_dict['dispense_rate'] = self._first_approximation
            if counter == 2:
                self._param_dict['aspiration_rate'] = self._first_approximation*self.bmax
                self._param_dict['dispense_rate'] = self._first_approximation*self.bmax
            if counter == 3:
                self._param_dict['aspiration_rate'] = self._first_approximation*self.bmax
                self._param_dict['dispense_rate'] = self._first_approximation*self.bmin
            if counter == 4:
                self._param_dict['aspiration_rate'] = self._first_approximation*self.bmin
                self._param_dict['dispense_rate'] = self._first_approximation*self.bmax
            if counter == 5:
                self._param_dict['aspiration_rate'] = self._first_approximation*self.bmin
                self._param_dict['dispense_rate'] = self._first_approximation*self.bmin


            #for loop
            for volume in volumes_list:
                #liquid transfer
                liquid_level,df = self.gravimetric_transfer(volume,liquid_level,source_well,balance_well)
                
                m=df.m.iloc[-1]

                self.set_data(pd.concat([self._data,df]).reset_index(drop=True))
                #printing checks
                print("LIQUID LEVEL: " +str(liquid_level) + "   LIQUID CHANGE: " +str(2*m/self.density) + "   ITERATION: " + str(counter) + ", " + "VOLUME: " + str(volume))    

                #liquid level checks
                if  (m/(volume*self.density) > 2) or (m/self.density < 0):
                    print('Balance measurement error')
                    break
                if (liquid_level > initial_liquid_level_source) or (liquid_level < 6):
                    print('Liquid level too high or too low')
                    break
                

            counter += 1


    def optimizeParameters(self, initial_liquid_level_source:float, source_well, balance_well, iterations:int = 5, file_name:Optional[str] = None):
        """
        Executes commands for n iterations of gravimetric test using a combination of liquid 
        handling parameters suggested by a BO algorithm. Each test is repeated for each
        value of attribute mean volumes.
        Args
            - initial_liquid_level_source (float) : Height of liquid column in source well
            - source_well (labware.well) : Well of labware where the target liquid 
            is stored
            - balance_well (labware.well) : Well of labware placed on top of automated mass balance
            - iterations (int) : Number of optimization iterations
            - file_name (str, optional) : Path to save the values recorded in property data during the optimization
        """


        liquid_level = initial_liquid_level_source

        #Check if new tip is required
        if self.pipetteRobot.liquid.isTipOn()== False:
            self.pipetteRobot.attachTip()

        volumes_list = self.mean_volumes
        
        #NOT TO BE CHANGED
        counter = 1
       
        #while loop
        while counter <= iterations:
            #getting botorch suggestions + implementing it in liquids_dict
            self.optimized_suggestions()

            self._param_dict['aspiration_rate'] = self._latest_suggestion['aspiration_rate']
            self._param_dict['dispense_rate'] = self._latest_suggestion['dispense_rate']
            #for loop
            for volume in volumes_list:
                #liquid transfer
                liquid_level,df = self.gravimetric_transfer(volume,liquid_level,source_well,balance_well)
                
                m=df.m.iloc[-1]

                self.set_data(pd.concat([self._data,df]).reset_index(drop=True))
                #printing checks
                print("LIQUID LEVEL: " +str(liquid_level) + "   LIQUID CHANGE: " +str(2*m/self.density) + "   ITERATION: " + str(counter) + ", " + "VOLUME: " + str(volume))    


                
                #printing checks
                print("LIQUID LEVEL: " +str(liquid_level) + "   LIQUID CHANGE: " +str(2*m/self.density) + "   ITERATION: " + str(counter) + ", " + "VOLUME: " + str(volume))    

                #liquid level checks
                if  (m/(volume*self.density) > 2) or (m/self.density < 0):
                    print('Balance measurement error')
                    break
                if (liquid_level > initial_liquid_level_source) or (liquid_level < 6):
                    print('Liquid level too high or too low')                   
                    break
            
            counter += 1
        if file_name != None:
            self._data.to_csv(file_name, index=False)


    def calibrateParameters(self, initial_liquid_level_source:float, source_well, balance_well, iterations:int = 10, file_name:Optional[str] = None):
        """
        Executes commands for n iterations of gravimetric test using the best recorded 
        combination of liquid handling parameters in property data. 
        The test is performed n times for each value of attribute mean volumes.
        Args
            - initial_liquid_level_source (float) : Height of liquid column in source well
            - source_well (labware.well) : Well of labware where the target liquid 
            is stored
            - balance_well (labware.well) : Well of labware placed on top of automated mass balance
            - iterations (int) : Number of transfer per volume calibrated
            - file_name (str, optional) : Path to save csv with the values recorded during calibration 
            and the statistical summary
        """

        liquid_level = initial_liquid_level_source

        # Check if new tip is required
        if self.pipetteRobot.liquid.isTipOn()== False:
            self.pipetteRobot.attachTip()

        volumes_list = self.mean_volumes
        
        #NOT TO BE CHANGED
     
        
        mean_average_data = self._data.where(self._data.volume == 'mean'+str(self.mean_volumes))
        mean_average_data = mean_average_data.where(mean_average_data.iteration>5).dropna()
        best_parameter_index = mean_average_data[mean_average_data['%error']==mean_average_data['%error'].max()].index

        self._param_dict['aspiration_rate'] = self._data.loc[best_parameter_index,'aspiration_rate'].values[0]
        self._param_dict['dispense_rate'] = self._data.loc[best_parameter_index,'dispense_rate'].values[0]
        
        calibration_df = pd.DataFrame(columns = ['liquid', 'pipette', 'volume', 'aspiration_rate', 'dispense_rate', 'delay_aspirate', 'delay_dispense','liquid_level','density', 'm', '%error'])
        
        #for loop
            
        for volume in volumes_list:
            counter = 1
            # #while loop
            while counter <= iterations:
             
                #liquid transfer
                liquid_level,df = self.gravimetric_transfer(volume,liquid_level,source_well,balance_well)
                
                calibration_df = pd.concat([calibration_df,df[calibration_df.columns]]).reset_index(drop=True)

                m=df.m.iloc[-1]

                #printing checks
                print("Mass: "+str(m)+"LIQUID LEVEL: " +str(liquid_level) + "   LIQUID CHANGE: " +str(2*m/self.density) + "   ITERATION: " + str(counter) + ", " + "VOLUME: " + str(volume))    

                #liquid level checks
                if (m/(volume*self.density) > 2) or (m/self.density < 0):
                    print('Balance measurement error')
                    break
                if (liquid_level > initial_liquid_level_source) or (liquid_level < 6):
                    print('Liquid level too high or too low')
                    break

                counter += 1
        
        calibration_df['volume_transferred'] = calibration_df['m']/calibration_df['density']*1000
        calibration_df['volume_error'] = calibration_df['volume_transferred'] - calibration_df['volume']
        calibration_df['time_asp_1000'] = 1000/calibration_df['aspiration_rate'] + 1000/calibration_df['dispense_rate'] + calibration_df['delay_aspirate'] + calibration_df['delay_dispense']       
        
        calibration_summary_df= self.calibration_summary(calibration_df)

        if file_name != None:
            calibration_df.to_csv(file_name, index=False)
            calibration_summary_df.to_csv(file_name[:-4]+'_summary.csv')
        
