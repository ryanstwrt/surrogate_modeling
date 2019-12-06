import pyearth
from pyearth import Earth
import sklearn
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn import neural_network
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class Surrogate_Models(object):
  """Class to hold the surrogate models"""
  def __init__(self):
    self.database = None
    self.ind_var = []
    self.obj_var = []

    self.random = None

    self.var_train = None
    self.var_test = None
    self.obj_test = None
    self.obj_train = None
    self.var_train_scaler = None
    self.var_test_scaler = None
    self.obj_train_scaler = None
    self.obj_test_scaler = None
    self.scaled_var_train = None
    self.scaled_var_test  = None
    self.scaled_obj_train = None
    self.scaled_obj_test  = None
    self.models = {}
    self._initialize_models()

  def _initialize_models(self):
      self.models['lr']   = {'model': linear_model.LinearRegression()}
      self.models['pr']   = None
      self.models['mars'] = {'model': Earth()}
      self.models['gpr']  = {'model': gaussian_process.GaussianProcessRegressor()}
      self.models['ann']  = {'model': neural_network.MLPRegressor(random_state=self.random)}
      self.models['rf']   = {'model': ensemble.RandomForestRegressor(random_state=self.random)}

  def update_database(self, variables, objectives):
    """Update the database with new data
    Note: Make sure our ind and obj are staying together in the list"""
    for var, obj in zip(variables, objectives):
        self.ind_var.append(var)
        self.obj_var.append(obj)
    self._split_database()
    self._scale_data_sets()

  def _split_database(self):
    """Split the database into seperate training and test sets"""
    self.var_train, self.var_test, self.obj_train, self.obj_test = train_test_split(self.ind_var, self.obj_var, random_state=self.random)

  def _scale_data_sets(self):
    """Scale the training and test databases to have a mean of 0
    aids the fitting techniques"""
    self.var_train_scaler = StandardScaler()
    self.var_test_scaler = StandardScaler()
    self.obj_train_scaler = StandardScaler()
    self.obj_test_scaler = StandardScaler()
    self.var_train_scaler.fit(self.var_train)
    self.var_test_scaler.fit(self.var_test)
    self.obj_train_scaler.fit(self.obj_train)
    self.obj_test_scaler.fit(self.obj_test)

    self.scaled_var_train = self.var_train_scaler.transform(self.var_train)
    self.scaled_var_test  = self.var_test_scaler.transform(self.var_test)
    self.scaled_obj_train = self.obj_train_scaler.transform(self.obj_train)
    self.scaled_obj_test  = self.obj_test_scaler.transform(self.obj_test)

  def set_model(self, model_type):
    """Create a surrogate model"""
    model = self.models[model_type]['model']
    fit = model.fit(self.scaled_var_train,self.scaled_obj_train)
    score = model.score(self.scaled_var_test,self.scaled_obj_test)
    self.models[model_type].update({'model': model, 'fit': fit, 'score': score})

  def add_model(self, model_type, model):
      """Add a new model which is not pre-defined"""
      try:
          self.models[model_type] = {'model': model}
      except:
          print("Error: Model of type `{}' does not contain the correct format for function set_model. Please check sklearn for proper formatting.".format(model_type))

  def update_model(self, model_type):
    """update a single model with new data"""
    self.set_model(model_type)

  def update_all_models(self):
    """update all models with new data"""
    for k in self.models.keys():
        if k != 'pr':
            self.set_model(k)
