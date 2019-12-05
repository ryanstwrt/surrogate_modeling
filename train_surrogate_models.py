import pyearth
from pyearth import Earth
import h5py
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn import neural_network
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

class Surrogate_Models(object):
  """Class to hold the surrogate models"""
  def __init__(self):
    self.database = None
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
      self.models['ann']  = {'model': neural_network.MLPRegressor()}
      self.models['rf']   = {'model': ensemble.RandomForestRegressor()}

  def update_database(self, variables, objectives):
    """Update the database with new data"""
    return

  def _split_database(self, variables, objectives):
    """Split the database into separate training and test sets"""
    self.var_train, self.var_test, self.obj_train, self.obj_test = train_test_split(variables, objectives)
    return

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
    self.models[model_type].update({'fit': fit, 'score': score})

  def update_model(self, model):
    """update a single model with new data"""
    self._split_database()
    self._scale_data_sets()

  def update_all_models(self):
    """update all models with new data"""
    self._split_database()
    self._scale_data_sets()
    for k,v in self.models.items():
        pass



def surrogate_evaluation(variables, objectives, test_var):

  test = np.array([[x for x in test_var.values()]])

  # Split our database into a training and testing set to ensure we get accurate R^2 values
  var_train, var_test, obj_train, obj_test = train_test_split(variables, objectives)

  # Scale our sets to ensure our surrogate models can handle the data
  var_train_scaler = StandardScaler()
  var_test_scaler = StandardScaler()
  obj_train_scaler = StandardScaler()
  obj_test_scaler = StandardScaler()
  var_train_scaler.fit(var_train)
  var_test_scaler.fit(var_test)
  obj_train_scaler.fit(obj_train)
  obj_test_scaler.fit(obj_test)

  # Transform out data into the scaled dimension
  scaled_var_train = var_train_scaler.transform(var_train)
  scaled_var_test  = var_test_scaler.transform(var_test)
  scaled_obj_train = obj_train_scaler.transform(obj_train)
  scaled_obj_test  = obj_test_scaler.transform(obj_test)

  # Transform our desired solution into the transform space
  scaled_test = var_train_scaler.transform(test)

  # Create model for Linear
  linear = linear_model.LinearRegression()
  linear_fit = linear.fit(scaled_var_train,scaled_obj_train)
  linear_score = linear.score(scaled_var_test,scaled_obj_test)
  linear_cv = cross_val_score(linear_fit, scaled_var_test, scaled_obj_test, scoring='neg_mean_squared_error')
  # Create model for Ridge
  ridge = linear_model.Ridge()
  ridge_fit = ridge.fit(scaled_var_train,scaled_obj_train)
  ridge_score = ridge.score(scaled_var_test,scaled_obj_test)
  ridge_cv = cross_val_score(ridge_fit, scaled_var_test, scaled_obj_test, scoring='neg_mean_squared_error')
  # Create model for MARS
  mars = Earth()
  parameters = {'endspan_alpha':(0.01, 0.05, 0.1, 0.25), 'minspan_alpha':(0.01, 0.05, 0.1, 0.25)}
  mars = model_selection.GridSearchCV(mars, parameters)
  mars_fit = mars.fit(scaled_var_train,scaled_obj_train)
  mars_score = mars.score(scaled_var_test,scaled_obj_test)
  mars_cv = cross_val_score(mars_fit, scaled_var_test, scaled_obj_test, scoring='neg_mean_squared_error')
  #Create model for Gaussian Process
  gpr = gaussian_process.GaussianProcessRegressor(optimizer='fmin_l_bfgs_b')
  parameters = {'alpha':(0.0001,0.001,0.01,0.1), 'kernel': (gaussian_process.kernels.Matern, gaussian_process.kernels.RationalQuadratic)}
  gpr_fit = gpr.fit(scaled_var_train,scaled_obj_train)
  gpr_score = gpr.score(scaled_var_test,scaled_obj_test)
  gpr_cv = cross_val_score(gpr_fit, scaled_var_test, scaled_obj_test, scoring='neg_mean_squared_error')
  # Create model for ANN
  ann = neural_network.MLPRegressor()
  parameters = {'solver':('lbfgs','sgd'), 'activation':('tanh','relu',), 'alpha':(0.0001,0.001,0.01,0.1)}
  #ann = model_selection.GridSearchCV(ann, parameters)
  ann_fit = ann.fit(scaled_var_train,scaled_obj_train)
  ann_score = ann.score(scaled_var_test,scaled_obj_test)
  ann_cv = cross_val_score(ann_fit, scaled_var_test, scaled_obj_test, scoring='neg_mean_squared_error')
  # Create model for random forrest (should be used for interpolation only!)
  rf = ensemble.RandomForestRegressor(n_estimators=100)
  rf_fit = rf.fit(scaled_var_train,scaled_obj_train)
  rf_score = rf.score(scaled_var_test,scaled_obj_test)
  rf_cv = cross_val_score(rf_fit, scaled_var_test, scaled_obj_test, scoring='neg_mean_squared_error')

  return [[linear_score, ridge_score, mars_score, gpr_score, ann_score, rf_score],
          [linear_cv, -linear_cv.mean(), linear_cv.std()]]#, rdige_cv, mars_cv, gpr_cv, ann_cv, rf_cv]]
