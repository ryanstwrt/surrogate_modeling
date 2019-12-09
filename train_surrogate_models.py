import pyearth
from pyearth import Earth
import sklearn
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.gaussian_process import kernels
from sklearn import neural_network
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
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
    self.hyper_parameters = {}
    self.models = {}
    self._initialize_models()
    self._initialize_hyper_parameters()

  def _initialize_hyper_parameters(self):
      self.hyper_parameters['lr'] = None
      self.hyper_parameters['pr'] = {'degree': (2,3,4,5,6,7)}
      self.hyper_parameters['mars'] = {'endspan_alpha':(0.01, 0.05, 0.1, 0.25),
                                       'minspan_alpha':(0.01, 0.05, 0.1, 0.25)}
      self.hyper_parameters['gpr'] = {'kernel': (kernels.RBF(), kernels.Matern(), kernels.RationalQuadratic()),
                                      'optmizer': ('fmin_l_bfgs_b')}
      self.hyper_parameters['ann'] = {'hidden_layer_size': (50,100,150),
                                      'activation': ('tanh', 'relu', 'logistic'),
                                      'solver': ('lbfgs', 'sgd', 'adam'),
                                      'alpha': (0.00001, 0.0001, 0.001)}
      self.hyper_parameters['rf'] = {'n_estimators': (10, 50, 100, 200)}

  def _initialize_models(self):
      self.models['lr']   = {'model': linear_model.LinearRegression()}
      #self.models['pr']   = None
      self.models['mars'] = {'model': Earth()}
      self.models['gpr']  = {'model': gaussian_process.GaussianProcessRegressor()}
      self.models['ann']  = {'model': neural_network.MLPRegressor(random_state=self.random)}
      self.models['rf']   = {'model': ensemble.RandomForestRegressor(random_state=self.random)}

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

  def _split_database(self):
    """Split the database into seperate training and test sets"""
    self.var_train, self.var_test, self.obj_train, self.obj_test = model_selection.train_test_split(self.ind_var, self.obj_var, random_state=self.random)

  def add_model(self, model_type, model):
      """Add a new model which is not pre-defined"""
      try:
          self.models[model_type] = {'model': model}
      except:
          print("Error: Model of type `{}' does not contain the correct format for function set_model. Please check sklearn for proper formatting.".format(model_type))

  def add_hyper_parameter(self, model_type, hyper_parameter):
      """Add a new hyper parameter to examine in the CV search space"""
      hyper_parameters = self.hyper_parameters[model_type]
      for hp in hyper_parameter:
           hyper_parameters[hp] = hyper_parameter[hp]

  def clear_surrogate_model(self):
      """Clear the the surrogate model of all values, but leave model types and hyper-parameters"""
      self.database = None
      self.ind_var = []
      self.obj_var = []
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

  def optimize_model(self, model_type, hyper_parameters):
    """Update the model by optimizing it's hyper_parameters"""
    self.set_model(model_type, hyper_parameters)

  def return_best_model(self):
    """Return the best model, based on R-squared value"""
    best_model = None
    best_score = 0
    for k in self.models.keys():
        score = self.models[k]['score']
        if  score > best_score:
            best_score = score
            best_model = k
    return best_model

  def predict(self, model_type, var):
      """Return the value for a prediction using the trained set"""
      scaled_var = self.var_train_scaler.transform(var)
      model = self.models[model_type]['fit']
      predictor = model.predict(scaled_var)
      inv_pred = self.obj_train_scaler.inverse_transform(predictor)
      return inv_pred

  def set_model(self, model_type, hyper_parameters=None):
    """Create a surrogate model"""
    model = self.models[model_type]['model']
    if hyper_parameters:
        model = model_selection.GridSearchCV(model, hyper_parameters)
    fit = model.fit(self.scaled_var_train,self.scaled_obj_train)
    score = model.score(self.scaled_var_test,self.scaled_obj_test)
    self.models[model_type].update({'model': model, 'fit': fit, 'score': score, 'hyper_parameters':hyper_parameters})

  def update_all_models(self):
    """update all models with new data"""
    for k in self.models.keys():
        if k != 'pr':
            self.set_model(k)

  def update_database(self, variables, objectives):
    """Update the database with new data
    Note: Make sure our ind and obj are staying together in the list"""
    for var, obj in zip(variables, objectives):
        self.ind_var.append(var)
        self.obj_var.append(obj)
    self._split_database()
    self._scale_data_sets()

  def update_model(self, model_type):
    """update a single model with new data"""
    self.set_model(model_type)
