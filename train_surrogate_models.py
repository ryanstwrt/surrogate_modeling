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
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

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
      self.hyper_parameters['pr'] = {'poly__degree': (2,3,4,5,6,7)}
      self.hyper_parameters['mars'] = {'endspan_alpha': (0.01, 0.025, 0.05)}
      self.hyper_parameters['gpr'] = {'kernel':( kernels.RBF(), kernels.Matern(), kernels.RationalQuadratic())}
      self.hyper_parameters['ann'] = {'hidden_layer_sizes': (2,3,4,6,8,10,50),
                                      'activation': ('tanh', 'relu', 'logistic'),
                                      'solver': ('lbfgs', 'sgd'),
                                      'alpha': (0.00001, 0.0001, 0.001)}
      self.hyper_parameters['rf'] = {'n_estimators': (100, 200, 300)}

  def _initialize_models(self):
      self.models['lr']   = {'model': linear_model.LinearRegression()}
      self.models['pr']   = {'model': Pipeline([('poly', PolynomialFeatures()),
                                                ('linear', linear_model.LinearRegression(fit_intercept=False))])}
      self.models['mars'] = {'model': Earth()}
      self.models['gpr']  = {'model': gaussian_process.GaussianProcessRegressor(optimizer='fmin_l_bfgs_b')}
      self.models['ann']  = {'model': neural_network.MLPRegressor(random_state=self.random, solver='lbfgs', activation='logistic')}
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
          print("Error: Model of type '{}' does not contain the correct format for function set_model. Please check sklearn for proper formatting.".format(model_type))

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

  def optimize_model(self, model_type, hp=None):
    """Update the model by optimizing it's hyper_parameters"""
    if hp:
        hyper_parameters = hp
    else:
        hyper_parameters = self.hyper_parameters[model_type]
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

  def plot_validation_curve(self, model_type, hyper_parameter, hp_range):
      """Plot the validation curve for a model_type, given a particular hyper-parameter"""
      model = self.models[model_type]['model']

      train, test = model_selection.validation_curve(model, self.scaled_var_train, self.scaled_obj_train,
                                                     param_name=hyper_parameter, param_range=hp_range,n_jobs=20)
      tr_m = np.mean(train, axis = 1)
      tr_s = np.std(train, axis = 1)
      ts_m = np.mean(test, axis = 1)
      ts_s = np.std(test, axis = 1)

      plt.title("Validation Curve for {} with {}".format(model_type,hyper_parameter))
      plt.xlabel("{}".format(hyper_parameter))
      plt.ylabel("Score")
      lw = 2
      plt.plot(hp_range, tr_m, label="Training score", color="darkorange", lw=lw)
      plt.fill_between(hp_range, tr_m - tr_s, tr_m + tr_s, alpha=0.2, color="darkorange", lw=lw)
      plt.plot(hp_range, ts_m, label="Cross-validation score", color="navy", lw=lw)
      plt.fill_between(hp_range, ts_m - ts_s, ts_m + ts_s, alpha=0.2, color="navy", lw=lw)
      plt.legend(loc="best")
      plt.show()

  def set_model(self, model_type, hyper_parameters=None):
    """Create a surrogate model"""
    base_model = self.models[model_type]['model']
    hyper_model = self.models[model_type]['model']
    if hyper_parameters:
        hyper_model = model_selection.GridSearchCV(estimator=base_model, param_grid=hyper_parameters, refit=True, n_jobs=16)
        fit = hyper_model.fit(self.scaled_var_train,self.scaled_obj_train)
        score = hyper_model.score(self.scaled_var_test,self.scaled_obj_test)
        self.models[model_type].update({'fit': fit, 'score': score, 'hyper_parameters':hyper_model.best_params_, 'cv_results':hyper_model.cv_results_})
    else:
        fit = base_model.fit(self.scaled_var_train,self.scaled_obj_train)
        score = base_model.score(self.scaled_var_test,self.scaled_obj_test)
        self.models[model_type].update({'fit': fit, 'score': score, 'hyper_parameters': None, 'cv_results': None})

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
