import train_surrogate_models as tm
import numpy as np
import sklearn
from sklearn import linear_model
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def test_surrogate_model_init():
    sm = tm.Surrogate_Models()
    assert sm.database == None
    assert sm.ind_var == []
    assert sm.obj_var == []

    assert sm.var_test == None
    assert sm.var_train == None
    assert sm.obj_test == None
    assert sm.obj_train == None
    assert sm.var_test_scaler == None
    assert sm.var_train_scaler == None
    assert sm.obj_test_scaler == None
    assert sm.obj_train_scaler == None
    assert sm.scaled_var_train == None
    assert sm.scaled_var_test  == None
    assert sm.scaled_obj_train == None
    assert sm.scaled_obj_test  == None
    for m in ['lr', 'mars', 'gpr', 'ann', 'rf']:
        assert m in sm.models.keys()

def test_train_test_split():
    sm = tm.Surrogate_Models()
    sm.ind_var = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    sm.obj_var = [[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]
    sm._split_database()
    assert len(sm.var_test) == 2
    assert len(sm.var_train) == 4
    assert len(sm.obj_test) == 2
    assert len(sm.obj_train) == 4

def test_update_database():
    sm = tm.Surrogate_Models()
    variables = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    objectives = [[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]
    sm.update_database(variables, objectives)
    assert sm.ind_var == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert sm.obj_var ==[[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]
    variables = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    objectives = [[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]
    sm.update_database(variables, objectives)
    assert sm.ind_var == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert sm.obj_var == [[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]

def test_scale_datasets():
    sm = tm.Surrogate_Models()
    sm.ind_var = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    sm.obj_var = [[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]
    sm._split_database()
    sm._scale_data_sets()
    assert sm.var_train_scaler.mean_.all() == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).all()
    assert sm.var_train_scaler.var_.all() == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).all()
    assert sm.var_test_scaler.mean_.all() == np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]).all()
    assert sm.var_test_scaler.var_.all() == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).all()
    assert sm.obj_train_scaler.mean_.all() == np.array([[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]).all()
    assert sm.obj_train_scaler.var_.all() == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).all()
    assert sm.obj_test_scaler.mean_.all() == np.array([[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]).all()
    assert sm.obj_test_scaler.var_.all() == np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).all()

model = tm.Surrogate_Models()
variables = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
objectives = [[0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75], [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]]
model.update_database(variables, objectives)
model._initialize_models()


def test_initialize_models():
    models = model.models
    assert 'lr' in models
    assert 'pr' in models
    assert 'mars' in models
    assert 'gpr' in models
    assert 'ann' in models
    assert 'rf' in models

def test_linear_model():
    model.set_model('lr')
    linear_model = model.models['lr']
    assert linear_model['model'] != None
    assert linear_model['fit'] != None
    assert linear_model['score'] == 1.0

def test_poly_model():
    pass

def test_mars_models():
    "MARS score is given a nan, for this small of a dataset and not checked"
    model.set_model('mars')
    mars_model = model.models['mars']
    assert mars_model['model'] != None
    assert mars_model['fit'] != None

def test_grp_model():
    model.set_model('gpr')
    gpr_model = model.models['gpr']
    assert gpr_model['model'] != None
    assert gpr_model['fit'] != None
    assert gpr_model['score'] == 1.0

def test_ann_model():
    model.set_model('ann')
    ann_model = model.models['ann']
    assert ann_model['model'] != None
    assert ann_model['fit'] != None
    assert ann_model['score'] == 1.0

def test_ann_model():
    model.set_model('rf')
    rf_model = model.models['rf']
    assert rf_model['model'] != None
    assert rf_model['fit'] != None
    assert rf_model['score'] == 1.0

def test_add_model():
    ridge = linear_model.Ridge()
    model.add_model('ridge', ridge)
    assert 'ridge' in model.models

def test_set_added_model():
    ridge = linear_model.Ridge()
    model.add_model('ridge', ridge)
    model.set_model('ridge')
    ridge_model = model.models['ridge']
    assert ridge_model['model'] != None
    assert ridge_model['fit'] != None
    assert ridge_model['score'] == 1.0
