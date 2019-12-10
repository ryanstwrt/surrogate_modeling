import train_surrogate_models as tm
import db_reshape as dbr
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")

var_tot, obj_tot = dbr.reshape_database(r'sfr_db_test.h5', ['height', 'smear', 'pu_content'], ['keff', 'void_coeff', 'doppler_coeff'])

sm = tm.Surrogate_Models()
data_col = ['r-squared', 'mean', 'std', 'index', 'hyper-parameters', 'cv_results']
models = ['lr', 'mars', 'gpr', 'ann', 'rf']
models = ['mars']
for model in models:
    sm_db = pd.DataFrame(columns = data_col)
    for i in range(1):
        sm.update_database(var_tot, obj_tot)
        sm.update_model(model)
        #sm.plot_validation_curve(model, 'hidden_layer_sizes', np.linspace(40,540,500,dtype=np.int16))
        #sm.plot_validation_curve(model, 'alpha', np.linspace(0.00001,0.1,500))
        #sm.plot_validation_curve(model, 'n_estimators', np.linspace(500,1000,500,dtype=np.int16))
        sm.plot_validation_curve(model, 'thresh', np.linspace(1E-5,1E-2,500))
        #sm.plot_validation_curve(model, 'min_samples_leaf', np.linspace(1,25,25,dtype=np.int16))
        #sm.plot_validation_curve(model, 'minspan_alpha', np.linspace(0.00001,0.1,250))
        sm.optimize_model(model)
        score = sm.models[model]['score']
        hp = sm.models[model]['hyper_parameters']
        cv = sm.models[model]['cv_results']
        append_dict = pd.DataFrame([[score,
                                    sm_db['r-squared'].mean(axis = 0),
                                    sm_db['r-squared'].std(axis = 0),
                                    i,
                                    hp,
                                    cv]], columns=data_col, index = [i])
        sm_db = sm_db.append(append_dict)
        sm.clear_surrogate_model()
    sm_db.to_csv('{}.csv'.format(model))

#sm_db.plot(kind='line', x='index', y='mean')
#sm_db.plot(kind='line', x='index', y='std')
#sm_db.plot(kind='line', x='index', y='r-squared')
plt.show()
