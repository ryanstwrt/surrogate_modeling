import train_surrogate_models as tm
import db_reshape as dbr
import warnings
warnings.filterwarnings("ignore")

var_tot, obj_tot = dbr.reshape_database(r'sfr_db_test.h5', ['height', 'smear', 'pu_content'], ['keff', 'void_coeff', 'doppler_coeff'])
print(type(var_tot), type(obj_tot))

surrogate_model = tm.Surrogate_Models()
surrogate_model._split_database(var_tot, obj_tot)
surrogate_model._scale_data_sets()
for m in ['lr', 'mars', 'gpr', 'ann', 'rf']:
    surrogate_model.set_model(m)
    print(surrogate_model.models[m]['score'])
