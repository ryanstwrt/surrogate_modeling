import train_surrogate_models as tm
import db_reshape as dbr
import warnings
warnings.filterwarnings("ignore")

var_tot, obj_tot = dbr.reshape_database(r'sfr_db_test.h5', {'height': 0.0, 'smear': 0.0, 'pu_content': 0.0}, ['keff', 'void_coeff', 'doppler_coeff'])
results = tm.surrogate_evaluation(var_tot, obj_tot, {'height':60, 'smear':51, 'pu_content':0.75})

print(results)
