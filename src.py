import random
import string

import numpy as np

import shap

from tensorflow import keras as K

import vaex
import vaex.ml

sensor_data = '''
T2|Total temperature at fan inlet|°R
T24|Total temperature at LPC outlet|°R
T30|Total temperature at HPC outlet|°R
T50|Total temperature at LPT outlet|°R
P2|Pressure at fan inlet| psia
P15|Total pressure in bypass-duct|psia
P30|Total pressure at HPC outlet|psia
Nf|Physical fan speed|rpm
Nc|Physical core speed|rpm
epr|Engine pressure ratio (P50/P2)|--
Ps30|Static pressure at HPC outlet|psia
phi|Ratio of fuel flow to Ps30|pps/psi
NRf|Corrected fan speed|rpm
NRc|Corrected core speed|rpm
BPR|Bypass Ratio|--
farB|Burner fuel-air ratio|--
htBleed|Bleed Enthalpy|--
Nf_dmd|Demanded fan speed|rpm
PCNfR_dmd|Demanded corrected fan speed|rpm
W31|HPT coolant bleed|lbm/s
W32|LPT coolant bleed|lbm/s'''.strip().split('\n')
sensor_data = [k.split('|') for k in sensor_data]


def r2_keras(y_true, y_pred):
    SS_res =  K.backend.sum(K.backend.square(y_true - y_pred))
    SS_tot = K.backend.sum(K.backend.square(y_true - K.backend.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.backend.epsilon()))


def load_data_sources():
    # Read the partially transformed test data for providing explanations
    df_test_expl = vaex.open('./data/df_test_expl.hdf5')
    # Read the fully transformed test data - on a per sequence basis
    df_test_trans = vaex.open('./data/df_test_trans.hdf5')
    # Read the final prediction data - on a per engine basis
    df_test_final = vaex.open('./data/df_test_final.hdf5')

    # Load the custom keras model
    model_path = './model/rul_model.hdf5'
    nn_model = K.models.load_model(model_path, custom_objects={'r2_keras': r2_keras})

    # Load the background data for the Shap explainer
    bg_seq_array = np.load('./model/bg_seq_array_data.npy')
    explainer = shap.GradientExplainer(model=nn_model, data=bg_seq_array)

    # For illustration purposes create a fake airplane by randomly pairing two engines.
    df_airplane = create_airplane(df_engine=df_test_final)

    return df_test_expl, df_test_trans, df_test_final, explainer, df_airplane


def create_airplane(df_engine):
    '''For illustration purposes create a fake airplane by randomly pairing two engines.
    '''
    state = np.random.RandomState(seed=42)
    engine_index = state.choice(df_engine.unit_number.to_numpy(), len(df_engine), replace=False)

    ids = []
    random.seed(42)
    for i in range(len(df_engine)//2):
        num = int(random.uniform(100, 999))
        c1 = random.choice(string.ascii_uppercase)
        c2 = random.choice(string.ascii_uppercase)
        id = f'N{num}{c1}{c2}'
        ids.append(id)

    left = engine_index[::2]
    right = engine_index[1::2]
    df_airplane = vaex.from_arrays(left=left, right=right, tail_number=ids)

    df_airplane = df_airplane.join(df_engine, left_on='left', right_on='unit_number', rsuffix='_left')
    df_airplane = df_airplane.join(df_engine, left_on='right', right_on='unit_number', rsuffix='_right')
    df_airplane['RUL_pred_shortest'] = df_airplane.RUL_pred.minimum(df_airplane.RUL_pred_right)

    return df_airplane

