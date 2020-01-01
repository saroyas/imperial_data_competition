# ==================== WHAT TO FILL OUT IN THIS FILE ===========================
"""
There are 3 sections that should be filled out in this file

1. Package imports below
2. The PricingModel class
3. The load function

There are also three other sections at the end of the file that can
safely be ignored. These are:

    - Probability calibration. An optional step to ensure
      probabilities predicted by your model are calibrated
      (see docstring)

    - Consistency check to make sure the code that you submit is the
      same as your trained model.

    - A main section that runs this file and produces the the trained model file
      This also checks whether your load_model function works properly
"""


# ========================== 1. PACKAGE IMPORTS ================================
# Include your package imports here

import hashlib
import numpy as np
import pandas as pd
import pickle

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

#====
from sklearn import linear_model
import numpy as np
import sklearn as sk
from tensorflow import keras
import pandas as pd
import scipy
import tensorflow as tf

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# ========================= 2. THE PRICING MODEL ===============================
class PricingModel():
    """
    This is the PricingModel template. You are allowed to:
    1. Add methods
    2. Add init variables
    3. Fill out code to train the model

    You must ensure that the predict_premium method works!
    """

    def __init__(self,):

        # =============================================================
        # This line ensures that your file (pricing_model.py) and the saved
        # model are consistent and can be used together.
        #self._init_consistency_check()  # DO NOT REMOVE
        self.model = linear_model.LinearRegression()

    def _preprocessor(self, X_raw):
        """

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : Pandas dataframe
            This is the raw data features excluding claims information 

        Returns
        -------
        X: Pandas DataFrame
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        columns_x = ['pol_bonus', 'pol_coverage', 'pol_duration',
                     'pol_sit_duration', 'pol_pay_freq', 'pol_payd', 'pol_usage',
                     'drv_drv2', 'drv_age1', 'drv_age2', 'drv_sex1',
                     'drv_sex2', 'drv_age_lic1', 'drv_age_lic2', 'vh_age', 'vh_cyl',
                     'vh_din', 'vh_fuel', 'vh_make', 'vh_sale_begin',
                     'vh_sale_end', 'vh_speed', 'vh_type', 'vh_value', 'vh_weight',
                     'town_mean_altitude', 'town_surface_area', 'population', 'city_district_code',
                     ]

        x_data_un = X_raw[columns_x]

        cat_to_int_dict = {'pol_coverage': {'Mini': 0, 'Median1': 1, 'Median2': 2, 'Maxi': 3},
                           'pol_pay_freq': {'Monthly': 0, 'Quarterly': 1, 'Biannual': 2, 'Yearly': 3},
                           'pol_payd': {'No': 0, 'Yes': 1},
                           'pol_usage': {'Retired': 0, 'WorkPrivate': 1, 'Professional': 2, 'AllTrips': 3},
                           'drv_drv2': {'No': 0, 'Yes': 1},
                           'drv_sex1': {'F': 0, 'M': 1},
                           'drv_sex2': {'F': -1, 'M': 1, None: 0},
                           'vh_type': {'Tourism': 0, 'Commercial': 1, }
                           }

        def car_make_categories(car_make):
            if car_make in ['RENAULT', 'PEUGEOT', 'CITROEN', 'VOLKSWAGEN', 'FORD', 'MERCEDES BENZ']:
                return car_make
            else:
                return 'OTHER'

        def missing_geo_data(x):
            if x:
                return 1
            else:
                return 0

        def zero_vh_weight(weight, avg_weight):
            if weight < 100:
                return avg_weight
            else:
                return weight

        x_data_f = x_data_un.replace(cat_to_int_dict, inplace=False)

        x_data_f.vh_make = x_data_f['vh_make'].apply(lambda x: car_make_categories(x))

        ##MAKE INDEP
        avg_vh_weight = x_data_f['vh_weight'].mean()
        x_data_f.vh_weight = x_data_f['vh_weight'].apply(lambda x: zero_vh_weight(x, avg_vh_weight))

        vh_make_cols = pd.get_dummies(x_data_f.vh_make)
        vh_fuel_cols = pd.get_dummies(x_data_f.vh_fuel)
        city_dist_cols = pd.get_dummies(x_data_f.city_district_code)
        x_data_f = x_data_f.drop(['vh_fuel', 'vh_make', 'city_district_code'], axis=1)

        x_data_f['geoNA'] = (x_data_un['population'].isnull()).apply(lambda x: missing_geo_data(x))

        x_data_f = pd.concat([x_data_f, vh_make_cols, vh_fuel_cols, city_dist_cols], axis=1, sort=False)

        #NEED TO CHANGE MEANS SO THAT ITS INDEPENDENT
        means = x_data_f.mean()
        x_data_f = x_data_f.fillna(means)

        #SIMILIAARLY I NEED TO CHANGE THIS SCALER SO THAT ITS INDEPENDENT
        #scaler = preprocessing.MinMaxScaler()
        #X = pd.DataFrame(scaler.fit_transform(x_data_f), columns=x_data_f.columns, index=x_data_f.index)
        X = x_data_f

        return X


    def fit(self, X_raw, y_made_claim, y_claims_amount):
        """
        Here you will use the fit function for your pricing model.

        Parameters
        ----------
        X_raw : Pandas DataFrame
            This is the raw data features excluding claims information 
        y_made_claim : Pandas DataFrame
            A one dimensional binary array indicating the presence of accidents
        y_claims_amount: Pandas DataFrame
            A one dimensional array which records the severity of claims (this is
            zero where y_made_claim is zero).

        """

        # YOUR CODE HERE

        # Remember to include a line similar to the one below
        X_clean = self._preprocessor(X_raw)

        (self.model).fit(X_clean, y_claims_amount)

        return None


    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Parameters
        ----------
        X_raw : Pandas DataFrame
            This is the raw data features excluding claims information

        Returns
        -------
        Pandas DataFrame
            A one dimensional array of the same length as the input with
            values corresponding to the offered premium prices
        """
        # =============================================================
        # You can include a pricing strategy here
        # For example you could scale all your prices down by a factor

        # YOUR CODE HERE

        # Remember to include a line similar to the one below
        X_clean = self._preprocessor(X_raw)

        predicted_claim_amounts = (self.model).predict(X_clean)

        return predicted_claim_amounts


    def save_model(self):
        """
        Saves a trained model to pricing_model.p.
        """

        # =============================================================
        # Default : pickle the trained model. Change this (and the load
        # function, below) only if the library you used does not support
        # pickling.
        with open('pricing_model.p', 'wb') as target:
            pickle.dump(self, target)


    def _init_consistency_check(self):
        """
        INTERNAL METHOD: DO NOT CHANGE.
        Ensures that the saved object is consistent with the file.
        This is done by saving a hash of the module file (pricing_model.py) as
        part of the object.
        For this to work, make sure your source code is named pricing_model.py.
        """
        try:
            with open('pricing_model.py', 'r') as ff:
                code = ff.read()
            m = hashlib.sha256()
            m.update(code.encode())
            self._source_hash = m.hexdigest()
        except Exception as err:
            print('There was an error when saving the consistency check: '
                  '%s (your model will still work).' % err)


# =========================== 3. LOAD FUNCTION =================================
def load_trained_model(filename = 'pricing_model.p'):
    """
    Include code that works in tandem with the PricingModel.save_model() method. 

    This function cannot take any parameters and must return a PricingModel object
    that is trained. 

    By default, this uses pickle, and is compatible with the default implementation
    of PricingGame.save_model. Change this only if your model does not support
    pickling (can happen with some libraries).
    """
    with open(filename, 'rb') as model:
        pricingmodel = pickle.load(model)
    return pricingmodel



# ========================= OPTIONAL CALIBRATION ===============================
def fit_and_calibrate_classifier(classifier, X, y):
    """
    Note:  This functions performs probability calibration
    This is an optional tool for you to use, it calibrates the probabilities from 
    your model if need be. 

    For more information see:
    https://scikit-learn.org/stable/modules/calibration.html 
    """
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier



# ==============================================================================

def check_consistency(trained_model, filename):
    """Returns True if the source file is consistent with the trained model."""
    # First, check that the model supports consistency checking (has _source_hash).
    if not hasattr(trained_model, '_source_hash'):
        return True  # No check was done (so we assume it's all fine).
    trained_source_hash = trained_model._source_hash
    with open(filename, 'r') as ff:
        code = ff.read()
    m = hashlib.sha256()
    m.update(code.encode())
    true_source_hash = m.hexdigest()
    return trained_source_hash == true_source_hash




# ============================ MAIN FUNCTION ================================
# Please do not write any executing code outside of the  __main__ safeguard.
# By default, this code trains your model (using training_data.csv) and saves
# it to pricing_model.p, then checks the consistency of the saved model and
# the pickle file.


if __name__ == '__main__':

    # Load the training data
    training_df = pd.read_csv('training_data.csv')
    y_claims_amount = training_df['claim_amount']
    y_made_claim = training_df['made_claim']
    X_train = training_df.drop(columns=['made_claim', 'claim_amount'])

    # Instantiate the pricing model and fit it
    my_pricing_model = PricingModel()
    my_pricing_model.fit(X_train, y_made_claim, y_claims_amount)

    # Save and load the pricing model
    my_pricing_model.save_model()
    loaded_model = load_trained_model()

    # Generate prices from the loaded model and the instantiated model
    predictions1 = my_pricing_model.predict_premium(X_train)
    predictions2 = loaded_model.predict_premium(X_train)

    # ensure that the prices are the same
    assert np.array_equal(predictions1, predictions2)