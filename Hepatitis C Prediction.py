import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load

#Loadind Data
hcv = pd.read_csv("hcv.csv")
#print(hcv)

train_set, test_set = train_test_split(hcv, test_size = 0.2, random_state = 42)
#print(f"Rows in train_set: {len(train_set)}\nRows in test_set: {len(test_set)}")

hcv = train_set.drop("Category", axis = 1)
hcv_labels = train_set["Category"].copy()

imputer = SimpleImputer(strategy="median")
imputer.fit(hcv)
X = imputer.transform(hcv)
hcv_tr = pd.DataFrame(X, columns = hcv.columns)
#hcv_tr.describe()

#To be able modify the code later without disturbing the dataset
#Here we are standardizing the values of every feature for better analysis
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    #    ...... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])
hcv_num_tr = my_pipeline.fit_transform(hcv_tr)
#hcv_num_tr.shape

#model = LogisticRegression(random_state = 42)
model = GaussianProcessClassifier(random_state = 42)
#model = KNeighborsClassifier(n_neighbors=2)
model.fit(hcv_num_tr, hcv_labels)

predictions = model.predict(hcv_num_tr)
mse = mean_squared_error(hcv_labels, predictions)
rmse = np.sqrt(mse)
#print(rmse)
#test_f_tr.shape

scores = cross_val_score(model, hcv_num_tr, hcv_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)
#rmse_scores

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
#print_scores(rmse_scores)

dump(model, 'HCV.joblib')

X_test = test_set.drop("Category", axis = 1)
Y_test = test_set["Category"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_rmse)
#Y_test

X_test_prepared[0]
#from joblib import dump, load
#import numpy as np
model = load('HCV.joblib')
features = np.array([[ 0.7953659 , -0.85580042, -2.40494829, -0.11086791, -0.49456308,
       -0.49724748, -0.45646026, -1.21434247, -0.86923583, -0.29970024,
       -0.23502719, -1.78698052]])
print(model.predict(features))
