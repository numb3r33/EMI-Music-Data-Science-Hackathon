from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb 


def fit(X, y):
	"""
	Creates a pipeline and fit on the dataset
	"""

	print('Train model')
	
	select = SelectKBest(f_regression, k=50)
	est = xgb.XGBRegressor()
	pipeline = Pipeline([('select', select), ('est', est)])

	pipeline.fit(X, y)

	return pipeline

