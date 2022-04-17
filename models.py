#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class ModelRunner:
    """
    This class handles all ML model fit and prediction processes
    """
    def __init__(self,model_type,X,y,problem):
        self.model_type = model_type
        self.X = X
        self.y = y
        self.problem = problem


    def runner(self):
        """
        Runner method
        returns score of model prediction
        """
        #decide model
        model = self._decide_model()
        #get X,y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=123)
        #run model
        y_pred, model = self._run_model(model,X_train,y_train,X_test)
        #evaluation metrics
        score = self._evaluate(y_test,y_pred)
        return score

    def _decide_model(self):
        if self.model_type == "Linear Regression":
            model = LinearRegression()
        elif self.model_type == "XGBoost":
            model = XGBRegressor()
        elif self.model_type == "ElasticNetCV":
            model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)
        elif self.model_type == "Decision Tree":
            model = DecisionTreeRegressor()
        elif self.model_type == "SGDRegressor":
            model = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
        elif self.model_type == "Ridge":
            model = Ridge()
        elif self.model_type == "Lasso":
            model = Lasso()
        elif self.model_type == "RandomForestRegressor":
            model = RandomForestRegressor()
        elif self.model_type == "GradientBoostingRegressor":
            model = GradientBoostingRegressor()
        elif self.model_type == "Support Vector Machine":
            model = SVR()
        return model

    def _run_model(self,model,X_train,y_train,X_test):
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return y_pred, model


    def _evaluate(self,y_test,y_pred):
        """
        Root mean square for Regression
        Accuracy for Classfication
        """
        if self.problem == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = round(math.sqrt(mse),2)
            return rmse
        else:
            score = round(accuracy_score(y_test,y_pred),2)
            return score

