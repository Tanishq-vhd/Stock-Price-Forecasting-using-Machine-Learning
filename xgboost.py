import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  
    'eval_metric': 'rmse', 
    'max_depth': 3, 
    'eta': 0.1  
}

model = xgb.train(params, dtrain, num_boost_round=100)


predictions = model.predict(dtest)

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'XGBoost RMSE: {rmse:.4f}')


plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Prices', color='blue')
plt.plot(predictions, label='Predicted Prices', color='red')
plt.legend()
plt.title('Stock Price Prediction using XGBoost with Technical Indicators')
plt.show()
