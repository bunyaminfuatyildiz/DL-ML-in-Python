from egeaML import DataIngestion, plots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
di = DataIngestion(df='boston.csv',col_to_drop=None, col_target='MEDV')
X = di.features()
y = di.target()
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=42)
reg = LinearRegression()
fit = reg.fit(X_train,y_train)
print('Regression R2 Score: {:.4f}'.format(reg.score (X_test,y_test)))
y_pred = reg.predict(X_test)
e = y_pred-y_test
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
cv_scores = cross_val_score(reg,X_train, y_train, cv=10)
print("Average 10-Fold CV Score: {}".format((np.mean(cv_scores)) ))
ridge = Ridge(normalize=True)
ridge.fit(X_train, y_train)
ridge.score(X_test,y_test)
score = format(ridge.score(X_test,y_test), '.4f')
print('Ridge Reg Score with Normalization: {}'.format(score))
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train,y_train)
score_pipe = format(pipe.score(X_test,y_test), '.4f')
print('Standardized Ridge Score:{}'.format(score_pipe))
lasso = Lasso(max_iter=10000,normalize=True)
coefs = list()
for alpha in alphas:
    lasso.set_params(alpha=alpha)
    lasso.fit(X_train,y_train)
    coefs.append(lasso.coef_)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim())
plt.xlabel('$\\alpha$ (alpha)')
plt.ylabel('Regression Coefficients')
plt.show()