# Python Examples
This page shows some Python demos.

## MVBLS.MVBLSClassifier


```python
import numpy as np
from MVBLS import MVBLSClassifier
from sklearn.linear_model import RidgeClassifier

data = np.load('clf_data.npz')
data_train = {}
data_train['view1'] = data['X_train'][:, :128]
data_train['view2'] = data['X_train'][:, 128:]
data_test = {}
data_test['view1'] = data['X_test'][:, :128]
data_test['view2'] = data['X_test'][:, 128:]
model = RidgeClassifier(alpha=0.01)
model.fit(data['X_train'], data['y_train'])
print('ACC of Baseline: %.3f' % model.score(data['X_test'], data['y_test']))

model = MVBLSClassifier(view_list=data_train.keys(), reg_lambda=0.01, n_nodes_Z=50, n_nodes_H=6000, n_groups_Z=10, reg_alpha=0.01,
                        random_state=1)
model.fit(data_train, data['y_train'])
print('ACC of MVBLSClassifier: %.3f' % model.score(data_test, data['y_test']))
```

    ACC of Baseline: 0.793
    ACC of MVBLSClassifier: 0.817


## MVBLS.MVBLSRegressor


```python
import numpy as np
from MVBLS import MVBLSRegressor
from sklearn.linear_model import Ridge

data = np.load('reg_data.npz')
data_train = {}
data_train['view1'] = data['X_train'][:, :128]
data_train['view2'] = data['X_train'][:, 128:]
data_test = {}
data_test['view1'] = data['X_test'][:, :128]
data_test['view2'] = data['X_test'][:, 128:]
model = Ridge(alpha=0.01)
model.fit(data['X_train'], data['y_train'])
print('R^2 of Baseline: %.3f' % model.score(data['X_test'], data['y_test']))

model = MVBLSRegressor(view_list=data_train.keys(), reg_lambda=0.01, n_nodes_Z=50, n_nodes_H=6000, n_groups_Z=10, reg_alpha=0.01)
model.fit(data_train, data['y_train'])
print('R^2 of  MVBLSRegressor: %.3f' % model.score(data_test, data['y_test']))
```

    R^2 of Baseline: 0.798
    R^2 of  MVBLSRegressor: 0.938


## MVBLS.SemiMVBLSClassifier


```python
import numpy as np
from MVBLS import SemiMVBLSClassifier
from sklearn.linear_model import RidgeClassifier

data = np.load('clf_data.npz')
few_short_index = data['few_short_index']
data_train = {}
unlabeled_data_train = {}
data_train['view1'] = data['X_train'][:, :128][few_short_index]
data_train['view2'] = data['X_train'][:, 128:][few_short_index]
unlabeled_data_train['view1'] = data['X_train'][:, :128][~few_short_index]
unlabeled_data_train['view2'] = data['X_train'][:, 128:][~few_short_index]
data_test = {}
data_test['view1'] = data['X_test'][:, :128]
data_test['view2'] = data['X_test'][:, 128:]

model = RidgeClassifier(alpha=0.01)
model.fit(data['X_train'][few_short_index], data['y_train'][few_short_index])
print('ACC of Baseline: %.3f' % model.score(data['X_test'], data['y_test']))

model = SemiMVBLSClassifier(view_list=data_train.keys(), unlabeled_data=unlabeled_data_train, reg_lambda=0.01, n_nodes_Z=50, n_nodes_H=6000,
                            n_groups_Z=10, reg_alpha=0.01, reg_laplacian=0.01, k_neighbors=10, sigma=1.0)
model.fit(data_train, data['y_train'][few_short_index])
print('ACC of SemiMVBLSClassifier: %.3f' % model.score(data_test, data['y_test']))
```

    ACC of Baseline: 0.668
    ACC of SemiMVBLSClassifier: 0.681


## MVBLS.SemiMVBLSRegressor


```python
import numpy as np
from MVBLS import SemiMVBLSRegressor
from sklearn.linear_model import Ridge

data = np.load('reg_data.npz')
few_short_index = data['few_short_index']
data_train = {}
unlabeled_data_train = {}
data_train['view1'] = data['X_train'][:, :128][few_short_index]
data_train['view2'] = data['X_train'][:, 128:][few_short_index]
unlabeled_data_train['view1'] = data['X_train'][:, :128][~few_short_index]
unlabeled_data_train['view2'] = data['X_train'][:, 128:][~few_short_index]
data_test = {}
data_test['view1'] = data['X_test'][:, :128]
data_test['view2'] = data['X_test'][:, 128:]

model = Ridge(alpha=0.01)
model.fit(data['X_train'][few_short_index], data['y_train'][few_short_index])
print('R^2 of Baseline: %.3f' % model.score(data['X_test'], data['y_test']))

model = SemiMVBLSRegressor(view_list=data_train.keys(), unlabeled_data=unlabeled_data_train, reg_lambda=0.01, n_nodes_Z=50, n_nodes_H=6000,
                           n_groups_Z=10, reg_alpha=0.01, reg_laplacian=0.01, k_neighbors=10, sigma=1.0)
model.fit(data_train, data['y_train'][few_short_index])
print('R^2 of SemiMVBLSRegressor: %.3f' % model.score(data_test, data['y_test'])) 
```

    R^2 of Baseline: 0.456
    R^2 of SemiMVBLSRegressor: 0.828

