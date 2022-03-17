import numpy as np
from numpy import random
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin
from abc import ABCMeta
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import Lasso, Ridge, RidgeClassifier
from scipy.spatial.distance import cdist
from scipy.linalg import orth
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight
from sklearn.utils import column_or_1d
import torch
import joblib


class SemiRidge:
    def __init__(self, reg_alpha=0.0, reg_laplacian=0.0, k_neighbors=3, sigma=1.0):
        self.reg_alpha = reg_alpha
        self.reg_laplacian = reg_laplacian
        self.k_neighbors = k_neighbors
        self.sigma = sigma

    @staticmethod
    def _generate_laplacian_matrix(X, k_neighbors, sigma=1.0):
        assert sigma > 0.0
        S = cdist(X, X, metric='euclidean')
        N = len(S)
        if k_neighbors > 0:
            W = np.zeros((N, N))
            distance = torch.from_numpy(S)
            _, idx_near = torch.topk(-distance, dim=-1, largest=True, k=k_neighbors + 1)
            idx_near = idx_near.numpy()
            idx_near = idx_near[:, 1:]
            for i in range(N):
                for j in idx_near[i]:
                    W[i][j] = np.exp(-S[i][j] / (2 * sigma ** 2))
                    W[j][i] = W[i][j]
            D = np.sum(W, axis=1)
            L = np.diag(D) - W
        else:
            L = np.zeros((N, N))
        return L

    def fit(self, X, y, sample_weight=None):
        X = X.copy()
        y = y.copy()
        y = np.asarray(y, dtype=X.dtype)
        num_labeled = len(y)
        X_offset = np.average(X, axis=0, weights=sample_weight)
        X -= X_offset
        y_offset = np.average(y, axis=0, weights=sample_weight[: num_labeled])
        y -= y_offset
        feature_dim = X.shape[1]
        X_labeled = X[: num_labeled]
        number_balance = sample_weight[: num_labeled].sum() / sample_weight.sum()
        L = self._generate_laplacian_matrix(X, min(self.k_neighbors, len(X) - 1), self.sigma)
        labeled_weight = np.diag(sample_weight[: num_labeled])
        sample_weight = np.diag(sample_weight)
        self.coef_ = np.linalg.inv(
            X_labeled.T @ labeled_weight @ X_labeled + self.reg_alpha * np.eye(
                feature_dim) + self.reg_laplacian * number_balance * X.T @ L @ sample_weight @ X) @ X_labeled.T @ labeled_weight @ y
        self.intercept_ = y_offset - np.dot(X_offset, self.coef_)

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


class NodeGenerator:
    def __init__(self, active_function='relu', n_nodes_H=10, n_nodes_Z=10, n_groups_Z=10, reg_lambda=0.0):
        self.n_nodes_H = n_nodes_H
        self.n_nodes_Z = n_nodes_Z
        self.n_groups_Z = n_groups_Z
        self.reg_lambda = reg_lambda
        self.active_function = active_function

    @staticmethod
    def _sigmoid(data):
        return 1.0 / (1 + np.exp(-data))

    @staticmethod
    def _linear(data):
        return data

    @staticmethod
    def _tanh(data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    @staticmethod
    def _relu(data):
        return np.maximum(data, 0)

    @staticmethod
    def _generator(fea_dim, node_size):
        W = 2 * random.random(size=(fea_dim, node_size)) - 1
        b = 2 * random.random(size=(1, node_size)) - 1
        return W, b

    def _generate_h(self, data, sample_weight=None):

        self.nonlinear_ = {
            'linear': self._linear,
            'sigmoid': self._sigmoid,
            'tanh': self._tanh,
            'relu': self._relu
        }[self.active_function]
        self.scaler_H_ = StandardScaler()
        fea_dim = data.shape[1]
        W_H, self.b_H_ = self._generator(fea_dim, self.n_nodes_H)
        if fea_dim >= self.n_nodes_H:
            self.W_H_ = orth(W_H)
        else:
            self.W_H_ = orth(W_H.T).T
        data_ = np.dot(data, self.W_H_) + self.b_H_
        self.scaler_H_.fit(data_, sample_weight=sample_weight)
        return self.nonlinear_(self.scaler_H_.transform(data_))

    def _generate_z(self, X, sample_weight=None, view=None):
        fea_dim = X.shape[1]
        if view is not None:
            try:
                self.W_Z_[view] = []
                self.scaler_Z_[view] = []
            except:
                self.W_Z_ = {}
                self.scaler_Z_ = {}
                self.W_Z_[view] = []
                self.scaler_Z_[view] = []
            Z = None
            for i in range(self.n_groups_Z):
                W_Z, b_Z = self._generator(fea_dim, self.n_nodes_Z)
                data_ = np.dot(X, W_Z) + b_Z
                model = Lasso(alpha=self.reg_lambda, max_iter=1000)
                if sample_weight is not None:
                    model.fit(data_, X, sample_weight=sample_weight.copy())
                else:
                    model.fit(data_, X)
                if fea_dim > 1:
                    self.W_Z_[view].append(model.coef_)
                else:
                    self.W_Z_[view].append(model.coef_.reshape(1, -1))
                self.scaler_Z_[view].append(StandardScaler())
                Z_ = np.dot(X, self.W_Z_[view][i])
                if sample_weight is not None:
                    self.scaler_Z_[view][i].fit(Z_, sample_weight=sample_weight.copy())
                else:
                    self.scaler_Z_[view][i].fit(Z_)
                Z_ = self.scaler_Z_[view][i].transform(Z_)
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        else:
            self.W_Z_ = []
            self.scaler_Z_ = []
            Z = None
            for i in range(self.n_groups_Z):
                W_Z, b_Z = self._generator(fea_dim, self.n_nodes_Z)
                data_ = np.dot(X, W_Z) + b_Z
                model = Lasso(alpha=self.reg_lambda, max_iter=1000)
                if sample_weight is not None:
                    model.fit(data_, X, sample_weight=sample_weight.copy())
                else:
                    model.fit(data_, X)
                if fea_dim > 1:
                    self.W_Z_.append(model.coef_)
                else:
                    self.W_Z_.append(model.coef_.reshape(1, -1))
                self.scaler_Z_.append(StandardScaler())
                Z_ = np.dot(X, self.W_Z_[i])
                if sample_weight is not None:
                    self.scaler_Z_[i].fit(Z_, sample_weight=sample_weight.copy())
                else:
                    self.scaler_Z_[i].fit(Z_)
                Z_ = self.scaler_Z_[i].transform(Z_)
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        return Z

    def _transform_h(self, X=None):
        return self.nonlinear_(self.scaler_H_.transform(np.dot(X, self.W_H_) + self.b_H_))

    def _transform_z(self, X=None, view=None):
        if view is not None:
            Z = None
            for i in range(self.n_groups_Z):
                Z_ = self.scaler_Z_[view][i].transform(np.dot(X, self.W_Z_[view][i]))
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        else:
            Z = None
            for i in range(self.n_groups_Z):
                Z_ = self.scaler_Z_[i].transform(np.dot(X, self.W_Z_[i]))
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        return Z


class MVBLS(NodeGenerator, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, n_nodes_H=1000, active_function='relu', n_nodes_Z=10, n_groups_Z=10, reg_alpha=0.1, reg_lambda=0.1, view_list=None,
                 random_state=0):

        NodeGenerator.__init__(self, active_function=active_function, n_nodes_H=n_nodes_H, n_nodes_Z=n_nodes_Z, n_groups_Z=n_groups_Z,
                               reg_lambda=reg_lambda)
        self.reg_alpha = reg_alpha
        self.view_list = view_list
        self.random_state = random_state

    def save_model(self, file):
        """
        Parameters
        ----------
        file: str
            Controls the filename.
        """
        check_is_fitted(self, ['estimator_'])
        joblib.dump(self, filename=file)

    def _decision_function(self, X):
        check_is_fitted(self, ['estimator_'])
        if self.view_list is not None:
            assert isinstance(X, dict)
            Z = None
            for view in self.view_list:
                X_ = check_array(X[view])
                Z_ = self._transform_z(X_, view)
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        else:
            X = check_array(X)
            Z = self._transform_z(X, view=None)
        H = self._transform_h(Z)
        return self.estimator_.predict(np.c_[Z, H])

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model.

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features)
                Training data

            y : ndarray of shape (n_samples,) or (n_samples, n_targets)
                Target values

            Returns
            -------
            self : returns an instance of self.
        """
        np.random.seed(self.random_state)
        self.estimator_ = Ridge(alpha=self.reg_alpha)
        # generate Z

        if self.view_list is not None:
            assert isinstance(X, dict)
            Z = None
            for view in self.view_list:
                X_, y = check_X_y(X[view], y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
                sample_weight = _check_sample_weight(sample_weight, X_, dtype=X_.dtype)
                Z_ = self._generate_z(X_, sample_weight=sample_weight, view=view)
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        else:
            X, y = check_X_y(X, y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            Z = self._generate_z(X, sample_weight=sample_weight, view=None)
        # generate H
        H = self._generate_h(Z, sample_weight)
        self.estimator_.fit(np.c_[Z, H], y, sample_weight=sample_weight)
        return self


class MVBLSRegressor(MultiOutputMixin, RegressorMixin, MVBLS):
    """

        Parameters
        ----------
        n_nodes_H: int, default=1000
                    Controls the number of enhancement nodes.
        active_function: {str, ('relu', 'tanh', 'sigmod' or 'linear')}, default='relu'
                        Controls the active function of enhancement nodes.
        n_nodes_Z: int, default=10
                    Controls the number of feature nodes in each group.
        n_groups_Z: int, default=10
                    Controls the number of feature node groups.
        reg_alpha: float, default=0.1
                    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
        reg_lambda: float, default=0.1
                    Constant that multiplies the L1 term. Defaults to 1.0. ``alpha = 0`` is equivalent to an ordinary least square.
        view_list: list, default=None
                    List of view names.
        random_state: int, default=0
                        Controls the randomness of the estimator.
    """

    def predict(self, X):
        """
                Predict using the linear model.

                Parameters
                ----------
                X : array_like or sparse matrix, shape (n_samples, n_features)
                    Samples.

                Returns
                -------
                C : array, shape (n_samples,)
                    Returns predicted values.
                """
        return self._decision_function(X)


class MVBLSClassifier(ClassifierMixin, MVBLS):
    """
        MVBLS classifier. Construct a broad learning systerm model.
        Parameters
        ----------
        n_nodes_H: int, default=1000
                    Controls the number of enhancement nodes.
        active_function: {str, ('relu', 'tanh', 'sigmod' or 'linear')}, default='relu'
                        Controls the active function of enhancement nodes.
        n_nodes_Z: int, default=10
                    Controls the number of feature nodes in each group.
        n_groups_Z: int, default=10
                    Controls the number of feature node groups.
        reg_alpha: float, default=0.1
                    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
        reg_lambda: float, default=0.1
                    Constant that multiplies the L1 term. Defaults to 1.0. ``reg_lambda = 0`` is equivalent to an ordinary least square.
        view_list: list, default=None
                    List of view names.
        random_state: int, default=0
                        Controls the randomness of the estimator.
    """

    def predict(self, X):
        """
            Predict class labels for samples in X.

            Parameters
            ----------
            X : array_like or sparse matrix, shape (n_samples, n_features)
                Samples.

            Returns
            -------
            C : array, shape [n_samples]
                Predicted class label per sample.
        """
        scores = self._decision_function(X)
        scores = scores.ravel() if scores.shape[1] == 1 else scores
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def fit(self, X, y, sample_weight=None):
        """
            Build a broad learning systerm model from the training set (X, y).

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features)
                Training data.

            y : ndarray of shape (n_samples,)
                Target values.

            sample_weight : float or ndarray of shape (n_samples,), default=None
                Individual weights for each sample. If given a float, every sample
                will have the same weight.

            Returns
            -------
            self : object
                Instance of the estimator.
        """
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            _ = column_or_1d(y, warn=True)
        else:
            # we don't (yet) support multi-label classification in Ridge
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))
        super().fit(X, Y, sample_weight)
        return self

    @property
    def classes_(self):
        """
        Classes labels
        """
        return self._label_binarizer.classes_


class SemiMVBLS(MVBLS):
    def __init__(self, reg_laplacian=0.0, k_neighbors=3, sigma=1.0, unlabeled_data=None, **kwargs):
        MVBLS.__init__(self, **kwargs)
        self.reg_laplacian = reg_laplacian
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.unlabeled_data = unlabeled_data

    def save_model(self, file):
        """
        Parameters
        ----------
        file: str
            Controls the filename.
        """
        check_is_fitted(self, ['estimator_'])
        self.unlabeled_data = None
        joblib.dump(self, filename=file)

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model.

            Parameters
            ----------
            X : {ndarray, sparse matrix} of shape (n_samples, n_features)
                Training data

            y : ndarray of shape (n_samples,) or (n_samples, n_targets)
                Target values

            Returns
            -------
            self : returns an instance of self.
        """
        if self.unlabeled_data is not None:
            assert isinstance(self.unlabeled_data, dict)
        np.random.seed(self.random_state)
        self.estimator_ = SemiRidge(reg_alpha=self.reg_alpha, reg_laplacian=self.reg_laplacian, k_neighbors=self.k_neighbors, sigma=self.sigma)
        # generate Z
        if self.view_list is not None:
            assert isinstance(X, dict)
            Z = None
            for view in self.view_list:
                X_, y = check_X_y(X[view], y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
                if Z is None:
                    sample_weight = _check_sample_weight(sample_weight, X_, dtype=X_.dtype)
                if self.unlabeled_data is None:
                    Z_ = self._generate_z(X_, sample_weight=sample_weight, view=view)
                else:
                    uX_ = check_array(self.unlabeled_data[view], dtype=[np.float64, np.float32])
                    if Z is None:
                        uW_ = np.ones(len(uX_))
                        sample_weight = np.r_[sample_weight, uW_]
                    Z_ = self._generate_z(np.r_[X_, uX_], sample_weight=sample_weight, view=view)
                if Z is None:
                    Z = Z_
                else:
                    Z = np.c_[Z, Z_]
        else:
            X, y = check_X_y(X, y, dtype=[np.float64, np.float32], multi_output=True, y_numeric=True)
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            if self.unlabeled_data is None:
                Z = self._generate_z(X, sample_weight=sample_weight, view=None)
            else:
                uX_ = check_array(self.unlabeled_data, dtype=[np.float64, np.float32])
                uW_ = np.ones(len(uX_))
                sample_weight = np.r_[sample_weight, uW_]
                Z = self._generate_z(np.r_[X, uX_], sample_weight=sample_weight, view=None)
        # generate H
        H = self._generate_h(Z, sample_weight=sample_weight)
        self.estimator_.fit(np.c_[Z, H], y, sample_weight)
        return self


class SemiMVBLSClassifier(ClassifierMixin, SemiMVBLS):
    """

            Parameters
            ----------
            reg_laplacian: float, default=1.0
                        Constant that multiplies the laplacian term. Defaults to 1.0. ``reg_laplacian = 0`` is equivalent to an Ridge regression.
            k_neighbors: int, default=5
                        Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method.
            sigma: float, default=1.0
                Kernel coefficient for rbf.
            unlabeled_data: {ndarray, sparse matrix} of shape (n_samples, n_features) or {dict}
                    Unlabeled training data.
            n_nodes_H: int, default=10
                        Controls the number of enhancement nodes.
            active_function: {str, ('relu', 'tanh', 'sigmod' or 'linear')}, default='relu'
                            Controls the active function of enhancement nodes.
            n_nodes_Z: int, default=10
                        Controls the number of feature nodes in each group.
            n_groups_Z: int, default=10
                        Controls the number of feature node groups.
            reg_alpha: float, default=1.0
                        Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
            reg_lambda: float, default=1.0
                        Constant that multiplies the L1 term. Defaults to 1.0. ``reg_lambda = 0`` is equivalent to an ordinary least square.
            view_list: list, default=None
                        List of view names.
            random_state: int, default=None
                            Controls the randomness of the estimator.
        """

    def predict(self, X):
        """
                Predict class labels for samples in X.

                Parameters
                ----------
                X : array_like or sparse matrix, shape (n_samples, n_features)
                    Samples.

                Returns
                -------
                C : array, shape [n_samples]
                    Predicted class label per sample.
                """
        scores = self._decision_function(X)
        scores = scores.ravel() if scores.shape[1] == 1 else scores
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge classifier model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target values.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

            .. versionadded:: 0.17
               *sample_weight* support to Classifier.

        Returns
        -------
        self : object
            Instance of the estimator.
        """
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            _ = column_or_1d(y, warn=True)
        else:
            raise ValueError("%s doesn't support multi-label classification" % (self.__class__.__name__))
        super().fit(X, Y, sample_weight=sample_weight)
        return self

    @property
    def classes_(self):
        """
        Classes labels
        """
        return self._label_binarizer.classes_


class SemiMVBLSRegressor(MultiOutputMixin, RegressorMixin, SemiMVBLS):
    """

            Parameters
            ----------
            reg_laplacian: float, default=1.0
                        Constant that multiplies the laplacian term. Defaults to 1.0. ``reg_laplacian = 0`` is equivalent to an Ridge regression.
            k_neighbors: int, default=5
                        Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method.
            sigma: float, default=1.0
                Kernel coefficient for rbf.
            unlabeled_data: {ndarray, sparse matrix} of shape (n_samples, n_features) or {dict}
                    Unlabeled training data.
            n_nodes_H: int, default=10
                        Controls the number of enhancement nodes.
            active_function: {str, ('relu', 'tanh', 'sigmod' or 'linear')}, default='relu'
                            Controls the active function of enhancement nodes.
            n_nodes_Z: int, default=10
                        Controls the number of feature nodes in each group.
            n_groups_Z: int, default=10
                        Controls the number of feature node groups.
            reg_alpha: float, default=1.0
                        Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization.
            reg_lambda: float, default=1.0
                        Constant that multiplies the L1 term. Defaults to 1.0. ``alpha = 0`` is equivalent to an ordinary least square.
            view_list: list, default=None
                        List of view names.
            random_state: int, default=None
                            Controls the randomness of the estimator.
        """

    def predict(self, X):
        """
                Predict using the linear model.

                Parameters
                ----------
                X : array_like or sparse matrix, shape (n_samples, n_features)
                    Samples.

                Returns
                -------
                C : array, shape (n_samples,)
                    Returns predicted values.
                """
        return self._decision_function(X)


if __name__ == "__main__":
    # Check if meet Sklearn's manner
    # ------------------------------------
    check_estimator(MVBLSRegressor())
    check_estimator(MVBLSClassifier())
    check_estimator(SemiMVBLSRegressor())
    check_estimator(SemiMVBLSClassifier())
