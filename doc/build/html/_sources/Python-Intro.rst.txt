Python-package Introduction
===========================

This document gives a basic walk-through of MVBLS Python-package.

**List of other helpful links**



-  `Python API <MVBLS.html>`__

-  `Parameters Tuning <Parameters-Tuning.html>`__

-  `Python Examples <Python-Examples.html>`__

Install
-------

The preferred way to install MVBLS is via pip:

::

    pip install MVBLS


To verify your installation, try to ``import MVBLS`` in Python:

::

    import MVBLS

Data Interface
--------------

The MVBLS Python module can load data from:

-  NumPy 2D array(s) for single-view
    .. code:: python

        import numpy as np
        data = np.random.rand(500, 10)
        label = np.random.randint(2, size=500)
    **Note**: if input only one view data to MVBLS, it works like the original BLS [2].

-  Dictionary for multi-view
    .. code:: python

        import numpy as np
        data={}
        data['view1'] = np.random.rand(500, 10)
        data['view2'] = np.random.rand(500, 10)
        data['view3'] = np.random.rand(500, 10)
        label = np.random.randint(2, size=500)



Setting Parameters
------------------

MVBLS can use a dictionary to set parameters.
For instance:

-  For single-view:

   .. code:: python

       param = {'view_list': None, 'reg_lambda': 0.01, 'n_nodes_Z':50, 'n_nodes_H': 6000, 'n_groups_Z': 10, 'reg_alpha': 0.01,
                        'random_state': 0}

-  For multi-view:

   .. code:: python

       param = {'view_list': data.keys(), 'reg_lambda': 0.01, 'n_nodes_Z':50, 'n_nodes_H': 6000, 'n_groups_Z': 10, 'reg_alpha': 0.01,
                        'random_state': 0}

Training
--------

Training a model requires a parameter list and data set:

.. code:: python


    estimator = MVBLS.MVBLSClassifier(**param).fit(data, label)

After training, the model can be saved:

.. code:: python

    estimator.save_model('model.joblib')

A saved model can be loaded:

.. code:: python

    import joblib
    estimator = joblib.load('model.joblib')


Predicting
----------

A model that has been trained or loaded can perform predictions on datasets:

-  For single-view:
    .. code:: python

        # 7 entities, each contains 10 features
        data = np.random.rand(7, 10)
        ypred = estimator.predict(data)
-  For multi-view:
    .. code:: python

        # 7 entities, each contains 10 features
        data={}
        data['view1'] = np.random.rand(7, 10)
        data['view2'] = np.random.rand(7, 10)
        data['view3'] = np.random.rand(7, 10)
        ypred = estimator.predict(data)