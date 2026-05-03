Ready.  Set.  Frost!
====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   changelog


Frosted Tracks is a library for clustering the trajectories of moving objects according to their behavior. 

How Frosted Tracks Works
------------------------

We proceed in two steps.  First, given a set of evenly sampled input trajectories, Frosted Tracks uses the unsupervised machine learning algorithm TICC (Toeplitz Inverse Covariance Clustering) [#]_ to assign a label to each point of each trajectory that describes its behavior.  This gives us a string of labels for each trajectory in the input.

Second, we use DBSCAN [#]_ to create clusters of the strings of labels.  

We use the `Fast TICC`_ package to compute the labels and `Metric DBSCAN`_ for the clustering.  

We also use `Tracktable_` to wrangle trajectories, including reading position data from CSV and resampling trajectories so that their points ar eevenly spaced in time.


Documentation Under Construction
--------------------------------

Stay tuned for frequent updates to these pages.  


.. [#] Hallac, David, et al. "Toeplitz inverse covariance-based clustering of multivariate time series data." *Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining.* 2017.

.. [#] Ester, Martin, et al. "A density-based algorithm for discovering clusters in large spatial databases with noise."  *KDD '96: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining.'* 1996.

.. _Fast TICC: https://fast-ticc.readthedocs.io

.. _Tracktable: https://tracktable.readthedocs.io

.. _Metric DBSCAN: https://github.com/sandialabs/metric_dbscan
