Welcome to the documentation for Frosted Tracks
===============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   changelog

This library uses the unsupervised machine learning algorithm TICC (Toeplitz Inverse Covariance Clustering) [#]_, implemented via the ``fast-ticc`` python module, to generate behavioral labels for trajectory points.

``frosted-tracks`` works in concert with the ``tracktable`` python module; ``tracktable`` is used to process raw location data into trajectory objects, and ``frosted-tracks`` creates behavioral labels for those trajectories.

Documentation for ``fast-ticc`` can be found `here <https://fast-ticc.readthedocs.io/>`_.
Documentation for ``tracktable`` can be found `here <https://tracktable.readthedocs.io/>`_.

.. [#] Hallac, David, et al. "Toeplitz inverse covariance-based clustering of multivariate time series data." *Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining.* 2017.