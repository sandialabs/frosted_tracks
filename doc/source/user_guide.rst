How to Use this Library
=======================

The ``frosted-tracks`` python module uses the unsupervised machine learning algorithm TICC (Toeplitz Inverse Covariance Clustering) [#]_, implemented via the ``fast-ticc`` python module, to generate behavioral labels for trajectory points.

``frosted-tracks`` works in concert with the ``tracktable`` python module; ``tracktable`` is used to process raw location data into trajectory objects, and ``frosted-tracks`` creates behavioral labels for those trajectories.

.. [#] Hallac, David, et al. "Toeplitz inverse covariance-based clustering of multivariate time series data." *Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining.* 2017.


Creating Trajectories
---------------------

``frosted-tracks`` operates on trajectory objects from the ``tracktable`` python module.  Please refer to the `tracktable documentation <https://tracktable.readthedocs.io/>`_ for instructions on how to create trajectories; a tutorial for generating trajectories from csv data containing (at minimum) latitude, longitude, time and a unique object idenfitifier can be found `here <https://tracktable.readthedocs.io/en/latest/examples/Tutorial_02.html>`_.


Generating Behavioral Labels for Trajectories
---------------------------------------------

You can create behavioral labels for your list of ``tracktable`` trajectory objects as follows:

1. Resample your trajectories to create equal spacing in time. That is, the timestep between sequential trajectory points should be the same for all trajectories. This equal temporal spacing is required to generate meaningful results from ``fast-ticc``.

   The ``frosted-tracks`` module provides various resampling methods in ``frosted_tracks.trajectory.resample``, or you may resample on your own. And example of using ``frosted-tracks`` for linear resampling is given below.

   .. code-block:: python

      from frosted_tracks.trajectory.resample.linear import resample_by_time_linear
      from datetime import timedelta

      timestep = timedelta(seconds=10)

      resampled_trajectory = resample_by_time_linear(trajectory,
                                                     timestep)

1. Choose the number of behaviors you believe to be present in your trajectories, which is analogous to selecting the number of clusters for TICC.

1. Choose a window size.  This should be the smallest number of sequential trajectory points you believe will be enough to capture a single trajectory behavior you wish to identify.

1. Identify two or more features (e.g. heading, speed, etc.) that you believe are relevant to the behaviors you wish to identify.  TICC will look at correlation between these features over the given time window to determine the behavioral labels.

   For each feature, create a python function that inputs a trajectory and outputs the timeseries for that feature.  ``frosted-tracks`` will need a list of these feature functions as part of its inputs.

1. Run ``frosted-tracks``:

   .. code-block:: python

       from frosted_tracks.segmentation.drivers import ticc_label_trajectories

       ticc_results = ticc_label_trajectories(trajectories,
                                              my_features,
                                              num_clusters=num_clusters,
                                              window_size=window_size,
                                              num_processors=1)

   Increasing ``num_processors`` from 1 will leverage parallel computing (assuming multiple processors are available) to decrease runtime.

1. The resulting behavioral labels can be found in ``ticc_results.point_labels``, which is a list of lists.  The value of ``ticc_results.point_labels[i]`` is a list of behavior labels (integers) for the ``i``-th trajectory.  As the behavioral labels are assigned to the center point of each window, the first and last ``window_size/2`` trajectory points will recieve a -1 behavioral label.  Please see the `documentation for fast-ticc <https://fast-ticc.readthedocs.io/en/latest/quirks.html#invalid-cluster-labels-at-beginning-and-end>`_ for more clarity on why this happens.


..
    Visualizing Trajectory Labels
    -----------------------------
    
    The ``frosted-tracks`` module extends ``tracktable``'s trajectory visualization capabilities via ``folium`` to visualize trajectories with their behavioral labels indicated by color.
    
    .. code-block:: python
    
