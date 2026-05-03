How to Use this Library
=======================


.. CAUTION:: 
   This page is under construction.  Do not rely on its contents until this message disappears.

   
To start with, you need a set of evenly- and sufficiently-sampled trajectories.  See "Creating Trajectories" and "Resampling Trajectories".  After that, you can create behavior labels for those trajectories and then stop, or create labels and then cluster trajectories based on those labels.  See "Generating Behavioral Labels for Trajectories" and "Clustering Labeled Trajectories" for details.


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

2. Choose the number of behaviors you believe to be present in your trajectories, which is analogous to selecting the number of clusters for TICC.

3. Choose a window size.  This should be the smallest number of sequential trajectory points you believe will be enough to capture a single trajectory behavior you wish to identify.

4. Identify two or more features (e.g. heading, speed, etc.) that you believe are relevant to the behaviors you wish to identify.  TICC will look at correlation between these features over the given time window to determine the behavioral labels.

   For each feature, create a python function that inputs a trajectory and outputs the timeseries for that feature.  ``frosted-tracks`` will need a list of these feature functions as part of its inputs.

5. Run ``frosted-tracks``:

   .. code-block:: python

       from frosted_tracks.segmentation.drivers import ticc_label_trajectories

       ticc_results = ticc_label_trajectories(trajectories,
                                              my_features,
                                              num_clusters=num_clusters,
                                              window_size=window_size,
                                              num_processors=1)

.. NOTE::
      Increasing ``num_processors`` from 1 *is usually not what you want*.  Most of the CPU time in Frosted Tracks happens in low-level linear algebra.  This is handled by `BLAS`_ libraries that usually do their own parallelism.  If your installation does this, setting ``num_processors`` greater than 1 will actually slow the algorithm down.



1. The resulting behavioral labels can be found in ``ticc_results.point_labels``, which is a list of lists.  The value of ``ticc_results.point_labels[i]`` is a list of behavior labels (integers) for the ``i``-th trajectory.  As the behavioral labels are assigned to the center point of each window, the first and last ``window_size/2`` trajectory points will recieve a -1 behavioral label.  Please see the `documentation for fast-ticc <https://fast-ticc.readthedocs.io/en/latest/quirks.html#invalid-cluster-labels-at-beginning-and-end>`_ for more clarity on why this happens.


Clustering Trajectories
-----------------------

This section coming soon.  The entry point is ``frosted_tracks.cluster_trajectories()``.


..
    Visualizing Trajectory Labels
    -----------------------------
    
    The ``frosted-tracks`` module extends ``tracktable``'s trajectory visualization capabilities via ``folium`` to visualize trajectories with their behavioral labels indicated by color.
    
    .. code-block:: python
    

.. _BLAS: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
