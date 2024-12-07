# Copyright 2024 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Front end for Frosted Tracks

Start here.  This file contains the function(s) you'll want to call to
label and cluster your trajectories.

"""

import copy
import math

import numba
import numpy as np
import pandas as pd
import metric_dbscan
import tqdm

import sklearn.metrics
import sklearn.metrics.pairwise

from sklearn.metrics import f1_score, accuracy_score

from frosted_tracks.frosted_tracks_types import FeatureFunction, Trajectory

from frosted_tracks import feature_functions, matrix_utilities, segmentation

from typing import List, Optional, Union

# Any default parameters that we can't store in the signature for
# cluster_trajectories (because they're mutable) go here.
DEFAULT_PARAMETERS = {
    'feature_functions': ['change_in_speed', 'signed_heading_change']
}

# We'll use this to store the distance between covariance/correlation
# matrices so we can use those as edit penalties in our distance function.

DISTANCE_BETWEEN_MATRICES=None

def cluster_trajectories(trajectories: List[Trajectory],
                         feature_functions: List[str]=None,
                         num_labels: Union[int, str]=5,
                         window_size: int=10,
                         minimum_cluster_size: int=10,
                         maximum_neighbor_distance: int=6
                         ):
    """Cluster trajectories according to behavioral segmentation

    Run TICC to assign a behavioral label to each point in each trajectory,
    then run DBSCAN to cluster those strings by weighted edit distance.

    See the Frosted Tracks report (link included once available) for details.

    Arguments:
        trajectories {list of Tracktable trajectories}: Input trajectories.
            Must be from Tracktable.

    Keyword Arguments:
        feature_functions {list of strings}: Names of feature functions
            to generate multivariate data series from trajectories for
            behavioral labeling. Defaults to ['change_in_speed',
            'signed_heading_change'].
            See frosted_tracks.feature_functions.register_feature_function
            for details.
        num_labels {int or str}: Number of different behavioral labels to
            use.  This should be a positive integer or "estimate".  If
            you use "estimate" then we will use an experimental method to
            try to make a reasonable guess based on the properties of
            the data.  Defaults to 5.
        window_size {int}: How many points to consider when assigning
            behavioral labels.  Defaults to 10.  This should be set to
            the smallest number of points that will reasonably display
            the behaviors in your trajectory data.
        minimum_cluster_size {int}: DBSCAN parameter.  Clusters are composed
            of groups of trajectories that all have at least this many
            neighbors in behavior-string space.  Trajectories with fewer than
            this many neighbors will usually be flagged as outliers.
            Defaults to 10.
        maximum_neighbor_distance {int}: How far apart two trajectories
            can be in behavior-string space (measured by weighted edit
            distance) and still be considered neighbors for clustering.

        Returns:
            Pandas DataFrame with the following columns:
                - "Object ID": Object ID for each trajectory
                - "Trajectory": Original trajectories
                - "Segment Labels": Point-by-point behavioral label for each
                  trajectory
                - "Condensed Segment Labels": Behavioral label strings after
                  long runs of the same behavior have been shortened
                - "Correlation Matrices": Feature-wise correlation matrices
                  identified by TICC (behavioral labeling step)
                - "dbscan_label": Cluster ID assigned by DBSCAN.  This
                  is the actual clustering.

        Also note that the segment labels will be saved in the
        "segment_label" property of each point in each trajectory.  This
        lets you use them to color trajectories during visualization.
    """
    # Store the trajectories in a Pandas DataFrame.  This will give us
    # a convenient place to keep the things we compute about them.

    object_ids = [t.object_id for t in trajectories]
    trajectories_df = pd.DataFrame({'Object ID': object_ids,
                                    'Trajectory': trajectories
                                })

    sensors = []
    for item in feature_functions:
        if isinstance(item, str):
            # If it's the name of a function, retrieve that function
            sensors.append(feature_functions.feature_function(item))
        else:
            # Assume it's a Callable
            sensors.append(item)

    # Have TICC label all the points in all the trajectories
    ticc_results = segmentation.ticc_label_trajectories(
        trajectories_df.Trajectory,
        sensors,
        num_clusters=num_labels,
        window_size=window_size,
        num_processors=10,
        iteration_limit=1000
        )

    # Attach point labels to the data frame
    trajectories_df['Segment Labels'] = ticc_results.point_labels

    # Now add the labels to the trajectory points
    for i, row in trajectories_df.iterrows():
        for trajectory_point, segment_label in zip(row.Trajectory, row['Segment Labels']):
            trajectory_point.set_property('segment_label', segment_label)

    # Save the correlation matrices from TICC in the DataFrame
    inverse_covariance_matrices = copy.deepcopy(ticc_results.markov_random_fields)
    correlation_matrices = matrix_utilities.inverse_covariance_to_correlation(
        inverse_covariance_matrices,
        window_size,
        len(sensors))

    correlation_matrices_first_toeplitz_column = [
        correlation_matrix[0] for correlation_matrix in correlation_matrices
    ]

    correlation_matrices_sequences = []
    for segment_labels in tqdm(trajectories_df['Segment Labels']):
        correlation_matrices_sequence = []
        for i in segment_labels:
            correlation_matrices_sequence.append(correlation_matrices_first_toeplitz_column[i])
        correlation_matrices_sequences.append(correlation_matrices_sequence)

    trajectories_df['Correlation Matrices'] = correlation_matrices_sequences

    # Logarithmically condense labels to reduce similar-length bias
    distance_between_matrices = [
        [0 for i in range(len(correlation_matrices_first_toeplitz_column))]
        for j in range(len(correlation_matrices_first_toeplitz_column))
        ]
    DISTANCE_BETWEEN_MATRICES = distance_between_matrices

    for i, corr_matrix1 in enumerate(correlation_matrices_first_toeplitz_column):
        corr_matrix_flat1 = corr_matrix1.flatten().reshape([1,-1])
        for j, corr_matrix2 in enumerate(correlation_matrices_first_toeplitz_column):
            corr_matrix_flat2 = corr_matrix2.flatten().reshape([1,-1])
            distance_between_matrices[i][j] = 1 - sklearn.metrics.pairwise.cosine_similarity(
                corr_matrix_flat1, corr_matrix_flat2)[0, 0]

    trajectories_df['Condensed Segment Labels'] = [
        condense_segment_labels(segment_labels)
        for segment_labels in trajectories_df['Segment Labels']
    ]

    segment_labels_as_arrays = [
        np.array(seg_labels)
        for seg_labels in trajectories_df['Segment Labels'].values
        ]
    condensed_segment_labels_as_arrays = [
        np.array(cond_seg_labels)
        for cond_seg_labels in trajectories_df['Condensed Segment Labels'].values
        ]

    # Cluster with DBSCAN
    dbscan_results = metric_dbscan.cluster_items(
        condensed_segment_labels_as_arrays,
        edit_distance_behavior_seq,
        minimum_cluster_size=minimum_cluster_size,
        maximum_neighbor_distance=maximum_neighbor_distance
        )

    # Save the cluster results in the data frame
    trajectories_df["dbscan_label"] = dbscan_results

    # Done!
    return trajectories_df





def condense_segment_labels(segment_labels):
    """Condense long runs of a single label into shorter runs"""
    max_linear = 10
    abbreviated_segment_labels = []

    linear_repeat_count = 1
    repeats_after_cutoff = 0

    for i in range(1,len(segment_labels)+1):
        if i == len(segment_labels) or segment_labels[i-1] != segment_labels[i]:
            abbreviated_segment_labels.extend([segment_labels[i-1]] * linear_repeat_count)
            abbreviated_segment_labels.extend([segment_labels[i-1]] * math.floor(math.log((repeats_after_cutoff+1))))
            linear_repeat_count = 1
            repeats_after_cutoff = 0
        else:
            if linear_repeat_count > max_linear:
                repeats_after_cutoff += 1
            else:
                linear_repeat_count += 1

    return abbreviated_segment_labels



@numba.jit(parallel=False, nopython=False)
def edit_distance_behavior_seq(labels_1: np.ndarray, labels_2: np.ndarray) -> float:
    """
    Compute the edit distance between two sequences of behavioral labels using dynamic programming.

    Parameters:
    labels_1: The first sequence of behavioral labels.
    labels_2: The second sequence of behavioral labels.

    Returns:
    int: The edit distance between two sequences of behavioral labels.
    """

    global DISTANCE_BETWEEN_MATRICES
    m = labels_1.shape[0]
    n = labels_2.shape[0]

    # Create a 2D array to store the edit distances
    dp = np.zeros([m + 1, n + 1])

    # Initialize the first row and first column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if labels_1[i - 1] == labels_2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                sub_penalty = min(2, 2*DISTANCE_BETWEEN_MATRICES[labels_1[i - 1]][labels_2[j - 1]] * 10)
                substitution = dp[i - 1][j - 1] + sub_penalty
                dp[i][j] = min(deletion, insertion, substitution)

    # Return the edit distance between the two sequences of behavioral labels
    return dp[m][n]