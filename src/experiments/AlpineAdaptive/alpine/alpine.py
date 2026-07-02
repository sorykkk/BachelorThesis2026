# Copyright (c) 2025 Valeo Comfort and Driving Assistance - Corentin Sautier @ valeo.ai

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import List, Dict
from scipy.sparse import csr_matrix
from scipy.spatial import QhullError
from scipy.spatial import cKDTree as KDTree
from .utils.box_fitting import fit_2d_box_modest
from scipy.sparse.csgraph import connected_components


class Alpine:
    """Class to perform the Alpine clustering

    Parameters
    ----------
    thing_indexes : List[int]
        List of indexes of the things, must match indexes given in the y array
    thing_bboxes : Dict[int, List[int]]
        Dictionary with the indexes of the things as keys and the bounding boxes as values
    k : int
        Number of neighbors to consider
    split : bool
        Whether to split the clusters or not using the box splitting scheme
    margin : float
        Margin to consider when splitting clusters

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
    
    References
    ----------
    Sautier, C., Puy, G., Boulch, A., Marlet, R., Lepetit, V., "Clustering is back: 
    Reaching state-of-the-art LiDAR instance segmentation without training".

    Examples
    --------
    >>> import numpy as np
    >>> from alpine import Alpine
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> thing_indexes = [0, 1]
    >>> thing_bboxes = {0: [2., 2.], 1: [2., 2.]}
    >>> alpine = Alpine(thing_indexes, thing_bboxes)
    >>> alpine.fit(X, y)
    >>> alpine.labels_
    array([1, 1, 2, 2])
    """

    def __init__(self, thing_indexes: List[int], thing_bboxes: Dict[int, List[float]], k:int=32, split:bool=False, margin:float=1.3, min_cluster_size:int=3):
        self.thing_indexes = thing_indexes
        for cls_key, v in thing_bboxes.items():
            if v[1] > v[0]:
                thing_bboxes[cls_key] = [v[1], v[0]]
        self.thing_bboxes = thing_bboxes
        self.k = k
        self.split = split
        self.margin = margin
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        assert len(self.thing_indexes) == len(self.thing_bboxes)

    def fit(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray = None):
        """Perform the Alpine clustering

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            The input point cloud in 2D euclidean space
        y : ndarray of shape (n_samples)
            The input semantic labels
        probs : ndarray of shape (n_samples, n_classes), optional
            Per-point class probabilities from the semantic backbone.
            When provided, graph edges are modulated by class confidence
            to reduce bridging between instances through uncertain points.
        """
        X = X[:, :2]
        offset = 1
        self.labels_ = np.zeros(X.shape[0], dtype=int)
        for object_class in self.thing_indexes:
            mask = y == object_class
            if (n:= mask.sum()) > 0:
                if n == 1:
                    # We need at least 2 points to cluster
                    self.labels_[mask] = offset
                    offset += 1
                    continue
                # Get the class's points
                pc_mask = X[mask]
                # Extract per-point probability for this class
                class_probs = probs[mask, object_class] if probs is not None else None
                # Obtain the clusters with probability-modulated edges
                clusters = self._clusterize(pc_mask, self.thing_bboxes[object_class][1], k=self.k, class_probs=class_probs)

                # Reassign small clusters to nearest large cluster
                if self.min_cluster_size > 0 and clusters[0] > 1:
                    clusters = self._reassign_small_clusters(pc_mask, clusters, self.min_cluster_size)

                if self.split:
                    inst_mask = np.zeros(pc_mask.shape[0], dtype=np.int32)
                    for id in range(clusters[0]):
                        # For all found clusters, split them if they are too big
                        mask_cluster = clusters[1] == id
                        sub_clusters = self._split_cluster(pc_mask[mask_cluster],
                                                          self.thing_bboxes[object_class][1], 
                                                          self.thing_bboxes[object_class],
                                                          margin=self.margin, 
                                                          k=self.k,
                                                          class_probs=class_probs[mask_cluster] if class_probs is not None else None)
                        inst_mask[mask_cluster] = sub_clusters[1] + offset
                        offset += sub_clusters[0]

                    self.labels_[mask] = inst_mask
                else:
                    self.labels_[mask] = clusters[1] + offset
                    offset += clusters[0]
        return self
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray = None):
        """Perform the Alpine clustering and return the labels

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            The input point cloud in 2D euclidean space
        y : ndarray of shape (n_samples)
            The input semantic labels
        probs : ndarray of shape (n_samples, n_classes), optional
            Per-point class probabilities from the semantic backbone.

        Returns
        -------
        labels : ndarray of shape (n_samples)
            The instance labels for each point in the dataset
        """
        self.fit(X, y, probs=probs)
        return self.labels_

    def _clusterize(self, pc, th, k, dist=None, neighbors=None, class_probs=None):
        # Project the pc to 2D
        pc = pc[:, :2]
        # Set k to at most the maximum number of points -1
        k = min((k, pc.shape[0]-1))
        # Get kNN
        if neighbors is None:
            kdtree = KDTree(pc)
            dist, neighbors = kdtree.query(pc, k=k+1)
            dist, neighbors = dist[:, 1:], neighbors[:, 1:]
        # Build graph
        orig = np.vstack([np.arange(pc.shape[0]) for _ in range(neighbors.shape[1])]).T
        # Apply threshold with optional probability modulation
        if class_probs is not None:
            # Modulate threshold by geometric mean of endpoint probabilities:
            # effective_th(i,j) = th * sqrt(p_i * p_j)
            # High confidence on both ends -> full threshold (unchanged)
            # Low confidence on either end -> reduced threshold (harder to connect)
            p_i = class_probs[orig.flatten()].reshape(orig.shape)
            p_j = class_probs[neighbors.flatten()].reshape(neighbors.shape)
            prob_scale = np.sqrt(np.maximum(p_i * p_j, 1e-10))
            weights = (dist < th * prob_scale).astype("float")
        else:
            weights = (dist < th).astype("float")
        W = csr_matrix(
            (weights.flatten(), (orig.flatten(), neighbors.flatten())), 
            shape=(pc.shape[0], pc.shape[0])
        )
        # Make graph non-oriented
        W = (W + W.T) / 2
        # Get connected components
        n, labels = connected_components(W, directed=False, return_labels=True)
        return n, labels

    def _split_cluster(self, pc, th, box_size, margin=1.3, k=32, class_probs=None):
        # If the cluster is of 2 points, box_fitting won't work.
        if pc.shape[0] < 3:
            return 1, np.zeros(pc.shape[0], dtype=np.int32)
        try:
            # Apply box_fitting and find box size in lxw
            _, x, y, _ = fit_2d_box_modest(pc[:, :2])
            # check if the box is smaller than the average (with margin)
            if max(x, y) < box_size[0] * margin and min(x, y) < box_size[1] * margin:
                return 1, np.zeros(pc.shape[0], dtype=np.int32)
            else:
                # binary search to find biggest th that splits the cluster in boxes smaller than box_size
                dt = th/2
                current_th = dt
                k_ = min((k, pc.shape[0]-1))
                kdtree = KDTree(pc)
                dist, neighbors = kdtree.query(pc, k=k_+1)
                dist, neighbors = dist[:, 1:], neighbors[:, 1:]
                while dt > 1e-3:
                    # we are looking for a threshold that splits the cluster in 2
                    dt = dt/2
                    clusters = self._clusterize(pc, current_th, k, dist, neighbors, class_probs=class_probs)
                    if clusters[0] == 1:
                        current_th -= dt
                    elif clusters[0] == 2:
                        # apply recursively to both clusters
                        mask_1 = clusters[1] == 0
                        mask_2 = clusters[1] == 1
                        n, clusters[1][mask_1] = self._split_cluster(pc[mask_1], current_th, box_size,
                                                                    margin=margin, k=k,
                                                                    class_probs=class_probs[mask_1] if class_probs is not None else None)
                        n2, cl = self._split_cluster(pc[mask_2], current_th, box_size,
                                                    margin=margin, k=k,
                                                    class_probs=class_probs[mask_2] if class_probs is not None else None)
                        clusters[1][mask_2] = cl + n
                        return n + n2, clusters[1]
                    else:
                        current_th += dt
        except QhullError:
            pass
        return 1, np.zeros(pc.shape[0], dtype=np.int32)

    def _reassign_small_clusters(self, pc, clusters, min_size):
        """Reassign points in small clusters to the nearest large cluster.
        Eliminates spurious tiny clusters that increase FP count.
        """
        n_clusters, labels = clusters
        if n_clusters <= 1:
            return clusters

        unique_labels, counts = np.unique(labels, return_counts=True)
        small_cluster_labels = set(unique_labels[counts < min_size])

        if not small_cluster_labels:
            return clusters

        large_cluster_labels = set(unique_labels[counts >= min_size])
        if not large_cluster_labels:
            return clusters  # all clusters are small, nothing to reassign to

        small_mask = np.isin(labels, list(small_cluster_labels))
        large_mask = ~small_mask

        # For each small-cluster point, find its nearest large-cluster point
        large_points = pc[large_mask, :2]
        large_point_labels = labels[large_mask]

        kdtree = KDTree(large_points)
        small_points = pc[small_mask, :2]
        _, nearest_idx = kdtree.query(small_points, k=1)

        new_labels = labels.copy()
        new_labels[small_mask] = large_point_labels[nearest_idx.flatten()]

        # Relabel to consecutive 0-based indices
        unique_new = np.unique(new_labels)
        label_remap = {old: new for new, old in enumerate(unique_new)}
        new_labels = np.vectorize(label_remap.__getitem__)(new_labels).astype(np.int32)

        return len(unique_new), new_labels
