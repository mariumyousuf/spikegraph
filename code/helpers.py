import io
import contextlib
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare
from scipy import stats
from scipy.fftpack import fft, ifft
import bisect 
from scipy import signal
import pandas as pd
import seaborn as sns
from cdt.metrics import SHD
from cdt.metrics import precision_recall
import time
import random
import heapq
import random
import bisect
from scipy.fftpack import fft, ifft
from scipy.signal import convolve, correlate, correlation_lags
from sklearn.metrics import precision_recall_curve, auc
from statsmodels.tsa.stattools import grangercausalitytests
import lingam

def getSpikesInfo(fn):
    """
    Reads the spike train data saved from NEURON
    Takes in the filename (fn) and reads it as a multi-size numpy array
    
    Returns:
    --------
        the number of neurons (int), 
        the number of spikes per neuron (list of int),
        the spike times for each neuron (list of lists of floats)
    """
    data = pd.read_table(fn, header=None).to_numpy()
    numSpikes = []
    spikeTimes = []
    for i in range(np.shape(data)[0]):
        string = data[i][0]
        d = np.fromstring(string, dtype=float, sep=' ')
        numSpikes.append(int(d[0]))
        spikeTimes.append(d[1:])
    spikeTimes = spikeTimes[1:]
    numNeurons = numSpikes[0]
    numSpikes = numSpikes[1:]
    
    return numSpikes, spikeTimes

def randomISI(spikeTimes):
    """
    Randomizes the inter-spike intervals (ISIs) within each spike train while preserving
    the overall number of spikes and approximate timing structure.

    The function destroys any precise temporal correlations or spike patterns in the original
    data by shuffling the ISIs (differences between consecutive spike times), to preserve
    the distribution of ISIs and remove their temporal ordering.

    Parameters:
    -----------
    spikeTimes : list of 1D numpy arrays
        Each element in the list represents the spike times of a single neuron (or trial),
        as a sorted array of time values (in milliseconds).

    Returns:
    --------
    newSpikeTrains : list of 1D numpy arrays
        A list of spike trains where each train has the same ISIs as the original,
        but in a randomized order. The first spike time preserved.
    """
    np.random.seed(17)
    newSpikeTrains = []
    for spike in spikeTimes:
        arr = np.diff(spike)
        np.random.shuffle(arr)  # Shuffle differences in place
        newTrains = np.zeros(len(arr) + 1)  # Initialize with zeros, one more than the shuffled array
        newTrains[0] = spike[0]
        # Compute new spike times
        for l in range(1, len(arr)+1):
            newTrains[l] = newTrains[l-1]+arr[l-1]
        newSpikeTrains.append(newTrains)
    # Optionally shuffle the list of spike trains if required
    np.random.shuffle(newSpikeTrains)
    return newSpikeTrains

def getSpikeCountMx(h, N, spikeTimes):
    """
    Computes a spike count matrix based on the number of spikes in one spike train 
    that occur within a time window h after spikes in another spike train.

    The function calculates, for each pair of neurons (i, j), how many spikes from neuron `j`
    occurred within h time units *before* each spike of neuron i. The diagonal elements
    (self-comparisons) are excluded and set to zero.

    Parameters:
    -----------
    h : float
        The time window (in the same units as spike times) within which to count spikes.
        Only spike pairs where the time difference falls in [0, h] are counted.
    
    N : int
        The number of neurons or spike trains.

    spikeTimes : list of 1D numpy arrays
        A list where each element is a sorted array of spike times for a neuron.

    Returns:
    --------
    X : 2D numpy array of shape (N, N)
        A matrix where element X[i, j] represents the number of spikes in neuron j
        that occurred within h time units before each spike in neuron i.
        Diagonal elements X[i, i] are zero.
    """
    X = np.zeros((N, N), dtype=int)
    for i in range(N):
        t_times = np.array(spikeTimes[i])
        for j in range(N):
            if i == j:
                continue
            s_times = np.array(spikeTimes[j])
            # Calculate all pairwise differences
            diffs = t_times[:, None] - s_times[None, :]
            # Count the number of differences within the range [0, h]
            count = np.sum((diffs >= 0) & (diffs <= h))
            X[i, j] = count
    return X

def getGroundTruth(fn):
    """
    Loads and processes ground truth spike data from a text file.

    Assumes the file contains space-separated spike trains as strings,
    one per line. Skips the first two lines and binarizes the result.

    Parameters:
    -----------
    fn : str
        Path to the ground truth file.

    Returns:
    --------
    gt : 2D numpy array
        Binarized ground truth matrix (float), shape (neurons, time bins),
        where 1 indicates a spike and 0 indicates no spike.
    """
    data = pd.read_table(fn, header=None).to_numpy()
    gt = []
    for i in range(np.shape(data)[0]):
        string = data[i][0]
        d = np.fromstring(string, dtype=float, sep=' ')
        gt.append(d)
    gt = np.array(gt[2:][:])
    gt = (gt>0).astype(float)
    return gt

def getBinarizedSpikeMatrix(spike_times, tstop, N, window):
    """
    Converts spike times into a binarized spike matrix.

    Parameters:
        spike_times (list of lists): Each sublist contains spike times (in ms) for one neuron.
        tstop (float): Total simulation time in seconds.
        N (int): Number of neurons.
        window (int): Time bin size in milliseconds.

    Returns:
        np.ndarray: Binary matrix of shape (N, num_bins), where 1 indicates at least one spike
                    occurred in the corresponding time bin.
    """
    timeline = tstop*1000 # tstop is in s and timeline is in ms
    discList = np.arange(0, timeline+1, window) 
    T=np.zeros((N,len(discList)))
    for n, spikes in enumerate(spike_times):
        for s in spikes:
            i = bisect.bisect_left(discList, s)
            T[n][i-1]=1 # multiple fires within the window represented with a 1 
    return np.array(T)

def getBinnedSpikeMatrix(spike_times, tstop, N, window):
    """
    Converts spike times into a binned spike count matrix.

    Parameters:
        spike_times (list of lists): Each sublist contains spike times (in ms) for one neuron.
        tstop (float): Total simulation time in seconds.
        N (int): Number of neurons.
        window (int): Time bin size in milliseconds.

    Returns:
        np.ndarray: Matrix of shape (N, num_bins), where each entry contains the count
                    of spikes for a neuron in the corresponding time bin.
    """
    timeline = tstop*1000 # tstop is in s and timeline is in ms
    discList = np.arange(0, timeline+1, window) 
    T=np.zeros((N,len(discList)))
    for n, spikes in enumerate(spike_times):
        for s in spikes:
            i = bisect.bisect_left(discList, s)
            T[n][i-1]+=1 # count of fires within the window  
    return np.array(T)

def getCorrDirect(data, N, h):
    """
    Computes a directional cross-correlation matrix based on peak correlation lag.

    Parameters:
        data (list or array-like): List of N time series (1D arrays), one per neuron or signal.
        N (int): Number of time series (neurons).
        h (int): Maximum allowed lag (in time steps) to consider a valid correlation.

    Returns:
        np.ndarray: Asymmetric NxN matrix where entry (i, j) is nonzero if the peak 
                    cross-correlation between neuron i and j occurs with a lag ≤ h. 
                    Direction is encoded such that the leading neuron (earlier spike) 
                    points to the lagging neuron.
    """
    C_MX = np.zeros((N, N))
    for i in range(N):
        cell1 = data[i]
        for j in range(i+1, N):
            if i != j:
                cell2 = data[j]
                corr = correlate(cell1, cell2)
                lags = correlation_lags(len(cell1), len(cell2))
                max_lag = lags[np.argmax(corr)]
                if abs(max_lag) <= h and abs(max_lag)!=0:
                    if max_lag > 0:
                        C_MX[i][j] = abs(max_lag)
                    else:
                        C_MX[j][i] = abs(max_lag)
    
    return C_MX

def silent_granger_test(data, maxlag, addconst=True, **kwargs):
    """
    Runs the Granger causality test silently (without console output)
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return grangercausalitytests(data, maxlag, addconst=addconst, **kwargs)

def compute_granger_matrix(data, max_lag, test='lrtest'):
    """
    Computes a Granger causality matrix between all pairs of binary time series.

    Parameters:
        data (array-like): 2D array of shape (N, T), where N is the number of binary variables 
                           (e.g., neurons) and T is the number of time steps.
        max_lag (int): Maximum lag to use in Granger causality testing.
        test (str): Test statistic to extract from the Granger test result. 
                    For binary data, 'lrtest' (likelihood ratio test) is recommended 
                    due to its suitability for discrete outcomes.

    Returns:
        np.ndarray: Normalized N x N matrix of Granger causality test statistics.
                    Entry (i, j) represents the strength with which time series j 
                    Granger-causes i.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    N = np.shape(data)[0]
    df = pd.DataFrame(data).T
    gc_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                result = silent_granger_test(df[[i, j]], [max_lag])#grangercausalitytests(df[[i, j]], [max_lag], verbose=False)
                val = result[max_lag][0][test][0]#result[max_lag][0][test][1]
                gc_matrix[i, j] = val
    
#     if sig_threshold is not None:
#         gc_matrix = (gc_matrix < sig_threshold).astype(int)
        
    return (gc_matrix - np.min(gc_matrix)) / (np.max(gc_matrix) - np.min(gc_matrix))

def compute_smooth_pr(gt_matrix, score_matrix, recall_levels=np.linspace(0, 1, 100)):
    """
    Computes a smoothed precision-recall (PR) curve by interpolating at fixed recall levels.

    Parameters:
        gt_matrix (np.ndarray): Ground truth binary matrix of shape (N, N).
        score_matrix (np.ndarray): Predicted scores (e.g., from Granger or correlation).
        recall_levels (np.ndarray): Recall levels at which to interpolate precision.

    Returns:
        interpolated_precision: Precision values at specified recall levels
    """
    y_true = gt_matrix.flatten().astype(int)
    y_scores = score_matrix.flatten().astype(float)

    if np.sum(y_true) == 0:
        print("Warning: GT has no positive edges.")
        return np.zeros_like(recall_levels), recall_levels

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    recall = np.array(recall)
    precision = np.array(precision)
    
    # enforce monotonicity for interpolation
    recall, indices = np.unique(recall, return_index=True)
    precision = precision[indices]

    # interpolate precision at fixed recall levels
    precision_interp = np.interp(recall_levels, recall, precision, left=1.0, right=0.0)
    return precision_interp

import numpy as np 
from typing import List, Tuple, Dict
from collections import Counter

def count_unique_elements(matrix):
    # Flatten the matrix into a single list
    flat_list = [element for row in matrix for element in row]
    
    # Count occurrences using Counter
    counts = Counter(flat_list)
    
    # Display results
    print("Unique elements and their counts:")
    for element, count in counts.items():
        print(f"Element: {element}, Count: {count}")
    
    return counts

def prepare_paths_from_spikes(spikes: np.ndarray, T: int, stride: int = 1) -> Tuple[np.ndarray, int]:
    """
    Converts spike trains into 'paths' suitable for model training.

    Args:
        spikes: (N, L) array of binarized spikes (0/1 or -1/1)
        T: trajectory length (same as in model config)
        stride: time steps between start of consecutive trajectories (default 1)

    Returns:
        paths: (K, T+1) array of state indices
        N: number of neurons
    """
    # Step 1: Transpose so each row is a time step
    if spikes.shape[0] < spikes.shape[1]:
        states = spikes.T  # (L, N)
    else:
        states = spikes  # already (L, N)

    # Step 2: Convert {-1, 1} to {0, 1} if needed
    if np.any(states == -1):
        print("Detected bipolar input. Converting to binary {0,1}.")
        states = ((states + 1) // 2).astype(int)

#     print("states, np.max(states)", states, np.max(states))
    
    L, N = states.shape
#     print("L, N", L, N)
    
    if L < T + 1:
        raise ValueError(f"Not enough time steps ({L}) to form a trajectory of length T+1 = {T+1}")

    # Step 3: Convert binary state to index (0 to 2^N - 1)
    powers = 2 ** np.arange(N - 1, -1, -1)
#     print("powers", powers)
    indices = np.dot(states, powers)  # (L,)
#     print("indices", indices, np.shape(indices))

    # Step 4: Create sliding window of trajectories
    K = (L - T) // stride
#     print("K", K)
    paths = np.empty((K, T + 1), dtype=np.int64)
    for k in range(K):
        start = k * stride
        paths[k] = indices[start : start + T + 1]

    return paths