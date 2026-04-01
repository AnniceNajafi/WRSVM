"""WRSVM: Weighted Relaxed Support Vector Machine for multiclass classification.

Four decomposition strategies (Crammer-Singer, SimMSVM, OVO, OVR) exposed
through a single scikit-learn compatible classifier.
"""

from wrsvm.classifier import WRSVMClassifier
from wrsvm.decomposition import OVOClassifier, OVRClassifier
from wrsvm.kernels import rbf_kernel
from wrsvm.noise import inject_outliers_majority, inject_outliers_minority
from wrsvm.simmsvm import predict_simmsvm, solve_simmsvm
from wrsvm.solver import predict, solve_crammer_singer

__version__ = "0.2.0"
__all__ = [
    "WRSVMClassifier",
    "OVOClassifier",
    "OVRClassifier",
    "rbf_kernel",
    "solve_crammer_singer",
    "solve_simmsvm",
    "predict",
    "predict_simmsvm",
    "inject_outliers_majority",
    "inject_outliers_minority",
]
