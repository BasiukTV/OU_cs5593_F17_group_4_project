**Folder for source code of our Data Mining Models**

## Directory/File Structure
- `clustering/` - Folder for contributor clustering models source code.
    - `clustering/cluster_model.py` - Abstract base class of a model produced by cluster modeling with different algorithms.
    - `clustering/hierarchical.py` - Source code for hierarchical contributor clustering.
    - `clustering/kmeans.py` - Source code for KMeans contributor clustering.
- `regression/` - Folder for repository successfulness regression models source code.
    - `regression/regression_model.py` - Abstract base class of a model produced by regression modeling with different algorithms.
    - `regression/logistic.py` - Source code for logistic repository regression.
    - `regression/svr.py` - Source code for SVR repository regression.
- `modeling.py` - Abstract base class for different implementations of contributor clustering and repository regression model creation.
- `serializable_model.py` - Abstract base class of a serializable data mining model.

## Usage Examples
- `python clustering/hierarchical.py --help` or `python3 clustering/hierarchical.py --help`
- `python clustering/kmeans.py --help` or `python3 clustering/kmeans.py --help`
- `python regression/regression_model.py --help` or `python3 regression/regression_model.py --help`
- `python regression/logistic.py --help` or `python3 regression/logistic.py --help`