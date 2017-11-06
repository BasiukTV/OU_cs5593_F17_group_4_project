**Folder for source code of our Data Mining Models**

## Directory/File Structure
- `clustering/` - Folder for contributor clustering models source code.
    - `clustering/cluster_model.py` - Abstract base class of a model produced by cluster modeling with different algorithms.
    - `clustering/kmeans.py` - Source code for KMeans contributor clustering.
- `regression/` - Folder for repository successfulness regression models source code.
    - `clustering/regression_model.py` - Abstract base class of a model produced by regression modeling with different algorithms.
- `modeling.py` - Abstract base class for different implementations of contributor clustering and repository regression model creation.
- `serializable_model.py` - Abstract base class of a serializable data mining model.

## Usage Examples
-- `python clustering/kmeans.py --help` or `python3 clustering/kmeans.py --help`