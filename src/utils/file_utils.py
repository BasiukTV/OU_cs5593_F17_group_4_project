def discover_directory_files(directory, filename_pattern="*", sort_return=True):
    """
    Returns a list of absolute paths to files matching a unix-like filename pattern in a given directory.
    By default returned list is sorted alphanumericaly.
    """
    import glob, os

    filepaths = glob.glob(os.path.join(directory, filename_pattern))
    return sorted(filepaths) if sort_return else filepaths