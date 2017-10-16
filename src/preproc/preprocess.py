import json, multiprocessing

DEFAULT_INPUT_DIRECTORY = "/samples/data/raw"; # Relative to home directory of the application
DEFAULT_FILENAME_PATTERN = "YYYY-MM-DD-HH*.json"; # Tail wildcard is for labels ("-sample" for example)
DEFAULT_THREADS_NUMBER = 5;

# Process files in one week of worth data batches. Assuming one file holds an hour worth of data.
DEFAULT_FILE_PREPROCESSING_BATCH_SIZE = 7 * 24;

def preprocess_files(files, threads_num):
    """This preprocesses given list of log files, using given number of threads."""

    # Initialize main database. TODO This is part of issue #11
    main_database = None

    # Determine size of a batch and number of batch tuns required to process all files
    batch_size = DEFAULT_FILE_PREPROCESSING_BATCH_SIZE
    batch_runs = 1 + (len(files) // batch_size)
    print("Preprocessing will take {} run(s), each processing a batch of up to {} file(s).".format(batch_runs, batch_size))

    # Initialize the thread pool
    thread_pool = multiprocessing.Pool(threads_num)

    # Loop which processes separate batches of input files
    for batch_run in range(1, batch_runs + 1):
        print("Starting run #{}".format(batch_run))

        starting_file_index = batch_size * (batch_run - 1)
        files_to_process = batch_size if batch_run != batch_runs else len(files) - starting_file_index
        print("This run will process {} files.".format(files_to_process))

        # Intialize the intermidiate_database TODO This is part of issue #6
        intermidiate_database = None

        thread_pool.map(process_file, zip(files[starting_file_index : starting_file_index + files_to_process], [intermidiate_database] * files_to_process))
        # TODO Merge Intermediate and Main Database here. This is part of issue #6

    # TODO Deserialize main database into output directory. This is part of issue #11

def process_file(path_to_file_and_database):
    (path_to_file, database) = path_to_file_and_database
    starcount = 0

    try:
        with open(path_to_file, encoding="utf8") as json_file:
            for line in json_file:
                obj = json.loads(line)
                t = obj["type"]
                if t == "WatchEvent":
                    starcount += 1
    except IOError as er:
        print(er)
        pass

    print("{} new stars on {}".format(starcount, path_to_file))

# Entry point for running pre-processing step separately
if __name__ == "__main__":
    import argparse, os, sys

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for parsing input GitHub logs into application specific dataset.")
    parser.add_argument("-dir", "--directory", help="Input logs file directory.")
    parser.add_argument("-t", "--threads", help="Number of threads to use for processing.", default=DEFAULT_THREADS_NUMBER)
    parser.add_argument("-y", "--year", help="Year of input data to process. In YYYY format.")
    parser.add_argument("-m", "--month", help="Month of input data to process. In MM format.")
    parser.add_argument("-d", "--day", help="Day of input data to process. In DD format.")
    parser.add_argument("-hr", "--hour", help="Hour of input data to process. In HH format.")
    args = parser.parse_args()

    print("Pre-processing started.")

    app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
    print("Application home directory is: {}".format(app_home_dir))

    # Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
    app_src_dir = os.path.realpath(app_home_dir + "/src")
    print("Application source code directory is: {}".format(app_src_dir))
    sys.path.insert(0, app_src_dir)

    app_data_dir = args.directory if args.directory else os.path.realpath(app_home_dir + DEFAULT_INPUT_DIRECTORY);
    print("Application input data directory is: {}".format(app_data_dir))

    filename_pattern = DEFAULT_FILENAME_PATTERN
    filename_pattern = filename_pattern.replace("YYYY", args.year if args.year else "*")
    filename_pattern = filename_pattern.replace("MM", args.month if args.month else "*")
    filename_pattern = filename_pattern.replace("DD", args.day if args.day else "*")
    filename_pattern = filename_pattern.replace("HH", args.hour if args.hour else "*")
    filename_pattern = filename_pattern.replace("**", "*") # Getting rid of a potential double-wildcard which is not what we want
    print("Input Data Filename Pattern: {}".format(filename_pattern))

    print("Using Number Of Threads: {}".format(args.threads))

    from utils.file_utils import discover_directory_files
    input_files = discover_directory_files(app_data_dir, filename_pattern=filename_pattern)
    print("Discovered Following Input Data Files: {}".format(input_files))

    preprocess_files(input_files, int(args.threads))
