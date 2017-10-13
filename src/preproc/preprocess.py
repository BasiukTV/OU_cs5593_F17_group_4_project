import json, multiprocessing

DEFAULT_INPUT_DIRECTORY = "/samples/data/raw"; # Relative to home directory of the application
DEFAULT_FILENAME_PATTERN = "YYYY-MM-DD-HH*.json"; # Tail wildcard is for labels ("-sample" for example)
DEFAULT_THREADS_NUMBER = 5;

def process_day(month_and_day):
    (month, day) = month_and_day
    starcount = 0
    for hour in range (0,24):
        filename = "/scratch/timo/github/2016-{:02}-{:02}-{}.json".format(month, day, hour)
        try:
            with open(filename) as json_file:
                for line in json_file:
                    obj = json.loads(line)
                    t = obj["type"]
                    if t == "WatchEvent":
                        starcount += 1
        except IOError as er:
            print(er)
            pass

    print("{} new stars on 2017-{:02}-{:02}".format(starcount, month, day))


def process_month(month):
    out = pool.map(process_day, zip([month] * 31, range(1,32)))

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

    pool = multiprocessing.Pool(int(args.threads))
    for month in range(1,4):
        process_month(month)
