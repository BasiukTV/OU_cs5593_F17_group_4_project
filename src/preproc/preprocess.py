import json
import multiprocessing
import sqlite3

DEFAULT_INPUT_DIRECTORY = "/samples/data/raw"; # Relative to home directory of the application
DEFAULT_FILENAME_PATTERN = "YYYY-MM-DD-HH*.json"; # Tail wildcard is for labels ("-sample" for example)
DEFAULT_THREADS_NUMBER = 5;
DB_PATH = 'database.db'; # TODO proper file path

# Process files in one week of worth data batches. Assuming one file holds an hour worth of data.
DEFAULT_FILE_PREPROCESSING_BATCH_SIZE = 7 * 24;

def setup_db_scheme(cur):
    # TODO: instead of repeating the same attributes again and again, we could create a sperate "event" table.
    # I'm not sure aobut the performance implications though: Basically every action would require a join.
    common_attrs = '''
        event_id primary key,
        repo_id number,
        time text,
        actor_id number
    '''
    cur.execute('''
        CREATE TABLE starrings (
            {}
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE creations (
            {},
            description_len number,
            pusher_type text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE pushes (
            {}
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE commits (
            event_id number,
            author_name text,
            message text,
            distinct_ bool,
            FOREIGN KEY (event_id) REFERENCES pushes
        )
    ''')
    cur.execute('''CREATE TABLE repos (
        repo_id number PRIMARY KEY,
        number_of_stars number
        )
    ''')

def aggregate_data(con):
    cur1 = con.cursor()
    cur2 = con.cursor()
    for row in cur1.execute("SELECT DISTINCT repo_id FROM starrings"):
        repoid = row[0]
        cur2.execute("SELECT count(*) FROM starrings WHERE repo_id = ?", (str(repoid),))
        stars = cur2.fetchone()[0]

        cur2.execute("INSERT INTO repos VALUES (?, ?)", (repoid, stars))

def preprocess_files(files, threads_num):
    """This preprocesses given list of log files, using given number of threads."""

    # Initialize main database.
    # TODO should we be able to update an existing db?
    try:
        os.remove(DB_PATH)
    except:
        pass
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    setup_db_scheme(cur);

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

        thread_pool.map(
                process_file, # execute process_file
                zip(
                    files[starting_file_index : starting_file_index + files_to_process], # on these files
                    [DB_PATH] * files_to_process) # with that database
                )

    aggregate_data(con)
    # DEBUG
    for row in cur.execute('SELECT * FROM repos'):
        print(row)

    con.commit()
    con.close()
    # TODO Deserialize main database into output directory. This is part of issue #11

def process_file(path_to_file_and_database):
    (path_to_file, path_to_database) = path_to_file_and_database
    db = sqlite3.connect(path_to_database)
    cur = db.cursor()
    print("Processing {}".format(path_to_file))

    try:
        with open(path_to_file) as json_file:
            for line in json_file:
                obj = json.loads(line)
                event_type = obj["type"]
                event_id = obj["id"]
                event_time = obj["created_at"]
                actor_id = obj["actor"]["id"]
                repo_id = obj["repo"]["id"]
                payload = obj["payload"]
                std = (event_id, repo_id, event_time, actor_id) # relevant attributes every event has

                if event_type == "WatchEvent":
                    cur.execute("INSERT INTO starrings VALUES(?, ?, ?, ?)", std)
                elif event_type == "CreateEvent":
                    cur.execute("INSERT INTO creations VALUES(?, ?, ?, ?, ?, ?)", std + (len(payload["description"] or ""), payload["pusher_type"]))
                elif event_type == "PushEvent":
                    cur.execute("INSERT INTO pushes VALUES(?, ?, ?, ?)", std )
                    for commit in payload["commits"]:
                        cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (event_id, commit["author"]["name"], commit["message"], commit["distinct"]))
    except IOError as er:
        print(er)
        pass
    db.commit()
    db.close()

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
