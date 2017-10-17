import json
import multiprocessing
import sqlite3
from time import gmtime, strftime

DEFAULT_INPUT_DIRECTORY = "/samples/data/raw"; # Relative to home directory of the application
DEFAULT_FILENAME_PATTERN = "YYYY-MM-DD-HH*.json"; # Tail wildcard is for labels ("-sample" for example)
DEFAULT_THREADS_NUMBER = 5;
DB_PATH = 'database.db'; # TODO proper file path

# Process files in one week of worth data batches. Assuming one file holds an hour worth of data.
DEFAULT_FILE_PREPROCESSING_BATCH_SIZE = 7 * 24;

def now():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def setup_db_scheme(cur):
    print("{}: setting up db scheme".format(now()))
    # TODO: instead of repeating the same attributes again and again, we could create a sperate "event" table.
    # I'm not sure aobut the performance implications though: Basically every action would require a join.
    common_attrs = '''
        id integer primary key autoincrement,
        event_id number,
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
            distinct_ bool
        )
    ''')
    cur.execute('''
        CREATE TABLE releases (
            {},
            tag_name text,
            name text,
            prerelease bool,
            num_assets number
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE pr_opens (
            {},
            title text,
            body_len number
        )
    '''.format(common_attrs))
    cur.execute('''CREATE TABLE repos (
        repo_id number PRIMARY KEY,
        number_of_stars number
        )
    ''')

def aggregate_data(con):
    print("{}: aggregating data".format(now()))
    cur1 = con.cursor()
    cur2 = con.cursor()
    for row in cur1.execute("SELECT DISTINCT repo_id FROM starrings"):
        repoid = row[0]
        cur2.execute("SELECT count(*) FROM starrings WHERE repo_id = ?", (str(repoid),))
        stars = cur2.fetchone()[0]

        cur2.execute("INSERT INTO repos VALUES (?, ?)", (repoid, stars))

def preprocess_files(files, threads_num):
    """This preprocesses given list of log files, using given number of threads."""
    print("{}: preprocessing files".format(now()))

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
    print("{}: Preprocessing will take {} run(s), each processing a batch of up to {} file(s).".format(now(), batch_runs, batch_size))

    # Initialize the thread pool
    thread_pool = multiprocessing.Pool(threads_num)

    # Loop which processes separate batches of input files
    for batch_run in range(1, batch_runs + 1):
        print("{}: Starting run #{}".format(now(), batch_run))

        starting_file_index = batch_size * (batch_run - 1)
        files_to_process = batch_size if batch_run != batch_runs else len(files) - starting_file_index
        print("{}: This run will process {} files.".format(now(), files_to_process))

        for f in files[starting_file_index: starting_file_index + files_to_process]:
            process_file((f, DB_PATH))

        # thread_pool.map(
        #         process_file, # execute process_file
        #         zip(
        #             files[starting_file_index : starting_file_index + files_to_process], # on these files
        #             [DB_PATH] * files_to_process) # with that database
        #         )

    aggregate_data(con)
    # DEBUG
    # for row in cur.execute('SELECT * FROM repos'):
    #     print(row)

    con.commit()
    con.close()
    # TODO Deserialize main database into output directory. This is part of issue #11

def process_file(path_to_file_and_database):
    (path_to_file, path_to_database) = path_to_file_and_database
    db = sqlite3.connect(path_to_database)
    cur = db.cursor()
    print("{}: Processing {}".format(now(), path_to_file))

    try:
        with open(path_to_file) as json_file:
            for lineno, line in enumerate(json_file):
                obj = json.loads(line)
                try:
                    event_type = obj["type"]
                    event_id = obj.get("id", -1) # TODO not all events have ids
                    event_time = obj["created_at"]
                    actor_id = obj["actor"]["id"]
                    repo_id = obj["repo"].get("id", -1) # TODO handle old events without repo ids
                    payload = obj["payload"]
                    std = (None, event_id, repo_id, event_time, actor_id) # relevant attributes every event has

                    if event_type == "WatchEvent":
                        cur.execute("INSERT INTO starrings VALUES(?, ?, ?, ?, ?)", std)
                    elif event_type == "CreateEvent":
                        # TODO handle old CreateEvents without description
                        # TODO and without pusher_type
                        cur.execute("INSERT INTO creations VALUES(?, ?, ?, ?, ?, ?, ?)", std + (len(payload.get("description", "") or ""), payload.get("pusher_type", "")))
                    elif event_type == "PushEvent":
                        cur.execute("INSERT INTO pushes VALUES(?, ?, ?, ?, ?)", std)
                        for commit in payload.get("commits", []): # TODO handle alternative 2011 format (shas instead of commits?)
                            # TODO handle without distinct
                            cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (event_id, commit["author"]["name"], commit["message"], commit.get("distinct", True)))
                    elif event_type == "ReleaseEvent":
                        release = payload["release"]
                        cur.execute("INSERT INTO releases VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (release["tag_name"], release["name"], release["prerelease"], len(release["assets"])))
                    elif event_type == "PullRequestEvent":
                        pr = payload["pull_request"]
                        a = payload["action"]
                        # TODO consider `locked` attribute
                        if a == "assigned":
                            pass # TODO
                        elif a == "unassigned":
                            pass # TODO
                        elif a == "review_requested":
                            pass # TODO
                        elif a == "review_request_removed":
                            pass # TODO
                        elif a == "labeled":
                            pass # TODO
                        elif a == "unlabeled":
                            pass # TODO
                        elif a == "opened":
                            cur.execute("INSERT INTO pr_opens VALUES(?, ?, ?, ?, ?, ?, ?)", std + (pr["title"] , len(pr.get("body", "") or ""))) # TODO handle old events without body
                        elif a == "edited":
                            pass # TODO
                        elif a == "closed":
                            if pr.get("merged", True): # TODO handle old events without merged
                                pass # pr merged # TODO
                            else:
                                pass # pr discarded # TODO
                        elif a == "reopened":
                            pass # TODO
                        elif a == "edited":
                            pass # TODO
                except KeyError as er:
                    print("{}-l{}: Found {} without key {} in event {}".format(path_to_file, lineno, event_type, er, event_id))
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
