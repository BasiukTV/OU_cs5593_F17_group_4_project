from __future__ import print_function
import json
import multiprocessing
import sqlite3
import io
import os
from time import gmtime, strftime

DEFAULT_INPUT_DIRECTORY = "/input/data/raw"; # Relative to home directory of the application
DEFAULT_INPUT_FILENAME_PATTERN = "YYYY-MM-DD-HH[-LABEL].json";

DEFAULT_OUTPUT_DIRECTORY = '/output/data/preproc'; # Relative to home directory of the application
DEFAULT_OUTPUT_FILENAME_PATTERN = "preprocessed[-LABEL].sqlite3";

DEFAULT_THREADS_NUMBER = 5;

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

def preprocess_files(files, threads_num, output_file_path):
    """This preprocesses given list of log files, using given number of threads."""
    print("{}: preprocessing files".format(now()))

    # Initialize main database.
    try:
        """
        If output file path already exists. Remove that file.
        It's OK to remove files in output directory.
        If main application needed them, they were copied over into input data directory.
        """
        os.remove(output_file_path)
    except:
        pass

    con = sqlite3.connect(output_file_path)
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
            process_file((f, output_file_path))

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
        with io.open(path_to_file, encoding="utf8") as json_file:
            for lineno, line in enumerate(json_file):
                try:
                    obj = json.loads(line)
                    event_type = obj.get("type")
                    event_id = obj.get("id") # not all events have ids, can be None
                    event_time = obj.get("created_at")
                    actor = obj.get("actor", {})
                    # old records store just the name of the actor with a seperate actor_attributes field (which doesn't contain the id either)
                    actor_id = actor.get("id") if isinstance(actor, dict) else None
                    repo_id = obj.get("repo", {}).get("id") # can be None
                    payload = obj.get("payload", {})
                    std = (None, event_id, repo_id, event_time, actor_id) # relevant attributes every event has

                    if event_type == "WatchEvent":
                        cur.execute("INSERT INTO starrings VALUES(?, ?, ?, ?, ?)", std)
                    elif event_type == "CreateEvent":
                        # TODO handle old CreateEvents without description
                        # TODO and without pusher_type
                        description = payload.get("description") # can be None
                        description_len = None if description is None else len(description)
                        cur.execute("INSERT INTO creations VALUES(?, ?, ?, ?, ?, ?, ?)", std + (description_len, payload.get("pusher_type")))
                    elif event_type == "PushEvent":
                        cur.execute("INSERT INTO pushes VALUES(?, ?, ?, ?, ?)", std)
                        for commit in payload.get("commits", []): # TODO handle alternative 2011 format (shas instead of commits?)
                            # TODO handle without distinct
                            cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (event_id, commit["author"]["name"], commit["message"], commit.get("distinct", True)))
                    elif event_type == "ReleaseEvent":
                        release = payload["release"]
                        cur.execute("INSERT INTO releases VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (release.get("tag_name"), release.get("name"), release.get("prerelease"), len(release.get("assets"))))
                    elif event_type == "PullRequestEvent":
                        pr = payload.get("pull_request")
                        a = payload.get("action")
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
                            body = payload.get("body") # can be None
                            body_len = None if body is None else len(body)
                            cur.execute("INSERT INTO pr_opens VALUES(?, ?, ?, ?, ?, ?, ?)", std + (pr.get("title") , body_len))
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
                except Exception as e: 
                    from sys import stderr
                    print("{}: Error while processing line {} of {}:\n{}".format(now(),lineno, path_to_file, e), file=stderr)
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
    parser.add_argument("-id", "--indir", help="Input logs file directory.")
    parser.add_argument("-od", "--outdir", help="Output directory for preprocessed dataset.")
    parser.add_argument("-l", "--label", help="Run label. Only input files with the label will be used. Output files will contain the label.")
    parser.add_argument("-t", "--threads", help="Number of threads to use for processing.", default=DEFAULT_THREADS_NUMBER)
    parser.add_argument("-y", "--year", help="Year of input data to process. In YYYY format.")
    parser.add_argument("-m", "--month", help="Month of input data to process. In MM format.")
    parser.add_argument("-d", "--day", help="Day of input data to process. In DD format.")
    parser.add_argument("-hr", "--hour", help="Hour of input data to process. In HH format.")
    args = parser.parse_args()

    print("Pre-processing started.")

    if args.label:
        print("This run is labeled: '{}'.".format(args.label))

    app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
    print("Application home directory is: {}".format(app_home_dir))

    # Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
    app_src_dir = os.path.realpath(app_home_dir + "/src")
    print("Application source code directory is: {}".format(app_src_dir))
    sys.path.insert(0, app_src_dir)

    app_data_indir = args.indir if args.indir else os.path.realpath(app_home_dir + DEFAULT_INPUT_DIRECTORY);
    print("Application input data directory is: {}".format(app_data_indir))

    filename_pattern = DEFAULT_INPUT_FILENAME_PATTERN
    filename_pattern = filename_pattern.replace("YYYY", args.year if args.year else "????")
    filename_pattern = filename_pattern.replace("MM", args.month if args.month else "??")
    filename_pattern = filename_pattern.replace("DD", args.day if args.day else "??")
    filename_pattern = filename_pattern.replace("HH", args.hour if args.hour else "??")
    filename_pattern = filename_pattern.replace("[-LABEL]", "-" + args.label if args.label else "")
    print("Input Data Filename Pattern: {}".format(filename_pattern))

    from utils.file_utils import discover_directory_files
    input_files = discover_directory_files(app_data_indir, filename_pattern=filename_pattern)
    file_count = len(input_files)
    if file_count > 0:
        print("Discovered {} Input Data Files, starting with {} and ending with {}".format(file_count, input_files[0], input_files[-1]))
    else:
        print("Discovered no Input Data Files")

    app_data_outdir = args.outdir if args.outdir else os.path.realpath(app_home_dir + DEFAULT_OUTPUT_DIRECTORY);
    print("Application preprocessed dataset directory will be: {}".format(app_data_outdir))

    output_filename = DEFAULT_OUTPUT_FILENAME_PATTERN
    output_filename = output_filename.replace("[-LABEL]", "-" + args.label if args.label else "")
    output_filename_path = os.path.join(app_data_outdir, output_filename)
    print("Application preprocessed dataset filename will be: {}".format(output_filename_path))

    print("Using Number Of Threads: {}".format(args.threads))

    preprocess_files(input_files, int(args.threads), output_filename_path)
