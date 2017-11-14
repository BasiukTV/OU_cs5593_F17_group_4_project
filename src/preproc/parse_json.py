import json
import multiprocessing
import sqlite3
import io
import os
import sys
from time import gmtime, strftime
from datetime import datetime
import queue

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from utils.logging import log, elog
from utils.parse import parse_isotime

class EventDBQueue:
    def __init__(self):
        self.queue = multiprocessing.Queue()

    def insert_event(self, event_id, repo_id, repo_name, repo_owner_name, repo_owner_id, time, actor_id, actor_name):
        self.queue_insert("INSERT INTO event VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", (None, event_id, repo_id, repo_name, repo_owner_name, repo_owner_id, time, actor_id, actor_name))

    def queue_insert(self, sql, args):
        self.queue.put((sql, args))

class EventDB:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.queue = EventDBQueue()
        self.setup_db_scheme()

    def setup_db_scheme(self):
        log("setting up db scheme")
        self.con.execute('''
            CREATE TABLE IF NOT EXISTS event (
            id integer primary key autoincrement,
            event_id integer,
            repo_id integer,
            repo_name text,
            repo_owner_name text,
            repo_owner_id integer,
            time text,
            actor_id integer,
            actor_name integer
        )
        ''')
        self.con.commit()

    def process_queue(self):
        log("Processing queue")
        count = 0
        # buf = []
        while True:
            try:
                # buf.append(self.queue.get(False))
                (sql, args) = self.queue.queue.get(False)
                self.con.execute(sql, args)
                count += 1
            except queue.Empty:
                break
        # log("Processing {} events", len(buf))
        # self.con.executemany("INSERT INTO event VALUES (?)", buf)
        self.con.commit()
        log("Done processing {} statements", count)


DEFAULT_INPUT_DIRECTORY = "/input/data/raw"; # Relative to home directory of the application
DEFAULT_INPUT_FILENAME_PATTERN = "YYYY-MM-DD-HH[-LABEL].json";

DEFAULT_OUTPUT_DIRECTORY = '/output/data/preproc'; # Relative to home directory of the application
DEFAULT_OUTPUT_FILENAME_PATTERN = "preprocessed[-LABEL].sqlite3";

DEFAULT_THREADS_NUMBER = 5;

# Process files in one week of worth data batches. Assuming one file holds an hour worth of data.
DEFAULT_FILE_PREPROCESSING_BATCH_SIZE = 7 * 24;

def sql_writer(db_path, queue):
    db = sqlite3.connect(db_path)
    counter = 0
    while True:
        args = queue.get() # blocks
        if args is None:
            break
        db.executemany("INSERT INTO event VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", args)
        # db.commit()
        log("SQL backlog: {}".format(queue.qsize()))
        counter += 1
        if counter > 100:
            db.commit()
    db.commit()
    db.close()
    log("Finished all writing")

def preprocess_files(files, threads_num, output_file_path):
    """This preprocesses given list of log files, using given number of threads."""
    log("preprocessing files")

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

    db = EventDB(output_file_path)

    queue = multiprocessing.Queue(100)
    writer = multiprocessing.Process(target=sql_writer, args = (output_file_path, queue))
    writer.start()

    # Determine size of a batch and number of batch tuns required to process all files
    # batch_size = DEFAULT_FILE_PREPROCESSING_BATCH_SIZE
    batch_size = 20
    batch_runs = 1 + (len(files) // batch_size)
    log("Preprocessing will take {} run(s), each processing a batch of up to {} file(s).", batch_runs, batch_size)

    # Initialize the thread pool
    thread_pool = multiprocessing.Pool(threads_num - 1)

    # Loop which processes separate batches of input files
    for batch_run in range(1, batch_runs + 1):
        log("Starting run #{}", batch_run)

        starting_file_index = batch_size * (batch_run - 1)
        files_to_process = batch_size if batch_run != batch_runs else len(files) - starting_file_index
        log("This run will process {} files.", files_to_process)

        for f in files[starting_file_index: starting_file_index + files_to_process]:
            thread_pool.apply_async(process_file, (f,), callback = queue.put)

    thread_pool.close()
    thread_pool.join()
    queue.put(None)
    writer.join()

def process_file(path_to_file):
    log("Processing {}", path_to_file)
    result = []

    with io.open(path_to_file, encoding="utf8") as json_file:
        for lineno, line in enumerate(json_file):
            try:
                obj = json.loads(line)
                payload = obj.get("payload", {})
                event_type = obj.get("type")
                event_id = obj.get("id") # not all events have ids, can be None
                event_time = obj.get("created_at")
                # normalize to ISO 8601 in UTC
                if event_time[-1] == 'Z':
                    event_time = event_time[:-1]
                else:
                    tmp = parse_isotime(event_time[:-6])
                    if event_time[-3] == ':':
                        offset_str = event_time[-6:-3] + event_time[-2:]
                    else:
                        offset_str = event_time[-5:]
                    offset = datetime.strptime(offset_str, "%z")
                    event_time = (tmp - offset.utcoffset()).isoformat()
                actor = obj.get("actor", {})
                # old records store just the name of the actor with a seperate actor_attributes field (which doesn't contain the id either)
                if isinstance(actor, dict):
                    actor_id = actor.get("id")
                    actor_name = actor.get("login")
                else:
                    actor_id = None
                    actor_name = actor

                # For some reason, the repo might be specified in either format ("repo" or "repository").
                # This even happens in newer records.
                repo = obj.get("repository") or obj.get("repo") or {}
                repo_name = repo.get("name")
                repo_owner = repo.get("owner")
                repo_owner_name = None
                repo_owner_id = None
                if repo_name == "/": # the actual repo is in the payload
                    repo_name = None
                    repo_fullname = payload.get("repo")
                    if repo_fullname is not None:
                        (repo_name, repo_owner_name) = tuple(repo_fullname.split("/"))
                elif repo_name is not None and repo_owner is None:
                    (repo_owner_name, repo_name) = tuple(repo_name.split("/"))
                elif isinstance(repo_owner, dict): # repo_owner may be just the name or a dict containing the user info
                    # TODO test in the whole dataset if this case ver actually occurs
                    repo_owner_name = repo_owner.get("login")
                    repo_owner_id = repo_owner.get("id")
                else:
                    repo_owner_name = repo_owner
                repo_id = repo.get("id")

                # db.insert_event(event_id, repo_id, repo_name, repo_owner_name, repo_owner_id, event_time, actor_id, actor_name)
                result.append((None, event_id, repo_id, repo_name, repo_owner_name, repo_owner_id, event_time, actor_id, actor_name))

            except Exception as e:
                elog("Error while processing line {} of {} (type {}):\n{}", lineno, path_to_file, obj.get("type"), e)
                raise(e)
        return result

def main():
    import argparse

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

    log("Json parsing started.")

    if args.label:
        log("This run is labeled: '{}'.",args.label)

    log("Application source code directory is: {}", app_src_dir)

    log("Application home directory is: {}", app_home_dir)

    app_data_indir = args.indir if args.indir else os.path.realpath(app_home_dir + DEFAULT_INPUT_DIRECTORY);
    log("Application input data directory is: {}", app_data_indir)

    filename_pattern = DEFAULT_INPUT_FILENAME_PATTERN
    filename_pattern = filename_pattern.replace("YYYY", args.year if args.year else "????")
    filename_pattern = filename_pattern.replace("MM", args.month if args.month else "??")
    filename_pattern = filename_pattern.replace("DD", args.day if args.day else "??")
    filename_pattern = filename_pattern.replace("HH", args.hour if args.hour else "??")
    filename_pattern = filename_pattern.replace("[-LABEL]", "-" + args.label if args.label else "")
    log("Input Data Filename Pattern: {}", filename_pattern)

    from utils.file_utils import discover_directory_files
    input_files = discover_directory_files(app_data_indir, filename_pattern=filename_pattern)
    file_count = len(input_files)
    if file_count > 0:
        log("Discovered {} Input Data Files, starting with {} and ending with {}", file_count, input_files[0], input_files[-1])
    else:
        log("Discovered no Input Data Files")

    app_data_outdir = args.outdir if args.outdir else os.path.realpath(app_home_dir + DEFAULT_OUTPUT_DIRECTORY);
    log("Application preprocessed dataset directory will be: {}", app_data_outdir)

    output_filename = DEFAULT_OUTPUT_FILENAME_PATTERN
    output_filename = output_filename.replace("[-LABEL]", "-" + args.label if args.label else "")
    output_filename_path = os.path.join(app_data_outdir, output_filename)
    log("Application preprocessed dataset filename will be: {}", output_filename_path)

    log("Using Number Of Threads: {}", args.threads)

    preprocess_files(input_files, int(args.threads), output_filename_path)

# Entry point for running the json parsing step separately
if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()
