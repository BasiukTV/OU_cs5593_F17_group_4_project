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
        event_id integer,
        repo_id integer,
        time text,
        actor_id integer
    '''
    cur.execute('''
        CREATE TABLE starrings (
            {}
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE publications (
            {}
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE creations (
            {},
            description_len integer,
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
            event_id integer,
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
            num_assets integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE pr_opens (
            {},
            pr_id integer,
            title text,
            body_len integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE issue_opens (
            {},
            issue_id number,
            title text,
            label_no number,
            milestone text,
            body_len number
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE pr_close (
            {},
            pr_id integer,
            merged bool
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE issue_close (
            {},
            issue_id integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE forks (
            {},
            forkee_id integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE deletes (
            {},
            type text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE issue_comments (
            {},
            issue_id integer,
            comment_id integer,
            body_len integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE commit_comments (
            {},
            comment_id integer,
            commit_id integer,
            body_len integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE member_events (
            {},
            member_id integer,
            event_type text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE wiki_events (
            {},
            event_type text
        )
    '''.format(common_attrs))
    cur.execute('''CREATE TABLE repos (
        repo_id integer PRIMARY KEY,
        number_of_stars integer
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

        # TODO either
        # - find a way to use sqlite with multiprocessing
        # - handle multiprocessing manually, keep multiple databases and merge later
        # - "officially" remove multithreading functinality
        # thread_pool.map(
        #         process_file, # execute process_file
        #         zip(
        #             files[starting_file_index : starting_file_index + files_to_process], # on these files
        #             [DB_PATH] * files_to_process) # with that database
        #         )

    aggregate_data(con)

    con.commit()
    con.close()

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
                        description = payload.get("description") # can be None
                        description_len = None if description is None else len(description)
                        cur.execute("INSERT INTO creations VALUES(?, ?, ?, ?, ?, ?, ?)", std + (
                            description_len,
                            payload.get("pusher_type")
                        ))
                    elif event_type == "PushEvent":
                        cur.execute("INSERT INTO pushes VALUES(?, ?, ?, ?, ?)", std)
                        commits = payload.get("commits")
                        if commits is None:
                            # old format
                            for commit in payload.get("shas"):
                                cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (event_id, commit[3], commit[2], None))
                        else:
                            # new format
                            for commit in payload.get("commits"):
                                cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (event_id, commit.get("author").get("name"), commit.get("message"), commit.get("distinct")))
                    elif event_type == "ReleaseEvent":
                        release = payload["release"]
                        cur.execute("INSERT INTO releases VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (release.get("tag_name"), release.get("name"), release.get("prerelease"), len(release.get("assets"))))
                    elif event_type == "PullRequestEvent":
                        pr = payload.get("pull_request")
                        a = payload.get("action")
                        # there are a lot of other actions, but they were all introduced after 2015:
                        # - assigned, unassigned
                        # - review_requested, review_request_removed
                        # - labeled, unlabeled
                        # - edited
                        # - reopened
                        if a == "opened":
                            body = pr.get("body") # can be None TODO Payload?
                            body_len = None if body is None else len(body)
                            cur.execute("INSERT INTO pr_opens VALUES(?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                pr.get("id"),
                                pr.get("title"),
                                body_len,
                            ))
                        elif a == "closed":
                            merged = pr.get("merged") # if false, pr was discarded. May be None in old events.
                            cur.execute("INSERT INTO pr_close VALUES(?, ?, ?, ?, ?, ?, ?)", std + (
                                pr.get("id"),
                                merged
                            ))
                    elif event_type == "IssuesEvent":
                        issue = payload.get("issue")
                        a = payload.get("action")
                        if a == "opened":
                            if isinstance(issue, dict):
                                # new format
                                body = issue.get("body")
                                body_len = None if body is None else len(body)
                                cur.execute("INSERT INTO issue_opens VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                    issue.get("id"),
                                    issue.get("title"),
                                    len(issue.get("labels")),
                                    (issue.get("milestone") or {}).get("id"),
                                    body_len
                                ))
                            else:
                                cur.execute("INSERT INTO issue_opens VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                    issue,
                                    None, # no title provided
                                    None, # no labels either
                                    None, # no milestones
                                    None, # no body
                                ))
                        elif a == "closed":
                            issue_id = issue.get("id") if isinstance(issue, dict) else issue
                            cur.execute("INSERT INTO issue_close VALUES(?, ?, ?, ?, ?, ?)", std + (
                                issue_id
                            ))
                    elif event_type == "CommitCommentEvent":
                        comment = payload.get("comment")
                        if comment is not None:
                            # new format
                            cur.execute("INSERT INTO commit_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                comment.get("id"),
                                commit.get("commit_id"),
                                len(comment.get("body")),
                            ))
                        else:
                            # old format
                            cur.execute("INSERT INTO commit_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                payload.get("comment_id"),
                                payload.get("commit"),
                                len(comment.get("body")),
                            ))
                    elif event_type == "DeleteEvent":
                        ref_type = payload.get("ref_type")
                        cur.execute("INSERT INTO deletes VALUES(?, ?, ?, ?, ?, ?)", std + (
                            ref_type,
                        ))
                    elif event_type == "ForkEvent":
                        # old vs new format
                        forkee = payload.get("forkee")
                        forkee_id = forkee.get("id") if isinstance(forkee, dict) else forkee
                        cur.execute("INSERT INTO forks VALUES(?, ?, ?, ?, ?, ?)", std + (
                            forkee_id,
                        ))
                    elif event_type == "GollumEvent":
                        # wiki modified
                        pages = payload.get("pages")
                        if pages is None:
                            # old format: just one page, directly in payload
                            cur.execute("INSERT INTO wiki_events VALUES(?, ?, ?, ?, ?, ?)", std + (
                                payload.get("action"),
                            ))
                        else:
                            # new format: multiple pages possible
                            for page in payload.get("pages"):
                                cur.execute("INSERT INTO wiki_events VALUES(?, ?, ?, ?, ?, ?)", std + (
                                    page.get("action"),
                                ))
                    elif event_type == "IssueCommentEvent":
                        issue = payload.get("issue")
                        comment = payload.get("comment")
                        cur.execute("INSERT INTO issue_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            issue.get("id"),
                            comment.get("commit_id"),
                            len(comment.get("body")),
                        ))
                    elif event_type == "MemberEvent":
                        member = payload.get("member")
                        cur.execute("INSERT INTO member_events VALUES(?, ?, ?, ?, ?, ?, ?)", std + (
                            # TODO in old format, member is the name
                            member.get("id") if isinstance(member, dict) else None,
                            payload.get("action"),
                        ))
                    elif event_type == "PublicEvent":
                        cur.execute("INSERT INTO publications VALUES(?, ?, ?, ?, ?)", std)
                    elif event_type == "PullRequestReviewCommentEvent":
                        pr = payload.get("pull_request")
                        comment = payload.get("comment")
                        cur.execute("INSERT INTO issue_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            pr.get("id"),
                            comment.get("commit_id"),
                            len(comment.get("body")),
                        ))
                except Exception as e: 
                    from sys import stderr
                    print("{}: Error while processing line {} of {} (type {}):\n{}".format(now(),lineno, path_to_file, obj.get("type"), e), file=stderr)
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
