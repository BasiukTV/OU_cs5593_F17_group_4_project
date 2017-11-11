from __future__ import print_function
import json
import multiprocessing
import sqlite3
import io
import os
import sys
from time import gmtime, strftime
from datetime import datetime

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from utils.logging import log, elog
from utils.parse import parse_isotime

DEFAULT_INPUT_DIRECTORY = "/input/data/raw"; # Relative to home directory of the application
DEFAULT_INPUT_FILENAME_PATTERN = "YYYY-MM-DD-HH[-LABEL].json";

DEFAULT_OUTPUT_DIRECTORY = '/output/data/preproc'; # Relative to home directory of the application
DEFAULT_OUTPUT_FILENAME_PATTERN = "preprocessed[-LABEL].sqlite3";

DEFAULT_THREADS_NUMBER = 5;

# Process files in one week of worth data batches. Assuming one file holds an hour worth of data.
DEFAULT_FILE_PREPROCESSING_BATCH_SIZE = 7 * 24;

def setup_db_scheme(cur):
    log("setting up db scheme")
    # TODO: instead of repeating the same attributes again and again, we could create a sperate "event" table.
    # I'm not sure aobut the performance implications though: Basically every action would require a join.
    common_attrs = '''
        id integer primary key autoincrement,
        event_id integer,
        repo_id integer,
        repo_name text,
        repo_owner_name text,
        repo_owner_id integer,
        time text,
        actor_id integer,
        actor_name integer
    '''
    cur.execute('''
        CREATE TABLE starrings (
            {}
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE milestone_misc (
            {},
            milestone_id integer,
            title_len integer,
            description_len integer,
            creator_id integer,
            action text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE publications (
            {}
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE repo_creations (
            {},
            description_len integer,
            pusher_type text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE branch_creations (
            {},
            description_len integer,
            pusher_type text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE tag_creations (
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
        CREATE TABLE pr_misc (
            {},
            pr_id integer,
            event text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE issue_close (
            {},
            issue_id integer
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE issue_misc (
            {},
            issue_id integer,
            event text
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
        CREATE TABLE pr_review (
            {},
            pr_id integer,
            review_id integer,
            body_len integer,
            action text
        )
    '''.format(common_attrs))
    cur.execute('''
        CREATE TABLE pr_review_comment (
            {},
            pr_id integer,
            review_id integer,
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

    con = sqlite3.connect(output_file_path)
    cur = con.cursor()
    setup_db_scheme(cur);

    # Determine size of a batch and number of batch tuns required to process all files
    batch_size = DEFAULT_FILE_PREPROCESSING_BATCH_SIZE
    batch_runs = 1 + (len(files) // batch_size)
    log("Preprocessing will take {} run(s), each processing a batch of up to {} file(s).", batch_runs, batch_size)

    # Initialize the thread pool
    thread_pool = multiprocessing.Pool(threads_num)

    # Loop which processes separate batches of input files
    for batch_run in range(1, batch_runs + 1):
        log("Starting run #{}", batch_run)

        starting_file_index = batch_size * (batch_run - 1)
        files_to_process = batch_size if batch_run != batch_runs else len(files) - starting_file_index
        log("This run will process {} files.", files_to_process)

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


    con.commit()
    con.close()

def process_file(path_to_file_and_database):
    (path_to_file, path_to_database) = path_to_file_and_database
    db = sqlite3.connect(path_to_database)
    # db = sqlite3.connect(':memory:') # TODO
    cur = db.cursor()
    log("Processing {}", path_to_file)

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
                    offset = datetime.strptime(event_time[-6:-3] + event_time[-2:], "%z")
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

                std = (None, event_id, repo_id, repo_name, repo_owner_name, repo_owner_id, event_time, actor_id, actor_name) # relevant attributes every event has

                if event_type == "WatchEvent":
                    cur.execute("INSERT INTO starrings VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", std)
                elif event_type == "CreateEvent":
                    # TODO seperate different create events
                    description = payload.get("description") # can be None
                    description_len = None if description is None else len(description)
                    t = payload.get("ref_type") or payload.get("object") # object is old format
                    if t == "branch":
                        cur.execute("INSERT INTO branch_creations VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            description_len,
                            payload.get("pusher_type")
                        ))
                    elif t == "tag":
                        cur.execute("INSERT INTO tag_creations VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            description_len,
                            payload.get("pusher_type")
                        ))
                        # TODO
                    elif t == "repository":
                        cur.execute("INSERT INTO repo_creations VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            description_len,
                            payload.get("pusher_type")
                        ))
                    else:
                        elog("Malformed CreateEvent: {}", json.dumps(obj))
                elif event_type == "PushEvent":
                    cur.execute("INSERT INTO pushes VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", std)
                    commits = payload.get("commits")
                    if commits is None:
                        # old format
                        for commit in payload.get("shas"):
                            cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (
                                event_id,
                                commit[3] if len(commit) >= 4 else None, # some commits don't have a name
                                commit[2] if len(commit) >= 3 else None, # I encountered a commit without a message
                                None
                            ))
                    else:
                        # new format
                        for commit in payload.get("commits"):
                            cur.execute("INSERT INTO commits VALUES(?, ?, ?, ?)", (event_id, commit.get("author").get("name"), commit.get("message"), commit.get("distinct")))
                elif event_type == "ReleaseEvent":
                    release = payload["release"]
                    cur.execute("INSERT INTO releases VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (release.get("tag_name"), release.get("name"), release.get("prerelease"), len(release.get("assets"))))
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
                        cur.execute("INSERT INTO pr_opens VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            pr.get("id"),
                            pr.get("title"),
                            body_len,
                        ))
                    elif a == "closed":
                        merged = pr.get("merged") # if false, pr was discarded. May be None in old events.
                        cur.execute("INSERT INTO pr_close VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            pr.get("id"),
                            merged
                        ))
                    else:
                        cur.execute("INSERT INTO pr_misc VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            pr.get("id"),
                            a
                        ))
                # TODO decide wether to use this and add more details if yes
                elif event_type == "MilestoneEvent":
                    milestone = payload.get("milestone")
                    milestone_id = milestone.get("id")
                    title_len = len(milestone.get("title"))
                    description_len = len(milestone.get("description", ""))
                    creator_id = len(milestone.get("creator").get("id"))
                    action = payload.get("action")
                    cur.execute("INSERT INTO milestone_misc VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                        milestone_id,
                        title_len,
                        description_len,
                        creator_id,
                        action
                    ))
                elif event_type == "IssuesEvent":
                    issue = payload.get("issue")
                    a = payload.get("action")
                    if isinstance(issue, dict):
                        issue_id = issue.get("id") if isinstance(issue, dict) else issue
                        body = issue.get("body")
                        body_len = None if body is None else len(body)
                        title = issue.get("title")
                        label_len = len(issue.get("labels"))
                        milestone_id = (issue.get("milestone") or {}).get("id")
                    else:
                        issue_id = issue
                        body_len = None
                        title = None
                        label_len = None
                        milestone_id = None
                    if a == "opened":
                        if isinstance(issue, dict):
                            cur.execute("INSERT INTO issue_opens VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                issue_id,
                                title,
                                label_len,
                                milestone_id,
                                body_len
                            ))
                        else:
                            cur.execute("INSERT INTO issue_opens VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                issue_id,
                                title,
                                label_len,
                                milestone_id,
                                body_len
                            ))
                    elif a == "closed":
                        cur.execute("INSERT INTO issue_close VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            issue_id,
                        ))
                    else:
                        cur.execute("INSERT INTO issue_misc VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            issue_id,
                            a
                        ))
                elif event_type == "CommitCommentEvent":
                    comment = payload.get("comment")
                    if comment is not None:
                        # new format
                        cur.execute("INSERT INTO commit_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            comment.get("id"),
                            comment.get("commit_id"),
                            len(comment.get("body")),
                        ))
                    else:
                        # old format
                        cur.execute("INSERT INTO commit_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            payload.get("comment_id"),
                            payload.get("commit"),
                            None, # no body provided
                        ))
                elif event_type == "DeleteEvent":
                    ref_type = payload.get("ref_type")
                    cur.execute("INSERT INTO deletes VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                        ref_type,
                    ))
                elif event_type == "ForkEvent":
                    # old vs new format
                    forkee = payload.get("forkee")
                    forkee_id = forkee.get("id") if isinstance(forkee, dict) else forkee
                    cur.execute("INSERT INTO forks VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                        forkee_id,
                    ))
                elif event_type == "GollumEvent":
                    # wiki modified
                    pages = payload.get("pages")
                    if pages is None:
                        # old format: just one page, directly in payload
                        cur.execute("INSERT INTO wiki_events VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            payload.get("action"),
                        ))
                    else:
                        # new format: multiple pages possible
                        for page in payload.get("pages"):
                            cur.execute("INSERT INTO wiki_events VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                                page.get("action"),
                            ))
                elif event_type == "IssueCommentEvent":
                    issue = payload.get("issue")
                    if isinstance(issue, dict):
                        # new format
                        comment = payload.get("comment")
                        cur.execute("INSERT INTO issue_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            issue.get("id"),
                            comment.get("commit_id"),
                            len(comment.get("body")),
                        ))
                    else:
                        # old format
                        cur.execute("INSERT INTO issue_comments VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                            payload.get("issue_id"),
                            payload.get("comment_id"),
                            None, # no body provided
                        ))
                elif event_type == "MemberEvent":
                    member = payload.get("member")
                    cur.execute("INSERT INTO member_events VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                        # TODO in old format, member is the name
                        member.get("id") if isinstance(member, dict) else None,
                        payload.get("action"),
                    ))
                elif event_type == "PublicEvent":
                    cur.execute("INSERT INTO publications VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", std)
                # TODO this event apparently never occurs?
                # Verify this on the whole dataset.
                elif event_type == "PullRequestReviewEvent":
                    pr = payload.get("pull_request")
                    review = payload.get("review")
                    cur.execute("INSERT INTO pr_review VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                        # TODO there is no pr id in the old format, but you can get the pr number indirectly
                        # through the pull request link (["_links"]["pull_request""])
                        # That can then lead to the pr id
                        None if pr is None else pr.get("id"),
                        review.get("id"),
                        len(review.get("body")),
                        payload.get("action")
                    ))
                elif event_type == "PullRequestReviewCommentEvent":
                    pr = payload.get("pull_request")
                    comment = payload.get("comment")
                    cur.execute("INSERT INTO pr_review_comment VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", std + (
                        # TODO there is no pr id in the old format, but you can get the pr number indirectly
                        # through the pull request link (["_links"]["pull_request""])
                        # That can then lead to the pr id
                        None if pr is None else pr.get("id"),
                        comment.get("id"),
                        len(comment.get("body")),
                    ))
            except Exception as e:
                elog("Error while processing line {} of {} (type {}):\n{}", lineno, path_to_file, obj.get("type"), e)
                raise(e)

    db.commit()
    db.close()

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
    import cProfile
    cProfile.run('main()')
