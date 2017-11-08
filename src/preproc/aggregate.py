import sqlite3
import io
import os
import argparse
import sys
from datetime import datetime

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)
from utils.logging import log, elog

# ignore errors (such as the tale we're trying to delete does not exist)
def tryexec(con, cmd, *args):
    try:
        con.execute(cmd, *args)
    except sqlite3.OperationalError:
        pass

def delete_indices(con):
    tryexec(con, "DROP INDEX star_repo");

def drop_tables(con):
    tryexec(con, "DROP TABLE repo"); # TODO
    tryexec(con, "DROP TABLE repositories"); # TODO

# for testing purposes, make sure to remove all results of the last run
def reset_database(con):
    # delete_indices(con) # not during development; TODO make configurable
    drop_tables(con)

# create all the nessessary tables
def setup_db(con):
    tryexec(con, '''CREATE TABLE repo (
        id integer PRIMARY KEY AUTOINCREMENT,
        repoID integer,
        owner text,
        name text,
        creation_date
        )''')
    tryexec(con, """
        CREATE TABLE repository (
            repositoryID integer,
            timestamp integer,
            star_count integer,
            total_contributor_count integer,
            contributor_type1_count integer,
            contributor_type2_count integer,
            contributor_type3_count integer,
            contributor_type4_count integer,
            contributor_type5_count integer,
            code_push_count integer,
            pull_request_count integer,
            fork_count integer,
            release_count integer,
            active_issues_count integer,
            resolved_issue_count integer,
            org_activity_count integer,
            reserve1 integer,
            reserve2 text
        )""")
    tryexec(con, "CREATE INDEX repo_id on repository(repositoryID)")
    log("creating index on starrings")
    # spend a few (~10) minutes on creating an index once instead of linearly searching for every repo

def initialize_repo_table(con):
    con.execute('''INSERT INTO repo SELECT DISTINCT Null, repo_id, repo_name, repo_owner_name, time FROM repo_creations''')

def create_indices(con, tables):
    for table in tables:
        log("Creating indices on {}".format(table))
        tryexec(con, "CREATE INDEX {0}_fullname on {0}(repo_owner_name, repo_name)".format(table));
        tryexec(con, "CREATE INDEX {0}_time on {0}(time)".format(table));

def parse_isotime(isostr): # TODO util
    return datetime.strptime(isostr, "%Y-%m-%dT%H:%M:%S")

# finds the time of the last event recorded
def find_stoptime(con, tables):
    stop = datetime.fromtimestamp(0)
    for table in tables:
        last = parse_isotime(con.execute("SELECT max(time) FROM {}".format(table)).fetchone()[0])
        if last > stop:
            stop = last
    return stop

# create all the entries in the repository table
def aggregate_repository(con, stoptime):
    from datetime import timedelta
    offset = timedelta(days = 7)
    for (id, owner, name, creation_date) in con.execute("SELECT id, owner, name, creation_date FROM repo"):
        cur_week = datetime.strptime(creation_date, "%Y-%m-%dT%H:%M:%S")
        while cur_week < stoptime:
            next_week = cur_week + offset
            star_count = con.execute("""
                SELECT count(*)
                FROM starrings
                WHERE repo_owner_name = ?
                  AND repo_name = ?
                  AND time BETWEEN ? AND ?
            """, (owner, name, cur_week.isoformat(), next_week.isoformat())).fetchone()[0]
            con.execute("INSERT INTO repository VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)", (
                id,
                cur_week.isoformat(),
                star_count
            )) # TODO make clear that id is *not* githubs repo id
            cur_week = next_week
        # TODO

# move from intermediate parsed database to the aggregated format
def aggregate_data(database_file):
    timed_events = [
        'starrings',
        'publications',
        'repo_creations',
        'branch_creations',
        'tag_creations',
        'pushes',
        'releases',
        'pr_opens',
        'issue_opens',
        'pr_close',
        'issue_close',
        'forks',
        'deletes',
        'issue_comments',
        'commit_comments',
        'member_events',
        'wiki_events'
    ]

    con = sqlite3.connect(database_file)
    log("resetting the database")
    reset_database(con);
    log("setting up the necessary tables")
    setup_db(con);
    log("setting up the indices")
    create_indices(con, timed_events)
    log("finding the last event")
    stoptime = find_stoptime(con, timed_events)

    log("aggregating data")
    log("initializing repositories table")
    initialize_repo_table(con)
    log("aggregating repositories")
    aggregate_repository(con, stoptime)
    con.commit()
    con.close()

DEFAULT_DB_FILE = "/output/data/preproc/preprocessed.sqlite3"; # Relative to home directory of the application

# Entry point for running the aggregation step separately
if __name__ == "__main__":

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for parsing the intermediate GitHub event database into an aggregated format.")
    parser.add_argument("-f", "--file", help="Database file (will be updated in place)")
    args = parser.parse_args()

    # TODO abstract this part
    print("Aggregation started.")

    app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
    print("Application home directory is: {}".format(app_home_dir))

    # Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
    app_src_dir = os.path.realpath(app_home_dir + "/src")
    print("Application source code directory is: {}".format(app_src_dir))
    sys.path.insert(0, app_src_dir)

    database_file = args.file if args.file else os.path.realpath(app_home_dir + DEFAULT_DB_FILE);
    print("Database file is located at: {}".format(database_file))

    aggregate_data(database_file)
