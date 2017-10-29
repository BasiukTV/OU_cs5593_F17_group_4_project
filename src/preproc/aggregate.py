import sqlite3
import io
import os
from utils.logging import log

# ignore errors (such as the tale we're trying to delete does not exist)
def tryexec(con, cmd):
    try:
        con.execute(cmd)
    except sqlite3.OperationalError:
        pass

def delete_indices(con):
    tryexec(con, "DROP INDEX star_repo");

def drop_tables(con):
    tryexec(con, "DROP TABLE repos");

# for testing purposes, make sure to remove all results of the last run
def reset_database(con):
    # delete_indices(con) # not during development; TODO make configurable
    drop_tables(con)

# create all the nessessary tables
def setup_db(con):
    tryexec(con, '''CREATE TABLE repos (
        repo_id integer PRIMARY KEY,
        number_of_stars integer
        )
    ''')
    log("creating index on starrings")
    # spend a few (~10) minutes on creating an index once instead of linearly searching for every repo
    tryexec(con, "CREATE INDEX star_repo on starrings(repo_id)");

# move from intermediate parsed database to the aggregated format
def aggregate_data(database_file):
    con = sqlite3.connect(database_file)
    log("resetting the database")
    reset_database(con);
    log("setting up the necessary tables")
    setup_db(con);

    log("aggregating data")
    cur1 = con.cursor()
    cur2 = con.cursor()
    log("aggregating stars")
    for row in cur1.execute("SELECT DISTINCT repo_id FROM starrings"):
        repoid = row[0]
        if repoid is None:
            # TODO don't use repoid, as it is unrelyable. Instead use owner/name.
            continue
        cur2.execute("SELECT count(*) FROM starrings WHERE repo_id = ?", (str(repoid),))
        stars = cur2.fetchone()[0]
        cur2.execute("INSERT INTO repos VALUES (?, ?)", (repoid, stars))
    con.commit()
    con.close()

DEFAULT_DB_FILE = "/input/data/preprocessed.sqlite3"; # Relative to home directory of the application

# Entry point for running the aggregation step separately
if __name__ == "__main__":
    import argparse, os, sys

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
