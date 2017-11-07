import sqlite3
import io
import os
import argparse
import sys

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)
from utils.logging import log, elog

# ignore errors (such as the tale we're trying to delete does not exist)
def tryexec(con, cmd):
    try:
        con.execute(cmd)
    except sqlite3.OperationalError:
        pass

def delete_indices(con):
    tryexec(con, "DROP INDEX star_repo");

def drop_tables(con):
    tryexec(con, "DROP TABLE repos"); # TODO

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
        name text
        )''')
    log("creating index on starrings")
    # spend a few (~10) minutes on creating an index once instead of linearly searching for every repo
    tryexec(con, "CREATE INDEX star_repo on starrings(repo_id)");

def initialize_repo_table(con):
    con.execute('''INSERT INTO repo SELECT DISTINCT Null, repo_id, repo_name, repo_owner_name FROM repo_creations''')

def aggregate_starrings(con):
    for (owner, name) in con.execute("SELECT owner, name FROM repo"):
        stars = con.execute("SELECT count(*) FROM starrings WHERE repo_owner_name = ? AND repo_name = ?", (owner, name)).fetchone()[0]
        # TODO
        # print(owner, name)
        # con.execute("UPDATE repos SET stars = stars + ? WHERE repo_owner_name = ? AND repo_name = ?", (stars, owner, name))

# move from intermediate parsed database to the aggregated format
def aggregate_data(database_file):
    con = sqlite3.connect(database_file)
    log("resetting the database")
    reset_database(con);
    log("setting up the necessary tables")
    setup_db(con);

    log("aggregating data")
    log("initializing repositories table")
    initialize_repo_table(con)
    log("aggregating starrings")
    aggregate_starrings(con)
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
