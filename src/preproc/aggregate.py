import sqlite3
import io
import os
from time import gmtime, strftime

# TODO put in utils module
def now():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def log(string, *args):
    print("{}: {}".format(now(), string).format(args))

# ignore errors (such as the tale we're trying to delete does not exist)
def tryexec(con, cmd):
    try:
        con.execute(cmd)
    except sqlite3.OperationalError:
        pass

# for testing purposes, make sure to remove all results of the last run
def reset_database(con):
    tryexec(con, "DROP INDEX star_repo");
    tryexec(con, "DROP TABLE repos");

# create all the nessessary tables
def setup_db(con):
    con.execute('''CREATE TABLE repos (
        repo_id integer PRIMARY KEY,
        number_of_stars integer
        )
    ''')
    print("{}: creating index on starrings".format(now()))
    # spend a few (~10) minutes on creating an index once instead of linearly searching for every repo
    con.execute("CREATE INDEX star_repo on starrings(repo_id)");

# move from intermediate parsed database to the aggregated format
def aggregate_data(database_file):
    con = sqlite3.connect(database_file)
    print("{}: resetting the database".format(now()))
    reset_database(con);
    print("{}: setting up the necessary tables".format(now()))
    setup_db(con);

    print("{}: aggregating data".format(now()))
    cur1 = con.cursor()
    cur2 = con.cursor()
    print("{}: aggregating stars".format(now()))
    i = 0
    for row in cur1.execute("SELECT DISTINCT repo_id FROM starrings"):
        repoid = row[0]
        if repoid is None:
            # TODO
            print("Skipping {}".format(i))
            continue
        cur2.execute("SELECT count(*) FROM starrings WHERE repo_id = ?", (str(repoid),))
        stars = cur2.fetchone()[0]
        cur2.execute("INSERT INTO repos VALUES (?, ?)", (repoid, stars))
        i += 1
        if (i % 100 == 0):
            print("{}: done with {} repos".format(now(), i))

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
