#!/usr/bin/env python3

# TODO use TagEvents?
# TODO performance analysis
# TODO test performance impact of generic event table
import sqlite3
import io
import os
import argparse
import sys
from datetime import datetime, timedelta

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)
from utils.logging import log, elog
from utils.parse import parse_isotime

def delete_indices(con):
    con.execute("DROP INDEX IF EXISTS star_repo");

def drop_tables(con):
    con.execute("DROP TABLE IF EXISTS repo");
    con.execute("DROP TABLE IF EXISTS repository");
    con.execute("DROP TABLE IF EXISTS user");
    con.execute("DROP TABLE IF EXISTS user_name");
    con.execute("DROP TABLE IF EXISTS contributor");

# for testing purposes, make sure to remove all results of the last run
def reset_database(con):
    # delete_indices(con) # not during development; TODO make configurable
    drop_tables(con)

# create all the nessessary tables
def setup_db(con):
    con.execute('''CREATE TABLE IF NOT EXISTS repo (
        id integer PRIMARY KEY AUTOINCREMENT,
        repoID integer,
        owner text,
        name text,
        creation_date
        )''')
    con.execute('''CREATE TABLE IF NOT EXISTS user (
        id integer PRIMARY KEY,
        first_encounter text
        )''')
    con.execute('''CREATE TABLE IF NOT EXISTS user_name (
        id integer,
        name text,
        first_encounter text,
        primary key (id, name)
        )''')
    # TODO use the functions in the database dir for this
    con.execute("""
        CREATE TABLE IF NOT EXISTS repository (
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
    con.execute("""
        CREATE TABLE IF NOT EXISTS contributor (
            contributorID integer,
            timestamp integer,
            repos_started_count integer,
            repos_forked_count integer,
            code_pushed_count integer,
            pull_request_created_count integer,
            pull_request_reviewed_count integer,
            issue_created_count integer,
            issue_resolved_count integer,
            issue_commented_count integer,
            issue_other_activity_count integer,
            owned_repos_starts_count integer,
            reserve1 integer,
            reserve2 text
        )""")
    # TODO probably better to create this index after all the insertions, as creating it once should
    # be faster than updating it all the time
    # con.execute("CREATE INDEX IF NOT EXISTS repo_id on repository(repositoryID)")

def initialize_repo_table(con):
    con.execute('''INSERT INTO repo SELECT DISTINCT Null, repo_id, repo_name, repo_owner_name, time FROM repo_creations''')

def create_indices(con, tables):
    for table in tables:
        log("Creating indices on {}".format(table))
        con.execute("CREATE INDEX IF NOT EXISTS {0}_fullname on {0}(repo_owner_name, repo_name)".format(table));
        con.execute("CREATE INDEX IF NOT EXISTS {0}_actor on {0}(actor_id)".format(table));
        con.execute("CREATE INDEX IF NOT EXISTS {0}_time on {0}(time)".format(table));

# finds the time of the last event recorded
def find_stoptime(con, tables):
    stop = datetime.fromtimestamp(0)
    for table in tables:
        last = con.execute("SELECT max(time) FROM {}".format(table)).fetchone()[0]
        if last is not None:
            last = parse_isotime(last)
            if last > stop:
                stop = last
    return stop

def count_repo_event(con, time_start, time_end, owner, name, event):
    return con.execute("""
        SELECT count(*)
        FROM {}
        WHERE repo_owner_name = ?
          AND repo_name = ?
          AND time >= ? AND time < ?
    """.format(event), (owner, name, time_start, time_end)).fetchone()[0]

def count_contributor_event(con, time_start, time_end, actor_id, event):
    return con.execute("""
        SELECT count(*)
        FROM {}
        WHERE actor_id = ?
          AND time between ? AND ?
    """.format(event), (actor_id, time_start, time_end)).fetchone()[0]
    # TODO measure performance impact of between vs manual range, make range exclusve

# create all the entries in the contributor table
def aggregate_contributor(con, stoptime, offset):
    log("Before count")
    contributor_count = con.execute("SELECT count(*) FROM user").fetchone()[0]
    log("After count")
    finished_with = 0

    for (user_id, first_encounter) in con.execute("SELECT id, first_encounter FROM user"):
        # TODO consider different aliases at different times (instead of only id)
        # TODO evaluate for which times a Null in the actor_id field actually occurs in relation with when renaming was introduced
        cur_week = parse_isotime(first_encounter)
        cur_week_iso = cur_week.isoformat()
        while cur_week < stoptime:
            next_week = cur_week + offset
            next_week_iso = next_week.isoformat()

            repos_forked_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "forks")
            code_pushed_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "pushes")
            pull_request_created_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "pr_opens")
            pull_request_reviewed_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "pr_review_comment")
            issue_created_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "issue_opens")
            issue_resolved_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "issue_close")
            issue_commented_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "issue_comments")
            # TODO analyze since when those are around
            issue_other_activity_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "issue_misc")
            owned_repos_starts_count = count_contributor_event(con, cur_week_iso, next_week_iso, user_id, "repo_creations")
            repos_started_count = owned_repos_starts_count + repos_forked_count # TODO is this right?

            con.execute("INSERT INTO contributor VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)", (
                user_id,
                cur_week,
                repos_started_count,
                repos_forked_count,
                code_pushed_count,
                pull_request_created_count,
                pull_request_reviewed_count,
                issue_created_count,
                issue_resolved_count,
                issue_commented_count,
                issue_other_activity_count,
                owned_repos_starts_count
            ))
            cur_week = next_week
            cur_week_iso = next_week_iso

        # status report
        finished_with += 1
        if finished_with % 1000 == 0:
            elog("Finished with {}/{} contributors".format(finished_with, contributor_count))

def initialize_user_table(con, tables):
    sources = 'SELECT id, actor_id, actor_name, time FROM {}'.format(tables[0])
    # All actors ever encountered
    for table in tables[1:]:
        sources = sources + """
        UNION
        SELECT id, actor_id, actor_name, time FROM {}
        """.format(table)

    # Put all actors into a single table, map their names to them
    for (actor_id, actor_name, first_encounter) in con.execute("SELECT actor_id, actor_name, min(time) FROM ({}) GROUP BY actor_id, actor_name".format(sources)):
        if actor_id is not None:
            con.execute("INSERT OR IGNORE INTO user VALUES (?, ?)", (actor_id, first_encounter))
            con.execute("INSERT OR IGNORE INTO user_name VALUES (?, ?, ?)", (actor_id, actor_name, first_encounter))
        elif actor_name is not None:
            con.execute("INSERT OR IGNORE INTO user_name VALUES (-1, ?, ?)", (actor_name, first_encounter)) # TODO analyze
            # 5618 don't appear elsewhere, 5712 do
            # given 1,069,793 total users
        else:
            elog("All none")

# create all the entries in the repository table
def aggregate_repository(con, stoptime, offset):
    log("Before count")
    repo_count = con.execute("SELECT count(*) FORM repo").fetchone()[0]
    log("After count")
    finished_with = 0

    for (id, owner, name, creation_date) in con.execute("SELECT id, owner, name, creation_date FROM repo"):
        cur_week = parse_isotime(creation_date)
        cur_week_iso = cur_week.isoformat()
        while cur_week < stoptime:
            next_week = cur_week + offset
            next_week_iso = next_week.isoformat()
            star_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "starrings")
            # TODO what is a contribution?
            # TODO don't only consider id, but also names
            total_contributor_count = con.execute("""
                SELECT count(actor_id)
                FROM pushes
                WHERE repo_owner_name = ?
                  AND repo_name = ?
                  AND time >= ? AND time < ?
            """, (owner, name, cur_week_iso, next_week_iso)).fetchone()[0]
            # TODO outsource the classification
            contributor_type1_count = None
            contributor_type2_count = None
            contributor_type3_count = None
            contributor_type4_count = None
            contributor_type5_count = None
            code_push_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pushes") # TODO should this count individual commits instead?
            pull_request_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pr_opens")
            fork_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "forks")
            release_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "releases")
            all_issues_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_opens")
            resolved_issue_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_close")
            active_issues_count = all_issues_count - resolved_issue_count
            org_activity_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_misc")
            + count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pr_misc")
            + count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "milestone_misc")

            con.execute("INSERT INTO repository VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)", (
                id,
                cur_week_iso,
                star_count,
                total_contributor_count,
                contributor_type1_count,
                contributor_type2_count,
                contributor_type3_count,
                contributor_type4_count,
                contributor_type5_count,
                code_push_count,
                pull_request_count,
                fork_count,
                release_count,
                active_issues_count,
                resolved_issue_count,
                org_activity_count
            )) # TODO make clear that id is *not* githubs repo id
            cur_week = next_week
            cur_week_iso = next_week_iso

        # status report
        finished_with += 1
        if finished_with % 1000 == 0:
            elog("Finished with {}/{} repos".format(finished_with, repo_count))

# move from intermediate parsed database to the aggregated format
def aggregate_data(database_file):
    std_tables = [
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
        'wiki_events',
        'issue_misc',
        'pr_misc',
        'milestone_misc',
        'pr_review_comment',
    ]

    con = sqlite3.connect(database_file)
    log("resetting the database")
    reset_database(con);
    log("setting up the necessary tables")
    setup_db(con);
    log("setting up the indices")
    create_indices(con, std_tables)
    log("finding the last event")
    stoptime = find_stoptime(con, std_tables)
    offset = timedelta(days = 7)

    log("aggregating data")
    log("initializing users table")
    initialize_user_table(con, std_tables)
    log("aggregating contributors")
    aggregate_contributor(con, stoptime, offset)
    log("initializing repositories table")
    initialize_repo_table(con)
    log("aggregating repositories")
    aggregate_repository(con, stoptime, offset)
    con.commit()
    con.close()

DEFAULT_DB_FILE = "/output/data/preproc/preprocessed.sqlite3"; # Relative to home directory of the application

def main():
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

# Entry point for running the aggregation step separately
if __name__ == "__main__":
    import cProfile
    cProfile.run('main()')
