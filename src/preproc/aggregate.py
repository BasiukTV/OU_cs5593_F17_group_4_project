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
    con.execute("DROP TABLE IF EXISTS OUT.repository");
    con.execute("DROP TABLE IF EXISTS main.user");
    con.execute("DROP TABLE IF EXISTS main.user_name");
    con.execute("DROP TABLE IF EXISTS OUT.contributor");

# for testing purposes, make sure to remove all results of the last run
def reset_database(con):
    delete_indices(con)
    drop_tables(con)

# create all the nessessary tables
def setup_db(con):
    # TODO use the functions in the database dir for this
    con.execute("""
        CREATE TABLE IF NOT EXISTS OUT.repository (
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
            pull_request_created_count integer,
            pull_request_reviewed_count integer,
            pull_request_resolved_count integer,
            fork_count integer,
            release_count integer,
            issue_created_count integer,
            issue_commented_count integer,
            issue_resolved_count integer,
            org_activity_count integer,
            reserve1 integer,
            reserve2 text,
            primary key (repositoryID, timestamp)
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS OUT.contributor (
            contributorID integer,
            timestamp integer,
            repos_started_count integer,
            repos_forked_count integer,
            code_pushed_count integer,
            pull_request_created_count integer,
            pull_request_reviewed_count integer,
            pull_request_resolved_count integer,
            issue_created_count integer,
            issue_resolved_count integer,
            issue_commented_count integer,
            issue_other_activity_count integer,
            owned_repos_stars_count integer,
            reserve1 integer,
            reserve2 text,
            primary key (contributorID, timestamp)
        )""")
    # TODO probably better to create this index after all the insertions, as creating it once should
    # be faster than updating it all the time
    # con.execute("CREATE INDEX IF NOT EXISTS repo_id on repository(repositoryID)")

def initialize_repo_table(con):
    # This is of course not 100% accurate, but a good approximation and saves a lot of time.
    if table_exists(con, 'OUT.repo'):
        log("Skipping repo table initialization, as it apparently is already initialized")
        return

    con.execute('''CREATE TABLE IF NOT EXISTS OUT.repo (
        id integer PRIMARY KEY AUTOINCREMENT,
        repoID integer,
        owner text,
        name text,
        creation_date,
        finished integer
        )''')
    con.execute('''CREATE TABLE IF NOT EXISTS OUT.repo_contributors (
        repo_id integer,
        contributor_id integer,
        primary key (repo_id, contributor_id)
        )''')
    con.commit()
    # con.execute('''INSERT INTO OUT.repo SELECT DISTINCT Null, repo_id, repo_name, repo_owner_name, time, 0 FROM repo_creations''')

def create_indices(con, tables):
    for table in tables:
        log("Creating indices on {}".format(table))
        con.execute("CREATE INDEX IF NOT EXISTS {0}_fullname on {0}(repo_owner_name, repo_name)".format(table));
        con.execute("CREATE INDEX IF NOT EXISTS {0}_actor_id on {0}(actor_id)".format(table));
        con.execute("CREATE INDEX IF NOT EXISTS {0}_actor_name on {0}(actor_name)".format(table));
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

def count_contributor_event(con, name_attr, time_start, time_end, aliases, event):
    count = 0
    aliases = aliases.copy() + [(None, time_end)]
    for i in range (0, len(aliases) - 1):
        alias = aliases[i][0]
        alias_start = max(aliases[i][1], time_start)
        alias_end = min(aliases[i + 1][1], time_end) # time when this alias will be outdated
        count += con.execute("""
            SELECT count(*)
            FROM {}
            WHERE {} = ?
              AND time >= ?
              AND time < ?
        """.format(event, name_attr), (alias, alias_start, alias_end)).fetchone()[0]
        if alias_end == time_end:
            break
    # TODO measure performance impact of between vs manual range, make range exclusve
    return count

# create all the entries in the contributor table
def aggregate_contributor(con, stoptime, offset):
    contributor_count = con.execute("SELECT count(*) FROM main.user WHERE finished = 0").fetchone()[0]
    finished_with = 0

    for (user_id, first_encounter) in con.execute("SELECT id, first_encounter FROM main.user WHERE finished = 0 ORDER BY RANDOM()"):
        aliases = []
        for alias in con.execute("SELECT name, first_encounter FROM main.user_name where id = ? ORDER BY first_encounter", (user_id,)):
            aliases = aliases + [ alias ]
        cur_week = parse_isotime(first_encounter)
        cur_week_iso = cur_week.isoformat()
        while cur_week < stoptime:
            next_week = cur_week + offset
            next_week_iso = next_week.isoformat()

            # remove all outdated aliases
            for _ in range(0, len(aliases) - 1):
                outdated_by = aliases[1][1]
                name = aliases[0][0]
                if outdated_by <= cur_week_iso:
                    aliases.pop(0)
                else:
                    break

            repos_forked_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "forks")
            code_pushed_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "pushes")
            pull_request_created_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "pr_opens")
            pull_request_reviewed_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "pr_review_comment")
            pull_request_resolved_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "pr_close")
            issue_created_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "issue_opens")
            issue_resolved_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "issue_close")
            issue_commented_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "issue_comments")
            # TODO analyze since when those are around
            issue_other_activity_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "issue_misc")
            owned_repos_stars_count = count_contributor_event(con, "repo_owner_name", cur_week_iso, next_week_iso, aliases, "starrings")
            repos_started_count = count_contributor_event(con, "actor_name", cur_week_iso, next_week_iso, aliases, "repo_creations")

            con.execute("INSERT INTO OUT.contributor VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)", (
                user_id,
                cur_week,
                repos_started_count,
                repos_forked_count,
                code_pushed_count,
                pull_request_created_count,
                pull_request_reviewed_count,
                pull_request_resolved_count,
                issue_created_count,
                issue_resolved_count,
                issue_commented_count,
                issue_other_activity_count,
                owned_repos_stars_count
            ))
            cur_week = next_week
            cur_week_iso = next_week_iso

        # status report
        con.execute("UPDATE main.user SET finished = 1 WHERE id = ?", (user_id, ))
        finished_with += 1
        if finished_with % 100 == 0:
            con.commit()
            elog("Finished with {}/{} contributors".format(finished_with, contributor_count))

def table_exists(con, table):
    return con.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0] == 1;

def initialize_user_table(con, tables):
    # This is of course not 100% accurate, but a good approximation and saves a lot of time.
    if table_exists(con, 'user'):
        log("Skipping user table initialization, as it apparently is already initialized")
        return

    con.execute('''CREATE TABLE IF NOT EXISTS main.user (
        id integer PRIMARY KEY,
        first_encounter text,
        finished integer,
        primary key (id, first_encounter)
        )''')
    con.execute('''CREATE TABLE IF NOT EXISTS main.user_name (
        id integer,
        name text,
        first_encounter text,
        primary key (id, name)
        )''')

    if not table_exists(con, 'all_actor_events'):
        log("Copying events into all_actor_events table")
        con.execute('''CREATE TABLE all_actor_events (
            actor_id int,
            actor_name int,
            time int,
            primary key (actor_id, time)
        )''')

        for table in tables:
            con.execute("INSERT INTO all_actor_events SELECT actor_id, actor_name, time FROM {}".format(table))
            log("Done copying over {}", table)
            con.commit()
        log("Done copying all events into one table")

    unknown_ids = {}

    # Put all actors into a single table, map their names to them
    for (actor_id, actor_name, first_encounter) in con.execute("SELECT actor_id, actor_name, min(time) FROM all_actor_events GROUP BY actor_id, actor_name"):
        if actor_id is not None:
            # if we already encountered the name before (just without an ID), use that as the first encounter date
            if unknown_ids.get(actor_name) is not None:
                first_encounter = unknown_ids.pop(actor_name)
                con.execute("UPDATE main.user SET first_encounter = min(first_encounter, ?) WHERE id = ?", (first_encounter, actor_id))

            con.execute("INSERT OR IGNORE INTO main.user VALUES (?, ?, 0)", (actor_id, first_encounter))
            con.execute("INSERT OR IGNORE INTO main.user_name VALUES (?, ?, ?)", (actor_id, actor_name, first_encounter))
        elif actor_name is not None:
            unknown_ids[actor_name] = first_encounter
            # con.execute("INSERT OR IGNORE INTO user_name VALUES (-1, ?, ?)", (actor_name, first_encounter)) # TODO analyze
            # 5618 don't appear elsewhere, 5712 do
            # given 1,069,793 total users
        else:
            elog("All none")

    # Actors that didn't do anything after the point where IDs where introduced.
    # I'm assuming renaming probably wasn't possible back then.
    # So just assign them fake IDs (negative).
    for (i, (actor_name, first_encounter)) in enumerate(unknown_ids.items(), 1):
        con.execute("INSERT INTO main.user VALUES (?, ?, 0)", (-i, first_encounter))
        con.execute("INSERT INTO main.user_name VALUES (?, ?, ?)", (-i, actor_name, first_encounter))


# create all the entries in the repository table
def aggregate_repository(con, stoptime, offset):
    repo_count = con.execute("SELECT count(*) FROM OUT.repo WHERE finished = 0").fetchone()[0]
    finished_with = 0

    for (id, owner, name, creation_date) in con.execute("SELECT id, owner, name, creation_date FROM OUT.repo WHERE finished = 0 ORDER BY RANDOM()"):
        cur_week = parse_isotime(creation_date)
        cur_week_iso = cur_week.isoformat()
        while cur_week < stoptime:
            next_week = cur_week + offset
            next_week_iso = next_week.isoformat()
            star_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "starrings")
            # TODO what is a contribution?
            # TODO duplicates
            con.execute("""
                INSERT INTO OUT.repo_contributors
                SELECT DISTINCT ?, actor_id
                FROM pushes
                WHERE repo_owner_name = ?
                  AND repo_name = ?
                  AND time >= ? AND time < ?
            """, (id, owner, name, cur_week_iso, next_week_iso))
            total_contributor_count = None
            contributor_type1_count = None
            contributor_type2_count = None
            contributor_type3_count = None
            contributor_type4_count = None
            contributor_type5_count = None
            code_push_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pushes") # TODO should this count individual commits instead?
            pull_request_created_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pr_opens")
            pull_request_reviewed_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pr_review_comment")
            pull_request_resolved_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pr_close")
            fork_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "forks")
            release_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "releases")
            issue_created_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_opens")
            issue_commented_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_comments")
            issue_resolved_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_close")
            org_activity_count = count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "issue_misc")
            + count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "pr_misc")
            + count_repo_event(con, cur_week_iso, next_week_iso, owner, name, "milestone_misc")

            con.execute("INSERT INTO OUT.repository VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)", (
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
                pull_request_created_count,
                pull_request_reviewed_count,
                pull_request_resolved_count,
                fork_count,
                release_count,
                issue_created_count,
                issue_commented_count,
                issue_resolved_count,
                org_activity_count
            )) # TODO make clear that id is *not* githubs repo id
            cur_week = next_week
            cur_week_iso = next_week_iso

        # status report
        con.execute("UPDATE OUT.repo SET finished = 1 WHERE id = ?", (id, ))
        finished_with += 1
        if finished_with % 100 == 0:
            con.commit()
            elog("Finished with {}/{} repos".format(finished_with, repo_count))

# move from intermediate parsed database to the aggregated format
def aggregate_data(database_in_file, database_out_file):
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

    con = sqlite3.connect(database_in_file)
    con.execute('ATTACH "{}" as OUT'.format(database_out_file))
    # Some queries needs a lot of temporary storage if the DB is big, which may exceed /tmp size
    con.execute("PRAGMA temp_store_directory = '{}'".format(os.path.dirname(database_in_file)))

    con.execute("PRAGMA journal_mode = WAL")
    log("setting up the necessary tables")
    setup_db(con);
    log("setting up the indices")
    create_indices(con, std_tables)
    log("finding the last event")
    stoptime = find_stoptime(con, std_tables)
    offset = timedelta(days = 7)

    log("aggregating data")
    # contributor_full(con, std_tables, stoptime, offset)
    repo_full(con, std_tables, stoptime, offset)
    con.close()

def repo_full(con, std_tables, stoptime, offset):
    log("initializing repositories table")
    initialize_repo_table(con)
    con.commit()
    log("aggregating repositories")
    aggregate_repository(con, stoptime, offset)
    con.commit()

def contributor_full(con, std_tables, stoptime, offset):
    log("initializing users table")
    initialize_user_table(con, std_tables)
    con.commit()
    log("aggregating contributors")
    aggregate_contributor(con, stoptime, offset)
    con.commit()

DEFAULT_IN_DB_FILE = "/output/data/preproc/preprocessed.sqlite3"; # Relative to home directory of the application
DEFAULT_OUT_DB_FILE = "/output/data/preproc/aggregated.sqlite3"; # Relative to home directory of the application

def main():
    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for parsing the intermediate GitHub event database into an aggregated format.")
    parser.add_argument("-i", "--infile", help="Database file with preprocessed info")
    parser.add_argument("-o", "--outfile", help="Database file to write to")
    args = parser.parse_args()

    # TODO abstract this part
    log("Aggregation started.")

    app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../..")
    log("Application home directory is: {}".format(app_home_dir))

    # Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
    app_src_dir = os.path.realpath(app_home_dir + "/src")
    log("Application source code directory is: {}".format(app_src_dir))
    sys.path.insert(0, app_src_dir)

    database_in_file = args.infile if args.infile else os.path.realpath(app_home_dir + DEFAULT_IN_DB_FILE);
    database_out_file = args.outfile if args.outfile else os.path.realpath(app_home_dir + DEFAULT_IN_DB_FILE);
    log("Database in file is located at: {}".format(database_in_file))

    aggregate_data(database_in_file, database_out_file)

# Entry point for running the aggregation step separately
if __name__ == "__main__":
    main()
