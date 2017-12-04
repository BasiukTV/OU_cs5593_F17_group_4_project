/*
	This script is meant to generate data for avg_contributor_data_explorattion.R script.
	Open one of the SQLite preprocessed datasets with contributor table in it
	with your favorite SQLite browser, run below query and export result as a CSV file.
*/
SELECT
	contributor.contributorID as contributorID,
	avg(contributor.repos_started_count) as repos_started_count,
	avg(contributor.repos_forked_count) as repos_forked_count,
	avg(contributor.code_pushed_count) as code_pushed_count,
	avg(contributor.pull_request_created_count) as pull_request_created_count,
	avg(contributor.pull_request_reviewed_count) as pull_request_reviewed_count,
	avg(contributor.pull_request_resolved_count) as pull_request_resolved_count,
	avg(contributor.issue_created_count) as issue_created_count,
	avg(contributor.issue_resolved_count) as issue_resolved_count,
	avg(contributor.issue_commented_count) as issue_commented_count,
	avg(contributor.issue_other_activity_count) as issue_other_activity_count
FROM contributor
GROUP BY contributor.contributorID;