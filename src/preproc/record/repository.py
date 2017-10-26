from preproc.record.record import Record

class Repository(Record):
    """Class for preprocessed dataset repository record."""

    def __init__(self,
            repositoryID,
            timestamp,
            star_count = 0,
            total_contributor_count = 0,
            contributor_type1_count = 0,
            contributor_type2_count = 0,
            contributor_type3_count = 0,
            contributor_type4_count = 0,
            contributor_type5_count = 0,
            code_push_count = 0,
            pull_request_count = 0,
            fork_count = 0,
            release_count = 0,
            active_issues_count = 0,
            resolved_issue_count = 0,
            org_activity_count = 0,
            reserve1 = 0,
            reserve2 = ""):

        Record.__init__(self, repositoryID, timestamp)
        self.star_count = star_count
        self.total_contributor_count = total_contributor_count
        self.contributor_type1_count = contributor_type1_count
        self.contributor_type2_count = contributor_type2_count
        self.contributor_type3_count = contributor_type3_count
        self.contributor_type4_count = contributor_type4_count
        self.contributor_type5_count = contributor_type5_count
        self.code_push_count = code_push_count
        self.pull_request_count = pull_request_count
        self.fork_count = fork_count
        self.release_count = release_count
        self.active_issues_count = active_issues_count
        self.resolved_issue_count = resolved_issue_count
        self.org_activity_count = org_activity_count
        self.reserve1 = reserve1
        self.reserve2 = reserve2
