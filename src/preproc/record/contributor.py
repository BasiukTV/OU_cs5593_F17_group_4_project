import math

from preproc.record.record import Record

class Contributor(Record):
    """Class for preprocessed dataset contributor record."""

    def __init__(self,
            contributorID,
            timestamp,
            repos_started_count = 0,
            repos_forked_count = 0,
            code_pushed_count = 0,
            pull_request_created_count = 0,
            pull_request_reviewed_count = 0,
            issue_created_count = 0,
            issue_resolved_count = 0,
            issue_commented_count = 0,
            issue_other_activity_count = 0,
            owned_repos_starts_count = 0,
            reserve1 = 0,
            reserve2 = ""):

        Record.__init__(self, contributorID, timestamp)
        self.repos_started_count = repos_started_count
        self.repos_forked_count = repos_forked_count
        self.code_pushed_count = code_pushed_count
        self.pull_request_created_count = pull_request_created_count
        self.pull_request_reviewed_count = pull_request_reviewed_count
        self.issue_created_count = issue_created_count
        self.issue_resolved_count = issue_resolved_count
        self.issue_commented_count = issue_commented_count
        self.issue_other_activity_count = issue_other_activity_count
        self.owned_repos_starts_count = owned_repos_starts_count
        self.reserve1 = reserve1
        self.reserve2 = reserve2

    # Below overrides string representation of the Contributor object. For printing and debugging.
    def __str__(self):
        return "Contributor: {},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
            self.entityID,
            self.timestamp,
            self.repos_started_count,
            self.repos_forked_count,
            self.code_pushed_count,
            self.pull_request_created_count,
            self.pull_request_reviewed_count,
            self.issue_created_count,
            self.issue_resolved_count,
            self.issue_commented_count,
            self.issue_other_activity_count,
            self.owned_repos_starts_count,
            self.reserve1,
            self.reserve2)

    def eucld_dist(self, other, weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        """Returns weighted Euclidian distance between this and other Contributor record."""
        return math.sqrt(
            weights[0] * (self.repos_started_count - other.repos_started_count) ** 2 +
            weights[1] * (self.repos_forked_count - other.repos_forked_count) ** 2 +
            weights[2] * (self.code_pushed_count - other.code_pushed_count) ** 2 +
            weights[3] * (self.pull_request_created_count - other.pull_request_created_count) ** 2 +
            weights[4] * (self.pull_request_reviewed_count - other.pull_request_reviewed_count) ** 2 +
            weights[5] * (self.issue_created_count - other.issue_created_count) ** 2 +
            weights[6] * (self.issue_resolved_count - other.issue_resolved_count) ** 2 +
            weights[7] * (self.issue_commented_count - other.issue_commented_count) ** 2 +
            weights[8] * (self.issue_other_activity_count - other.issue_other_activity_count) ** 2 +
            weights[9] * (self.owned_repos_starts_count - other.owned_repos_starts_count) ** 2)
