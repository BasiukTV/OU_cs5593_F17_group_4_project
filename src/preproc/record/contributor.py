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

    def __repr__(self):
        """Print override for containers."""
        return self.__str__()

    def eucld_dist(self, other, weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], self_cluster_size=1, other_cluster_size=1):
        """
            Returns weighted Euclidian distance between this and other Contributor record.
            If self_cluster_size and/or other_cluster_size are given, returns distance between cluster centorids
            multiplied by size of self and other cluster.
        """
        return self_cluster_size * other_cluster_size * math.sqrt((
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
                / sum(weights))

def calc_avg_contribution(contributors, resultID=0, resultTimestamp=0):
    """
        Returns a Contributor record which contains attributes whose values are average
        across given list of contributors. As averaging out contibutor ID and record timestamp
        dosen't make a lot of sense they're set to given values.
    """
    result = Contributor(
        contributorID=resultID,
        timestamp=resultTimestamp)

    num = len(contributors)

    # Accumulate overall contributions
    for c in contributors:
        result.repos_started_count += c.repos_started_count
        result.repos_forked_count += c.repos_forked_count
        result.code_pushed_count += c.code_pushed_count
        result.pull_request_created_count += c.pull_request_created_count
        result.pull_request_reviewed_count += c.pull_request_reviewed_count
        result.issue_created_count += c.issue_created_count
        result.issue_resolved_count += c.issue_resolved_count
        result.issue_commented_count += c.issue_commented_count
        result.issue_other_activity_count += c.issue_other_activity_count
        result.owned_repos_starts_count += c.owned_repos_starts_count

    #Average out the contibutions
    result.repos_started_count /= num
    result.repos_forked_count /= num
    result.code_pushed_count /= num
    result.pull_request_created_count /= num
    result.pull_request_reviewed_count /= num
    result.issue_created_count /= num
    result.issue_resolved_count /= num
    result.issue_commented_count /= num
    result.issue_other_activity_count /= num
    result.owned_repos_starts_count /= num

    return result
