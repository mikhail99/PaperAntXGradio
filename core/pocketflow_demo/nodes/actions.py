from enum import Enum
from typing import TypedDict

class Action(str, Enum):
    do_generate_queries = "do-generate-queries"
    do_literature_review = "do-literature-review"
    do_literature_review_gap = "do-literature-review-gap"
    do_write_proposal = "do-write-proposal"
    do_follow_up = "do-follow-up"
    do_result_notification = "do-result-notification"
    # Review actions for human-in-the-loop steps
    review_queries = "review-queries"
    review_report = "review-report"
