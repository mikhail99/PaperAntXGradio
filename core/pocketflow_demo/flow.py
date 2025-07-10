from pocketflow import Flow

from core.pocketflow_demo.nodes import (
    DecideAction,
    CheckWeather,
    BookHotel,
    FollowUp,
    ResultNotification,
)


def create_flow():
    """
    Create and connect the nodes to form a complete agent flow.
    """
    decide_action = DecideAction()
    generate_query = CheckWeather()
    generate_literature_review = BookHotel()
    follow_up = FollowUp()
    result_notification = ResultNotification()

    decide_action - "check-weather" >> generate_query
    generate_query >> decide_action
    decide_action - "book-hotel" >> generate_literature_review
    generate_literature_review >> decide_action
    decide_action - "follow-up" >> follow_up
    decide_action - "result-notification" >> result_notification

    return Flow(start=decide_action)
