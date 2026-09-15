"""Voting ensemble utilities."""


def majority_vote(predictions):
    """Return the majority vote for a list of predictions."""
    return predictions[0] if not predictions else max(set(predictions), key=predictions.count)
