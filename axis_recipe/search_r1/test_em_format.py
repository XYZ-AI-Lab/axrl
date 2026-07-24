import pytest

from axis_recipe.search_r1.qa_em_format import compute_score_em


def test_correct_answer_valid_format() -> None:
    solution = "<think>Compute</think><answer>42</answer>"
    ground_truth = {"target": ["42"]}

    assert compute_score_em(solution, ground_truth, structure_format_score=0.2) == 1.0


def test_wrong_answer_with_retrieval_hits_structure_score() -> None:
    solution = (
        "<think>Need info</think>"
        "<search>capital of france</search>"
        "<information>Paris is the capital</information>"
        "<think>Now answer</think>"
        "<answer>London</answer>"
    )
    ground_truth = {"target": ["Paris"]}

    score = compute_score_em(
        solution,
        ground_truth,
        structure_format_score=0.2,
        retrieval_score=0.1,
    )

    assert score == pytest.approx(0.3)


def test_correct_format_retrieval_and_answer() -> None:
    solution = (
        "<think>Need info</think>"
        "<search>capital of france</search>"
        "<information>Paris is the capital</information>"
        "<think>Now answer</think>"
        "<answer>Paris</answer>"
    )
    ground_truth = {"target": ["Paris"]}

    score = compute_score_em(
        solution,
        ground_truth,
        structure_format_score=0.2,
        retrieval_score=0.1,
    )

    assert score == pytest.approx(1.0)


def test_multiple_search_tags_no_answer_scores_zero() -> None:
    """Multiple <search> tags in one response without intervening <information> is invalid format."""
    solution = "<think>First thought</think><search>query one</search><think>Second thought</think><search>query two</search>"
    ground_truth = {"target": ["Paris"]}

    score = compute_score_em(
        solution,
        ground_truth,
        structure_format_score=0.2,
        retrieval_score=0.1,
    )

    # No answer tag + invalid format -> 0
    assert score == pytest.approx(0.0)


def test_multiple_search_tags_with_correct_answer_loses_format_score() -> None:
    """Multiple <search> without <information> is invalid even if the final answer is correct."""
    solution = "<think>First thought</think><search>query one</search><search>query two</search><answer>Paris</answer>"
    ground_truth = {"target": ["Paris"]}

    score = compute_score_em(
        solution,
        ground_truth,
        structure_format_score=0.2,
        retrieval_score=0.1,
    )

    # Correct answer but invalid format -> score - structure_format_score = 0.8
    assert score == pytest.approx(0.8)


def test_multiple_valid_searches_with_correct_answer_scores_full() -> None:
    """Multiple <search>/<information> rounds with valid format and correct answer -> score 1."""
    solution = (
        "<think>Let me search for the capital of France</think>"
        "<search>capital of France</search>"
        "<information>Paris is the capital of France</information>"
        "<think>Let me verify with another search</think>"
        "<search>Paris France capital</search>"
        "<information>Paris has been the capital of France since 987 AD</information>"
        "<think>I am confident now</think>"
        "<answer>Paris</answer>"
    )
    ground_truth = {"target": ["Paris"]}

    score = compute_score_em(
        solution,
        ground_truth,
        structure_format_score=0.2,
        retrieval_score=0.1,
    )

    assert score == pytest.approx(1.0)


def test_no_think() -> None:
    solution = "<answer>0</answer>"
    ground_truth = {"target": ["1"]}

    score = compute_score_em(
        solution,
        ground_truth,
        structure_format_score=0.2,
        final_format_score=0.1,
    )

    assert score == pytest.approx(0.1)
