# Copied and Adapted from Search-R1 repo:
# https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em_format.py

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
import string


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: str | list[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        normalized_answer = normalize_answer(golden_answer)
        if normalized_answer == normalized_prediction:
            return 1
    return 0


def is_valid_sequence(text: str) -> tuple[bool, str]:
    # Find the position of "<|im_start|>assistant" with potential whitespace
    start_pos = 0
    content = text[start_pos:]

    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)

    tag_transitions = {
        ("start", "<think>"): "in_think",
        ("information", "<think>"): "in_think",
        ("in_think", "</think>"): "after_think",
        ("after_think", "<search>"): "in_search",
        ("in_search", "</search>"): "after_search",
        ("after_search", "<information>"): "in_information",
        ("in_information", "</information>"): "information",
        ("after_think", "<answer>"): "in_answer",
        ("in_answer", "</answer>"): "end",
    }
    content_allowed_states = {"in_think", "in_search", "in_information", "in_answer"}
    whitespace_only_states = {"start", "after_think", "after_search", "information"}

    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end

    # 3. Check each part
    for part in parts:
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            next_state = tag_transitions.get((state, part))
            if next_state is None:
                return False, f"Unexpected tag {part} in state {state}"
            state = next_state
        elif state in content_allowed_states:
            continue
        elif state in whitespace_only_states:
            # Only whitespace is allowed between tags
            if part.strip():
                return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
        else:
            return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str: str) -> str | None:
    """Extract the equation from the solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> bool:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(
    solution_str: str,
    ground_truth: dict[str, list[str]],
    structure_format_score: float = 0,
    final_format_score: float = 0,
    retrieval_score: float = 0,
    score: float = 1.0,
) -> float:
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth containing a `target` list of correct answers
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        structure_format_score: the score for well-formed structure
        final_format_score: fallback score when the answer is wrong and format is invalid
        retrieval_score: the score when retrieval is correct
        score: the score for the correct answer
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth["target"])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 512) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if not is_valid_format:
            return 0
        if retrieval_correct:
            return structure_format_score + retrieval_score  # 0.3
        return structure_format_score  # 0.2

    if em_check(answer, ground_truth["target"]):
        if is_valid_format:
            return score  # 1
        return score - structure_format_score  # 0.8

    if not is_valid_format:
        return final_format_score  # 0.1

    if retrieval_correct:
        return structure_format_score + retrieval_score  # 0.3
    return structure_format_score  # 0.2
