from __future__ import annotations

import re


def _multiple_choice_query(payload: dict, query_profile: dict | None) -> bool:
    if query_profile and query_profile.get("multiple_choice"):
        return True
    labels = re.findall(
        r"(?:^|\s)([A-H])\.\s+",
        str(payload.get("query", "")),
        flags=re.IGNORECASE,
    )
    return len(set(label.upper() for label in labels)) >= 2


def longmem_answer_rules(
    payload: dict,
    *,
    query_profile: dict | None = None,
) -> str:
    """Return protocol-derived answer rules without using reference answers."""
    metadata = payload.get("metadata", {})
    eval_function = str(metadata.get("eval_function", ""))
    question_type = str(metadata.get("memory_category", ""))
    rules = [
        "Copy exact visible UI labels from evidence; do not paraphrase them.",
        "Return only the requested answer value in the answer field.",
    ]
    if eval_function.startswith("norm_phrase_set_match_ordered"):
        separator = ";" if "separators=;" in eval_function else ","
        rules.extend([
            "Preserve the source order exactly.",
            f"Separate answer phrases with '{separator}' and add no prose.",
        ])
    elif eval_function.startswith("norm_phrase_set_match"):
        separator = ";" if "separators=;" in eval_function else ","
        rules.extend([
            "Return all and only the qualifying phrases.",
            f"Separate answer phrases with '{separator}' and add no prose.",
        ])
    elif eval_function.startswith("mc_choice_set_match"):
        rules.append(
            "Return only the selected uppercase option letters separated by commas."
        )
    elif eval_function.startswith("mc_choice_match"):
        rules.append("Return exactly one uppercase option letter.")
    elif _multiple_choice_query(payload, query_profile):
        rules.extend([
            "This is a multiple-choice question; return only the selected "
            "uppercase option letter or letters requested by the question.",
            "Evaluate each option as a conjunction: every named item and property "
            "in the chosen option must be supported.",
            "An exact visible label is required for each named UI field or control; "
            "do not substitute a semantically related label for a missing one.",
        ])
    elif eval_function.startswith("llm_abstention_checker"):
        rules.extend([
            "Evaluate the premise before answering.",
            "Reject it only when cited evidence directly contradicts it.",
            "If contradiction is not established, answer UNKNOWN rather than guessing.",
        ])
    if question_type.endswith("-abs"):
        rules.extend([
            "This query requires premise verification.",
            "Set premise_status to contradicted only with direct cited counter-evidence; "
            "otherwise set it to unknown.",
        ])
    return " ".join(rules)


def longmem_reader_schema(
    payload: dict,
    *,
    evidence_ids: bool = False,
    query_profile: dict | None = None,
) -> str:
    """Describe a stable JSON schema shared by all LongMemEval methods."""
    metadata = payload.get("metadata", {})
    question_type = str(metadata.get("memory_category", ""))
    premise_field = (
        ', "premise_status":"supported|contradicted|unknown"'
        if question_type.endswith("-abs")
        else ""
    )
    evidence_field = (
        '"evidence_ids":["at most three exact visible evidence IDs"]'
        if evidence_ids
        else '"evidence":["at most three short supporting excerpts"]'
    )
    return (
        'Return JSON only: {"answer":"short final answer only", '
        f"{evidence_field}, "
        '"stage":"memory_retrieval", "dependencies":[]'
        f"{premise_field}" + "}. "
        + longmem_answer_rules(payload, query_profile=query_profile)
    )
