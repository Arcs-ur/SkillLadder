"""Prepare and score a human audit of the ER entity detector.

The experiment uses two evidence layers:

1. Structured silver labels:
   High-precision entity values are recovered from schema-like key/value fields
   in tau2 session traces (for example user_id, email, phone_number, address,
   reservation_id, and API credentials). These labels are useful for finding
   obvious detector misses, but are not treated as complete ground truth.

2. Human gold labels:
   The prepare command exports blind candidate sheets plus readable source
   documents. Annotators accept/reject candidates and add any missed spans.
   The score command computes span precision/recall/F1, per-type results,
   optional inter-annotator agreement, and detector-vs-human ER on the same
   audited corpus.

True negatives are not reported because token/span NER has no meaningful finite
negative universe. LLMs may be used to propose additional candidates, but their
labels should be human-adjudicated before scoring.

Prepare:
  .venv/bin/python tools/entity_detector_human_audit.py prepare \
    --run-name er_human_v1 --detector-profile paper-er

Score one adjudicated annotation:
  .venv/bin/python tools/entity_detector_human_audit.py score \
    --audit-dir output/entity_detector_human_audit/er_human_v1 \
    --annotations adjudicated.csv \
    --additions additions_adjudicated.csv

Check agreement between two independent annotators:
  .venv/bin/python tools/entity_detector_human_audit.py score \
    --audit-dir output/entity_detector_human_audit/er_human_v1 \
    --annotations annotator_a.csv --annotations-b annotator_b.csv \
    --additions additions_a.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_ARTIFACT_DIRS = [
    OUTPUT_ROOT / "tau2_airline_train50" / "skills_from_train",
    OUTPUT_ROOT / "tau2_banking_knowledge_all97_max50" / "skills_from_train",
    OUTPUT_ROOT / "tau2_retail_all114" / "skills_from_train",
    OUTPUT_ROOT / "tau2_telecom_remaining200_max50" / "skills_from_train",
]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    entity_type: str
    source: str


@dataclass
class Document:
    item_id: str
    domain: str
    doc_type: str
    level: int | None
    source_id: str
    text: str


SAFE_PATHS = {
    "/bin/bash", "/bin/sh", "/usr/bin/env", "/dev/null", "/tmp",
    "/etc/hosts", "/usr/local", "/home", "/root",
}

RELEASE_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("FILE_PATH", re.compile(
        r"(?<!\w)(?:/[\w.\-]+){2,}|(?:[A-Z]:\\[\w.\-\\]+)",
        re.IGNORECASE,
    ), 0),
    ("URL", re.compile(r"https?://[^\s'\"<>)\],}]+", re.IGNORECASE), 0),
    ("IP_ADDRESS", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"
    ), 0),
    ("API_KEY", re.compile(
        r"(?:sk-[\w\-]{20,}|(?:api[_-]?key|token|bearer|password|secret)"
        r"\s*[=:]\s*[\"']?[\w\-]{16,})",
        re.IGNORECASE,
    ), 0),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), 0),
    ("USERNAME", re.compile(
        r"(?:user(?:name)?|login)\s*[=:]\s*[\"']?[\w.\-]+",
        re.IGNORECASE,
    ), 0),
]

PAPER_EXTRA_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), 0),
    ("PERSON", re.compile(
        r"\b(?:customer|passenger|name|Mr\.|Mrs\.|Ms\.)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"
    ), 1),
]

# Schema keys are deliberately broader than the detector patterns. This is
# what allows the silver audit to expose detector blind spots.
KEY_TYPES = {
    "first_name": "PERSON",
    "last_name": "PERSON",
    "full_name": "PERSON",
    "name": "PERSON",
    "email": "EMAIL",
    "email_address": "EMAIL",
    "phone": "PHONE",
    "phone_number": "PHONE",
    "mobile": "PHONE",
    "address": "ADDRESS",
    "address1": "ADDRESS",
    "address2": "ADDRESS",
    "street": "ADDRESS",
    "city": "LOCATION",
    "state": "LOCATION",
    "zip": "POSTAL_CODE",
    "zipcode": "POSTAL_CODE",
    "postal_code": "POSTAL_CODE",
    "country": "LOCATION",
    "dob": "DATE_OF_BIRTH",
    "date_of_birth": "DATE_OF_BIRTH",
    "user_id": "USER_ID",
    "customer_id": "CUSTOMER_ID",
    "account_id": "ACCOUNT_ID",
    "order_id": "ORDER_ID",
    "reservation_id": "RESERVATION_ID",
    "booking_id": "RESERVATION_ID",
    "ticket_id": "TICKET_ID",
    "transaction_id": "TRANSACTION_ID",
    "tracking_id": "TRACKING_ID",
    "payment_method_id": "PAYMENT_ID",
    "card_id": "PAYMENT_ID",
    "credit_card": "PAYMENT_ID",
    "username": "USERNAME",
    "login": "USERNAME",
    "api_key": "API_KEY",
    "token": "API_KEY",
    "bearer": "API_KEY",
    "password": "PASSWORD",
    "secret": "API_KEY",
    "url": "URL",
    "uri": "URL",
    "ip": "IP_ADDRESS",
    "ip_address": "IP_ADDRESS",
    "file_path": "FILE_PATH",
    "path": "FILE_PATH",
}

KEY_ALTERNATION = "|".join(
    sorted((re.escape(key) for key in KEY_TYPES), key=len, reverse=True)
)
QUOTED_KV = re.compile(
    rf"(?P<keyquote>[\"']?)(?P<key>{KEY_ALTERNATION})(?P=keyquote)"
    rf"\s*[=:]\s*(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
UNQUOTED_KV = re.compile(
    rf"(?P<keyquote>[\"']?)(?P<key>{KEY_ALTERNATION})(?P=keyquote)"
    rf"\s*[=:]\s*(?P<value>[A-Za-z0-9_#@./:\\\-]+)",
    re.IGNORECASE,
)


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "skills" in data:
        data = data["skills"]
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def clean_match(text: str, start: int, end: int) -> tuple[int, int, str]:
    raw = text[start:end]
    left = len(raw) - len(raw.lstrip(" \t\"'"))
    right_text = raw.rstrip(" \t\"'.,;:)")
    right = len(raw) - len(right_text)
    new_start = start + left
    new_end = end - right
    return new_start, new_end, text[new_start:new_end]


def detector_spans(text: str, profile: str) -> list[Span]:
    patterns = list(RELEASE_PATTERNS)
    if profile == "paper-er":
        patterns.extend(PAPER_EXTRA_PATTERNS)
    spans: list[Span] = []
    for entity_type, pattern, group in patterns:
        for match in pattern.finditer(text):
            start, end = match.span(group)
            start, end, value = clean_match(text, start, end)
            if not value or len(value) <= 2:
                continue
            if entity_type == "FILE_PATH" and value in SAFE_PATHS:
                continue
            spans.append(Span(start, end, value, entity_type, "detector"))
    return dedupe_spans(spans)


def plausible_schema_value(value: str, entity_type: str) -> bool:
    value = value.strip()
    if not value or value.lower() in {
        "null", "none", "true", "false", "unknown", "n/a",
    }:
        return False
    if entity_type == "PERSON" and value.lower() in {
        "user", "customer", "passenger", "assistant", "agent",
    }:
        return False
    if entity_type in {"LOCATION", "ADDRESS"} and len(value) < 2:
        return False
    if entity_type == "FILE_PATH" and value in SAFE_PATHS:
        return False
    return len(value) >= 2


def structured_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    occupied: set[tuple[int, int]] = set()
    for pattern in (QUOTED_KV, UNQUOTED_KV):
        for match in pattern.finditer(text):
            start, end = match.span("value")
            if (start, end) in occupied:
                continue
            key = match.group("key").lower()
            entity_type = KEY_TYPES[key]
            start, end, value = clean_match(text, start, end)
            if not plausible_schema_value(value, entity_type):
                continue
            occupied.add((start, end))
            spans.append(Span(start, end, value, entity_type, "structured"))

    # If first and last names occur together elsewhere, include the full name as
    # a structured-derived candidate rather than relying on the detector.
    first_names = {
        span.text for span in spans
        if span.entity_type == "PERSON"
        and any(
            text[max(0, span.start - 30):span.start].lower().rstrip().endswith(key)
            for key in ("first_name\": \"", "first_name': '", "first_name=")
        )
    }
    last_names = {
        span.text for span in spans
        if span.entity_type == "PERSON"
        and any(
            text[max(0, span.start - 30):span.start].lower().rstrip().endswith(key)
            for key in ("last_name\": \"", "last_name': '", "last_name=")
        )
    }
    for first in first_names:
        for last in last_names:
            full_name = f"{first} {last}"
            for match in re.finditer(re.escape(full_name), text, flags=re.IGNORECASE):
                spans.append(Span(
                    match.start(),
                    match.end(),
                    match.group(0),
                    "PERSON",
                    "structured",
                ))
    return dedupe_spans(spans)


def dedupe_spans(spans: Iterable[Span]) -> list[Span]:
    merged: dict[tuple[int, int, str], Span] = {}
    for span in spans:
        key = (span.start, span.end, span.entity_type)
        existing = merged.get(key)
        if existing and existing.source != span.source:
            merged[key] = Span(
                span.start,
                span.end,
                span.text,
                span.entity_type,
                "both",
            )
        else:
            merged[key] = span
    return sorted(merged.values(), key=lambda span: (span.start, span.end, span.entity_type))


def skill_text(skill: dict[str, Any]) -> str:
    values: list[str] = []
    for field in (
        "name", "description", "when_to_use", "procedure", "examples", "scripts",
    ):
        value = skill.get(field)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, dict)):
            values.append(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            values.append(str(value))
    return "\n\n".join(values)


def all_documents(levels: set[int]) -> list[Document]:
    documents: list[Document] = []
    for artifact_dir in DEFAULT_ARTIFACT_DIRS:
        if not artifact_dir.exists():
            continue
        domain = artifact_dir.parts[-2]
        for session in load_json(artifact_dir / "sessions.json"):
            task_id = str(session.get("task_id"))
            traces = session.get("traces") or []
            if not isinstance(traces, list):
                traces = [str(traces)]
            for turn_index, trace in enumerate(traces):
                text = str(trace)
                if text.strip():
                    documents.append(Document(
                        item_id=f"{domain}__trace__{task_id}__{turn_index}",
                        domain=domain,
                        doc_type="trace_snippet",
                        level=None,
                        source_id=task_id,
                        text=text,
                    ))

        if 0 in levels:
            skills = [
                skill for skill in load_json(artifact_dir / "level0_all.json")
                if int(skill.get("level") or 0) == 0
            ]
            for skill in skills:
                documents.append(Document(
                    item_id=f"{domain}__skill__L0__{skill.get('id')}",
                    domain=domain,
                    doc_type="skill",
                    level=0,
                    source_id=str(skill.get("id") or ""),
                    text=skill_text(skill),
                ))
        for level in sorted(levels - {0}):
            skills = [
                skill for skill in load_json(
                    artifact_dir / f"round{level}_skills.json"
                )
                if int(skill.get("level") or 0) == level
            ]
            for skill in skills:
                documents.append(Document(
                    item_id=f"{domain}__skill__L{level}__{skill.get('id')}",
                    domain=domain,
                    doc_type="skill",
                    level=level,
                    source_id=str(skill.get("id") or ""),
                    text=skill_text(skill),
                ))
    return documents


def sample_documents(
    documents: list[Document],
    detector_profile: str,
    trace_snippets_per_domain: int,
    skills_per_domain_level: int,
    seed: int,
) -> list[Document]:
    rng = random.Random(seed)
    selected: list[Document] = []
    domains = sorted({document.domain for document in documents})

    for domain in domains:
        trace_docs = [
            document for document in documents
            if document.domain == domain and document.doc_type == "trace_snippet"
        ]
        positive = [
            document for document in trace_docs
            if detector_spans(document.text, detector_profile)
            or structured_spans(document.text)
        ]
        negative = [
            document for document in trace_docs
            if document not in positive
        ]
        rng.shuffle(positive)
        rng.shuffle(negative)
        # Include candidate-free snippets so humans can discover false negatives.
        positive_quota = math.ceil(trace_snippets_per_domain * 0.7)
        chosen = positive[:positive_quota]
        chosen.extend(negative[:trace_snippets_per_domain - len(chosen)])
        if len(chosen) < trace_snippets_per_domain:
            remaining = [document for document in positive if document not in chosen]
            chosen.extend(remaining[:trace_snippets_per_domain - len(chosen)])
        selected.extend(chosen)

        levels = sorted({
            int(document.level)
            for document in documents
            if document.domain == domain
            and document.doc_type == "skill"
            and document.level is not None
        })
        for level in levels:
            skill_docs = [
                document for document in documents
                if document.domain == domain
                and document.doc_type == "skill"
                and document.level == level
            ]
            rng.shuffle(skill_docs)
            selected.extend(skill_docs[:skills_per_domain_level])
    return sorted(selected, key=lambda document: document.item_id)


def span_overlap(left: Span, right: Span) -> bool:
    return left.start < right.end and right.start < left.end


def match_spans(
    predictions: list[Span],
    gold: list[Span],
    exact: bool,
) -> tuple[list[tuple[Span, Span]], list[Span], list[Span]]:
    matches: list[tuple[Span, Span]] = []
    used_gold: set[int] = set()
    for prediction in predictions:
        candidates = []
        for idx, truth in enumerate(gold):
            if idx in used_gold:
                continue
            is_match = (
                prediction.start == truth.start
                and prediction.end == truth.end
                if exact else span_overlap(prediction, truth)
            )
            if is_match:
                type_bonus = int(prediction.entity_type == truth.entity_type)
                overlap = min(prediction.end, truth.end) - max(prediction.start, truth.start)
                candidates.append((type_bonus, overlap, idx, truth))
        if candidates:
            _, _, idx, truth = max(candidates)
            used_gold.add(idx)
            matches.append((prediction, truth))
    matched_predictions = {id(prediction) for prediction, _ in matches}
    false_positives = [
        prediction for prediction in predictions
        if id(prediction) not in matched_predictions
    ]
    false_negatives = [
        truth for idx, truth in enumerate(gold)
        if idx not in used_gold
    ]
    return matches, false_positives, false_negatives


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_spans(
    predictions_by_item: dict[str, list[Span]],
    gold_by_item: dict[str, list[Span]],
    exact: bool,
) -> dict[str, Any]:
    all_matches: list[tuple[Span, Span]] = []
    all_fp: list[Span] = []
    all_fn: list[Span] = []
    for item_id in sorted(set(predictions_by_item) | set(gold_by_item)):
        matches, fp, fn = match_spans(
            predictions_by_item.get(item_id, []),
            gold_by_item.get(item_id, []),
            exact,
        )
        all_matches.extend(matches)
        all_fp.extend(fp)
        all_fn.extend(fn)

    result = prf(len(all_matches), len(all_fp), len(all_fn))
    types = sorted({
        span.entity_type
        for spans in gold_by_item.values()
        for span in spans
    } | {
        span.entity_type
        for spans in predictions_by_item.values()
        for span in spans
    })
    result["by_type"] = {}
    for entity_type in types:
        type_tp = sum(
            1 for prediction, truth in all_matches
            if prediction.entity_type == entity_type
            and truth.entity_type == entity_type
        )
        type_fp = sum(1 for span in all_fp if span.entity_type == entity_type)
        type_fp += sum(
            1 for prediction, truth in all_matches
            if prediction.entity_type == entity_type
            and truth.entity_type != entity_type
        )
        type_fn = sum(1 for span in all_fn if span.entity_type == entity_type)
        type_fn += sum(
            1 for prediction, truth in all_matches
            if truth.entity_type == entity_type
            and prediction.entity_type != entity_type
        )
        result["by_type"][entity_type] = prf(type_tp, type_fp, type_fn)

    confusion: Counter[str] = Counter()
    for prediction, truth in all_matches:
        confusion[f"{truth.entity_type}->{prediction.entity_type}"] += 1
    for span in all_fn:
        confusion[f"{span.entity_type}->MISSED"] += 1
    for span in all_fp:
        confusion[f"NONE->{span.entity_type}"] += 1
    result["type_confusion"] = dict(sorted(confusion.items()))
    return result


def normalize_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().strip("\"'.,;:()[]{}"))


def corpus_er(
    documents: dict[str, Document],
    spans_by_item: dict[str, list[Span]],
) -> dict[str, Any]:
    trace_entities: dict[str, set[str]] = defaultdict(set)
    skill_entities: dict[tuple[str, int], set[str]] = defaultdict(set)
    levels_by_domain: dict[str, set[int]] = defaultdict(set)
    for document in documents.values():
        if document.doc_type == "trace_snippet":
            trace_entities[document.domain]  # Preserve entity-free domains.
        elif document.doc_type == "skill" and document.level is not None:
            levels_by_domain[document.domain].add(document.level)
            skill_entities[(document.domain, document.level)]
    for item_id, spans in spans_by_item.items():
        document = documents.get(item_id)
        if not document:
            continue
        values = {
            normalize_entity(span.text)
            for span in spans
            if normalize_entity(span.text)
        }
        if document.doc_type == "trace_snippet":
            trace_entities[document.domain].update(values)
        elif document.doc_type == "skill" and document.level is not None:
            skill_entities[(document.domain, document.level)].update(values)

    report: dict[str, Any] = {}
    for domain in sorted(trace_entities):
        source = trace_entities[domain]
        report[domain] = {}
        levels = sorted(levels_by_domain[domain])
        for level in levels:
            retained = source & skill_entities[(domain, level)]
            report[domain][str(level)] = {
                "n_source_entities": len(source),
                "n_skill_entities": len(skill_entities[(domain, level)]),
                "n_retained": len(retained),
                "er": round(len(retained) / len(source), 4) if source else None,
            }
    return report


def build_candidates(
    documents: list[Document],
    detector_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, list[Span]], dict[str, list[Span]]]:
    rows: list[dict[str, Any]] = []
    detector_by_item: dict[str, list[Span]] = {}
    structured_by_item: dict[str, list[Span]] = {}
    for document in documents:
        detector = detector_spans(document.text, detector_profile)
        structured = structured_spans(document.text)
        detector_by_item[document.item_id] = detector
        structured_by_item[document.item_id] = structured

        grouped: dict[tuple[int, int, str], dict[str, Any]] = {}
        for span in detector + structured:
            key = (span.start, span.end, span.entity_type)
            grouped.setdefault(key, {
                "span": span,
                "detector": set(),
                "structured": set(),
            })
            if span.source in {"detector", "both"}:
                grouped[key]["detector"].add(span.entity_type)
            if span.source in {"structured", "both"}:
                grouped[key]["structured"].add(span.entity_type)
        for index, group in enumerate(
            sorted(grouped.values(), key=lambda row: (
                row["span"].start,
                row["span"].end,
                row["span"].entity_type,
            ))
        ):
            span: Span = group["span"]
            context_start = max(0, span.start - 80)
            context_end = min(len(document.text), span.end + 80)
            rows.append({
                "candidate_id": f"{document.item_id}__c{index}",
                "item_id": document.item_id,
                "domain": document.domain,
                "doc_type": document.doc_type,
                "level": "" if document.level is None else document.level,
                "start": span.start,
                "end": span.end,
                "entity_text": span.text,
                "proposed_type": span.entity_type,
                "detector_types": "|".join(sorted(group["detector"])),
                "structured_types": "|".join(sorted(group["structured"])),
                "context": document.text[context_start:context_end].replace("\n", "\\n"),
            })
    return rows, detector_by_item, structured_by_item


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_guidelines(path: Path, detector_profile: str) -> None:
    path.write_text(
        f"""# Entity Detector Human Audit

Detector profile under audit: `{detector_profile}`

## Annotation UI

Start the local interface from the repository root:

```bash
.venv/bin/python tools/entity_detector_annotation_ui.py \\
  --audit-dir output/entity_detector_human_audit/er_human_v1
```

Open `http://127.0.0.1:8766`. Use the A and B tabs for independent blind
annotation. Enter the adjudication tab only after both annotators finish.
Selections and decisions are saved directly to the CSV files in this folder.

## Sensitive Entity Taxonomy

Mark concrete values that could identify a person, account, transaction, or
private environment: names, emails, phone numbers, addresses/locations tied to
a user, dates of birth, user/customer/account/order/reservation/transaction
identifiers, payment identifiers, credentials, internal URLs/IPs, usernames,
and non-generic file paths.

Do not mark generic schema words (`user_id`, `email`), placeholders
(`[EMAIL]`), ordinary product names, generic public system paths such as
`/bin/bash`, or purely instructional examples with no concrete value.

## Candidate Annotation

1. Open `annotator_a.csv` or `annotator_b.csv`.
2. Inspect the context and the full file named in `documents.csv`.
3. Set `human_sensitive` to `1` if the candidate is a concrete sensitive
   entity under the taxonomy, otherwise `0`.
4. Correct `human_type` when the proposed type is wrong.
5. Do not infer labels from the detector: annotator sheets intentionally hide
   whether a candidate came from the detector or structured fields.

## Missing Spans

Read every assigned document, including documents with no candidates. Add any
missed entity to the corresponding `additions_*.csv` with exact zero-indexed
`start` and exclusive `end` offsets. Verify that `text[start:end]` equals the
entered `entity_text`.

## Adjudication

Annotators work independently. Resolve disagreements and merge missed-span
additions into `adjudicated.csv` and `additions_adjudicated.csv` before the
final score. Model suggestions may be reviewed, but no model-only label should
enter the adjudicated gold set.
""",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> None:
    levels = {int(value) for value in args.levels.split(",") if value.strip()}
    all_docs = all_documents(levels)
    sampled = sample_documents(
        all_docs,
        args.detector_profile,
        args.trace_snippets_per_domain,
        args.skills_per_domain_level,
        args.seed,
    )
    audit_dir = Path(args.output_dir) / args.run_name
    documents_dir = audit_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows, detector_by_item, structured_by_item = build_candidates(
        sampled,
        args.detector_profile,
    )
    master_fields = [
        "candidate_id", "item_id", "domain", "doc_type", "level",
        "start", "end", "entity_text", "proposed_type", "detector_types",
        "structured_types", "context",
    ]
    write_csv(audit_dir / "candidates_master.csv", candidate_rows, master_fields)

    blind_fields = [
        "candidate_id", "item_id", "domain", "doc_type", "level",
        "start", "end", "entity_text", "proposed_type", "context",
        "human_sensitive", "human_type", "notes",
    ]
    blind_rows = [
        {
            **{field: row.get(field, "") for field in blind_fields},
            "human_sensitive": "",
            "human_type": row["proposed_type"],
            "notes": "",
        }
        for row in candidate_rows
    ]
    write_csv(audit_dir / "annotator_a.csv", blind_rows, blind_fields)
    write_csv(audit_dir / "annotator_b.csv", blind_rows, blind_fields)

    addition_fields = [
        "item_id", "start", "end", "entity_text", "entity_type", "notes",
    ]
    write_csv(audit_dir / "additions_a.csv", [], addition_fields)
    write_csv(audit_dir / "additions_b.csv", [], addition_fields)

    document_rows: list[dict[str, Any]] = []
    for document in sampled:
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", document.item_id) + ".txt"
        (documents_dir / filename).write_text(document.text, encoding="utf-8")
        document_rows.append({
            "item_id": document.item_id,
            "domain": document.domain,
            "doc_type": document.doc_type,
            "level": "" if document.level is None else document.level,
            "source_id": document.source_id,
            "text_path": str(Path("documents") / filename),
            "n_chars": len(document.text),
            "review_complete": "",
            "notes": "",
        })
    document_fields = [
        "item_id", "domain", "doc_type", "level", "source_id",
        "text_path", "n_chars", "review_complete", "notes",
    ]
    write_csv(audit_dir / "documents.csv", document_rows, document_fields)

    # Automatic silver calibration over the sampled documents.
    silver_exact = evaluate_spans(
        detector_by_item,
        structured_by_item,
        exact=True,
    )
    silver_overlap = evaluate_spans(
        detector_by_item,
        structured_by_item,
        exact=False,
    )
    full_rows, full_detector, full_structured = build_candidates(
        all_docs,
        args.detector_profile,
    )
    del full_rows
    full_silver = {
        "exact": evaluate_spans(full_detector, full_structured, exact=True),
        "overlap": evaluate_spans(full_detector, full_structured, exact=False),
        "structured_er": corpus_er(
            {document.item_id: document for document in all_docs},
            full_structured,
        ),
        "detector_er": corpus_er(
            {document.item_id: document for document in all_docs},
            full_detector,
        ),
    }
    manifest = {
        "run_name": args.run_name,
        "detector_profile": args.detector_profile,
        "levels": sorted(levels),
        "seed": args.seed,
        "sampling": {
            "trace_snippets_per_domain": args.trace_snippets_per_domain,
            "skills_per_domain_level": args.skills_per_domain_level,
        },
        "n_all_documents": len(all_docs),
        "n_sampled_documents": len(sampled),
        "n_candidates": len(candidate_rows),
        "sampled_by_domain_type_level": dict(Counter(
            f"{document.domain}|{document.doc_type}|"
            f"{document.level if document.level is not None else 'NA'}"
            for document in sampled
        )),
        "sampled_structured_silver": {
            "exact": silver_exact,
            "overlap": silver_overlap,
        },
        "warning": (
            "Structured labels are high-precision but incomplete silver labels; "
            "use adjudicated human labels for final claims."
        ),
    }
    (audit_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (audit_dir / "full_structured_silver_report.json").write_text(
        json.dumps(full_silver, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_guidelines(audit_dir / "ANNOTATION_GUIDELINES.md", args.detector_profile)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[write] {audit_dir}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cohen_kappa(left: list[int], right: list[int]) -> dict[str, Any]:
    if not left or len(left) != len(right):
        return {"n": 0}
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_rate = sum(left) / len(left)
    right_rate = sum(right) / len(right)
    expected = (
        left_rate * right_rate
        + (1 - left_rate) * (1 - right_rate)
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "n": len(left),
        "agreement": round(observed, 4),
        "cohen_kappa": round(kappa, 4),
        "positive_rate_a": round(left_rate, 4),
        "positive_rate_b": round(right_rate, 4),
    }


def parse_label(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return 1
    if normalized in {"0", "false", "no", "n"}:
        return 0
    return None


def annotation_gold(
    annotations: list[dict[str, str]],
    additions: list[dict[str, str]],
    master_by_candidate: dict[str, dict[str, str]],
    documents: dict[str, Document],
) -> tuple[dict[str, list[Span]], list[str]]:
    gold: dict[str, list[Span]] = defaultdict(list)
    unlabeled: list[str] = []
    for row in annotations:
        candidate_id = row.get("candidate_id", "")
        master = master_by_candidate.get(candidate_id)
        if not master:
            continue
        label = parse_label(row.get("human_sensitive", ""))
        if label is None:
            unlabeled.append(candidate_id)
            continue
        if label == 0:
            continue
        entity_type = row.get("human_type", "").strip() or master["proposed_type"]
        gold[master["item_id"]].append(Span(
            int(master["start"]),
            int(master["end"]),
            master["entity_text"],
            entity_type,
            "human_candidate",
        ))
    for row in additions:
        item_id = row.get("item_id", "")
        if not item_id or item_id not in documents:
            continue
        try:
            start = int(row["start"])
            end = int(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = documents[item_id].text[start:end]
        entered = row.get("entity_text", "")
        if entered and text != entered:
            raise SystemExit(
                f"Addition offset mismatch for {item_id}: "
                f"text[{start}:{end}]={text!r}, entered={entered!r}"
            )
        gold[item_id].append(Span(
            start,
            end,
            text,
            row.get("entity_type", "").strip() or "OTHER_SENSITIVE",
            "human_addition",
        ))
    return {
        item_id: dedupe_spans(spans)
        for item_id, spans in gold.items()
    }, unlabeled


def score(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    manifest = json.loads((audit_dir / "manifest.json").read_text(encoding="utf-8"))
    master_rows = read_csv(audit_dir / "candidates_master.csv")
    master_by_candidate = {
        row["candidate_id"]: row for row in master_rows
    }
    document_rows = read_csv(audit_dir / "documents.csv")
    documents: dict[str, Document] = {}
    for row in document_rows:
        text = (audit_dir / row["text_path"]).read_text(encoding="utf-8")
        level = int(row["level"]) if row.get("level", "") != "" else None
        documents[row["item_id"]] = Document(
            item_id=row["item_id"],
            domain=row["domain"],
            doc_type=row["doc_type"],
            level=level,
            source_id=row["source_id"],
            text=text,
        )

    annotation_path = Path(args.annotations)
    if not annotation_path.is_absolute():
        annotation_path = audit_dir / annotation_path
    additions_path = Path(args.additions)
    if not additions_path.is_absolute():
        additions_path = audit_dir / additions_path
    annotations = read_csv(annotation_path)
    additions = read_csv(additions_path)
    gold_by_item, unlabeled = annotation_gold(
        annotations,
        additions,
        master_by_candidate,
        documents,
    )
    if unlabeled and not args.allow_incomplete:
        raise SystemExit(
            f"{len(unlabeled)} candidates are unlabeled. Complete the sheet or "
            "pass --allow-incomplete for a provisional report."
        )

    detector_by_item = {
        item_id: detector_spans(document.text, manifest["detector_profile"])
        for item_id, document in documents.items()
    }
    report: dict[str, Any] = {
        "detector_profile": manifest["detector_profile"],
        "n_documents": len(documents),
        "n_candidates": len(master_rows),
        "n_unlabeled_candidates": len(unlabeled),
        "n_human_additions": len(additions),
        "exact_span": evaluate_spans(
            detector_by_item,
            gold_by_item,
            exact=True,
        ),
        "overlap_span": evaluate_spans(
            detector_by_item,
            gold_by_item,
            exact=False,
        ),
        "human_er_on_audited_corpus": corpus_er(documents, gold_by_item),
        "detector_er_on_audited_corpus": corpus_er(documents, detector_by_item),
        "er_note": (
            "These ER values are computed on the same stratified human-audited "
            "corpus and calibrate detector error; they do not replace the full "
            "main-paper ER estimate."
        ),
    }

    if args.annotations_b:
        annotation_b_path = Path(args.annotations_b)
        if not annotation_b_path.is_absolute():
            annotation_b_path = audit_dir / annotation_b_path
        rows_b = read_csv(annotation_b_path)
        labels_a = {
            row["candidate_id"]: parse_label(row.get("human_sensitive", ""))
            for row in annotations
        }
        labels_b = {
            row["candidate_id"]: parse_label(row.get("human_sensitive", ""))
            for row in rows_b
        }
        shared = sorted(
            candidate_id for candidate_id in labels_a
            if labels_a[candidate_id] is not None
            and labels_b.get(candidate_id) is not None
        )
        report["inter_annotator"] = cohen_kappa(
            [int(labels_a[candidate_id]) for candidate_id in shared],
            [int(labels_b[candidate_id]) for candidate_id in shared],
        )
        report["inter_annotator"]["n_disagreements"] = sum(
            labels_a[candidate_id] != labels_b[candidate_id]
            for candidate_id in shared
        )

    output_path = (
        Path(args.output)
        if args.output else audit_dir / "human_gold_score.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[write] {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-name", default="er_human_v1")
    prepare_parser.add_argument(
        "--detector-profile",
        choices=["paper-er", "release-regex"],
        default="paper-er",
    )
    prepare_parser.add_argument("--levels", default="0,1,2,3")
    prepare_parser.add_argument("--trace-snippets-per-domain", type=int, default=30)
    prepare_parser.add_argument("--skills-per-domain-level", type=int, default=5)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "entity_detector_human_audit"),
    )
    prepare_parser.set_defaults(func=prepare)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--audit-dir", required=True)
    score_parser.add_argument("--annotations", required=True)
    score_parser.add_argument("--annotations-b", default="")
    score_parser.add_argument("--additions", required=True)
    score_parser.add_argument("--allow-incomplete", action="store_true")
    score_parser.add_argument("--output", default="")
    score_parser.set_defaults(func=score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
