"""Generate model suggestions for the entity-detector human audit.

The output is intentionally a suggestion layer: it does not write to
annotator_a.csv or annotator_b.csv. Humans can inspect and accept/reject the
suggestions in the local annotation UI.

Example:
  DEEPSEEK_API_KEY=... .venv/bin/python tools/entity_detector_model_prelabel.py \
    --audit-dir output/entity_detector_human_audit/er_human_v1 \
    --provider deepseek --model deepseek-chat
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from openai import OpenAI  # noqa: E402

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
    },
    "generic": {
        "base_url": "",
        "api_key_env": "MODEL_AUDIT_API_KEY",
        "model": "",
    },
}

SYSTEM_PROMPT = """You are a privacy/entity audit assistant.

Your task is to help human annotators audit an entity detector. Review the
document text and the detector's candidate spans.

Treat as sensitive if the span could identify or link a user, account, order,
booking, payment method, credential, private location/address/contact detail,
or system-specific resource. Examples: person names, emails, phone numbers,
addresses, user/customer/account/order/reservation/ticket/transaction/tracking
IDs, usernames, passwords, API keys/tokens, URLs, IP addresses, file paths, and
birth dates. Do not mark generic product names, generic API/tool names, common
non-user-specific words, or ordinary task descriptions as sensitive.

Return JSON only with this shape:
{
  "candidates": [
    {
      "candidate_id": "string",
      "sensitive": true,
      "entity_type": "PERSON|EMAIL|PHONE|ADDRESS|LOCATION|POSTAL_CODE|DATE_OF_BIRTH|USER_ID|CUSTOMER_ID|ACCOUNT_ID|ORDER_ID|RESERVATION_ID|TICKET_ID|TRANSACTION_ID|TRACKING_ID|PAYMENT_ID|USERNAME|PASSWORD|API_KEY|URL|IP_ADDRESS|FILE_PATH|OTHER_SENSITIVE",
      "confidence": 0.0,
      "reason": "short reason"
    }
  ],
  "missed_entities": [
    {
      "text": "exact text span from the document",
      "start": 0,
      "end": 0,
      "entity_type": "PERSON",
      "confidence": 0.0,
      "reason": "short reason"
    }
  ]
}

For missed_entities, propose only spans that are not already present in the
candidate list. Prefer exact document offsets when possible. Keep reasons under
12 words. Return at most 30 missed_entities for a document.
"""

USER_PROMPT = """### Document metadata
item_id: {item_id}
domain: {domain}
doc_type: {doc_type}
level: {level}

### Document text
{text}

### Detector candidates
{candidates_json}
"""

ENTITY_TYPES = {
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "LOCATION",
    "POSTAL_CODE",
    "DATE_OF_BIRTH",
    "USER_ID",
    "CUSTOMER_ID",
    "ACCOUNT_ID",
    "ORDER_ID",
    "RESERVATION_ID",
    "TICKET_ID",
    "TRANSACTION_ID",
    "TRACKING_ID",
    "PAYMENT_ID",
    "USERNAME",
    "PASSWORD",
    "API_KEY",
    "URL",
    "IP_ADDRESS",
    "FILE_PATH",
    "OTHER_SENSITIVE",
}


class ModelOutputError(RuntimeError):
    """Raised when the model response cannot be parsed as JSON."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "sensitive"}:
        return True
    if text in {"0", "false", "no", "n", "non-sensitive", "nonsensitive"}:
        return False
    return None


def normalize_type(value: Any, fallback: str = "OTHER_SENSITIVE") -> str:
    text = str(value or "").strip().upper()
    return text if text in ENTITY_TYPES else fallback


def normalize_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score > 1.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise


def resolve_provider(args: argparse.Namespace) -> tuple[str, str, str]:
    spec = PROVIDERS[args.provider]
    api_key = args.api_key or os.environ.get(args.api_key_env or spec["api_key_env"], "")
    base_url = args.base_url or spec["base_url"]
    model = args.model or spec["model"]
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env or spec['api_key_env']} "
            "or pass --api-key."
        )
    if not base_url:
        raise SystemExit("Missing base URL. Pass --base-url for provider=generic.")
    if not model:
        raise SystemExit("Missing model. Pass --model for provider=generic.")
    return api_key, base_url, model


def response_content(response: Any) -> str:
    message = response.choices[0].message
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def candidate_payload(candidates: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "text": row["entity_text"],
            "proposed_type": row["proposed_type"],
            "context": row.get("context", "").replace("\n", " ")[:400],
        }
        for row in candidates
    ]


def call_model(
    client: OpenAI,
    model: str,
    document: dict[str, str],
    text: str,
    candidates: list[dict[str, str]],
    timeout_s: float,
    temperature: float,
    max_tokens: int,
    raw_dir: Path,
    repair_attempts: int,
    json_mode: bool,
) -> dict[str, Any]:
    prompt = USER_PROMPT.format(
        item_id=document["item_id"],
        domain=document["domain"],
        doc_type=document["doc_type"],
        level=document.get("level", ""),
        text=text,
        candidates_json=json.dumps(
            candidate_payload(candidates),
            indent=2,
            ensure_ascii=False,
        ),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for attempt in range(repair_attempts + 1):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as error:
            if json_mode and "response_format" in str(error):
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        raw_text = response_content(response)
        try:
            return extract_json(raw_text)
        except json.JSONDecodeError as error:
            last_error = error
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{document['item_id']}_attempt{attempt}_bad_response.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            if attempt >= repair_attempts:
                break
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair malformed JSON. Return only one valid JSON "
                        "object with keys candidates and missed_entities. "
                        "Do not add explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "The previous response was invalid JSON. Repair it "
                        "without changing the intended labels:\n\n"
                        f"{raw_text}"
                    ),
                },
            ]
    raise ModelOutputError(
        f"Invalid JSON from model for {document['item_id']}: {last_error}"
    )


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def resolve_extra_span(
    raw: dict[str, Any],
    text: str,
    candidate_spans: list[tuple[int, int]],
) -> dict[str, Any]:
    entity_text = str(raw.get("text") or "").strip()
    start: int | None = None
    end: int | None = None
    note = ""

    try:
        proposed_start = int(raw.get("start"))
        proposed_end = int(raw.get("end"))
    except (TypeError, ValueError):
        proposed_start = proposed_end = -1

    if (
        entity_text
        and 0 <= proposed_start < proposed_end <= len(text)
        and text[proposed_start:proposed_end] == entity_text
    ):
        start, end = proposed_start, proposed_end
        note = "offset"
    elif entity_text:
        matches = [m for m in re.finditer(re.escape(entity_text), text)]
        if len(matches) == 1:
            start, end = matches[0].span()
            note = "unique_text"
        elif len(matches) > 1:
            note = f"ambiguous_text:{len(matches)}"
        else:
            note = "text_not_found"
    else:
        note = "empty_text"

    resolved = start is not None and end is not None
    overlaps_candidate = (
        resolved
        and any(spans_overlap(start, end, c_start, c_end) for c_start, c_end in candidate_spans)
    )
    return {
        "text": entity_text,
        "start": start,
        "end": end,
        "entity_type": normalize_type(raw.get("entity_type")),
        "confidence": normalize_confidence(raw.get("confidence")),
        "reason": str(raw.get("reason") or "").strip()[:240],
        "resolved": resolved,
        "resolution_note": note,
        "overlaps_existing_candidate": bool(overlaps_candidate),
    }


def normalize_model_output(
    parsed: dict[str, Any],
    document: dict[str, str],
    text: str,
    candidates: list[dict[str, str]],
) -> dict[str, Any]:
    by_candidate = {row["candidate_id"]: row for row in candidates}
    candidate_suggestions: dict[str, dict[str, Any]] = {}
    for raw in parsed.get("candidates") or []:
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("candidate_id") or "")
        if candidate_id not in by_candidate:
            continue
        sensitive = normalize_bool(raw.get("sensitive"))
        if sensitive is None:
            continue
        fallback_type = by_candidate[candidate_id].get("proposed_type") or "OTHER_SENSITIVE"
        candidate_suggestions[candidate_id] = {
            "candidate_id": candidate_id,
            "human_sensitive": "1" if sensitive else "0",
            "sensitive": sensitive,
            "entity_type": normalize_type(raw.get("entity_type"), fallback_type),
            "confidence": normalize_confidence(raw.get("confidence")),
            "reason": str(raw.get("reason") or "").strip()[:240],
        }

    candidate_spans = [
        (int(row["start"]), int(row["end"]))
        for row in candidates
    ]
    extras: list[dict[str, Any]] = []
    seen: set[tuple[int | None, int | None, str]] = set()
    for raw in parsed.get("missed_entities") or []:
        if not isinstance(raw, dict):
            continue
        extra = resolve_extra_span(raw, text, candidate_spans)
        key = (extra["start"], extra["end"], extra["text"])
        if key in seen:
            continue
        seen.add(key)
        extra["item_id"] = document["item_id"]
        extra["suggestion_id"] = (
            f"{document['item_id']}__model_extra__{len(extras)}"
        )
        extras.append(extra)

    return {
        "item_id": document["item_id"],
        "candidate_suggestions": candidate_suggestions,
        "extra_suggestions": extras,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_total = sum(len(row["candidate_suggestions"]) for row in rows)
    extras = [extra for row in rows for extra in row["extra_suggestions"]]
    resolved_extras = [
        extra for extra in extras
        if extra.get("resolved") and not extra.get("overlaps_existing_candidate")
    ]
    confidences = [
        float(suggestion["confidence"])
        for row in rows
        for suggestion in row["candidate_suggestions"].values()
    ]
    return {
        "documents": len(rows),
        "candidate_suggestions": candidate_total,
        "extra_suggestions": len(extras),
        "resolved_nonoverlap_extra_suggestions": len(resolved_extras),
        "mean_candidate_confidence": (
            round(statistics.mean(confidences), 3) if confidences else None
        ),
    }


def consolidate(
    rows: list[dict[str, Any]],
    audit_dir: Path,
    provider: str,
    model: str,
) -> dict[str, Any]:
    by_item = {
        row["item_id"]: {
            "candidates": row["candidate_suggestions"],
            "extras": row["extra_suggestions"],
        }
        for row in rows
    }
    return {
        "audit_dir": str(audit_dir.resolve()),
        "provider": provider,
        "model": model,
        "summary": build_summary(rows),
        "by_item": by_item,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--item-id", action="append", default=[])
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-document-chars", type=int, default=40000)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_dir = Path(args.audit_dir)
    if not audit_dir.is_absolute():
        audit_dir = Path.cwd() / audit_dir
    documents = read_csv(audit_dir / "documents.csv")
    candidates = read_csv(audit_dir / "candidates_master.csv")
    candidates_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_item[row["item_id"]].append(row)

    if args.item_id:
        wanted = set(args.item_id)
        documents = [row for row in documents if row["item_id"] in wanted]
    if args.limit:
        documents = documents[: args.limit]

    output_dir = audit_dir / "model_suggestions"
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key, base_url, model = resolve_provider(args)
    model_tag = model.replace("/", "_").replace(":", "_")
    prefix = f"{args.provider}_{model_tag}"
    rows_path = output_dir / f"{prefix}_rows.jsonl"
    errors_path = output_dir / f"{prefix}_errors.jsonl"
    raw_dir = output_dir / "raw_bad_responses"
    suggestions_path = output_dir / f"{prefix}_suggestions.json"
    latest_path = output_dir / "latest_suggestions.json"

    done: set[str] = set()
    rows: list[dict[str, Any]] = []
    if rows_path.exists() and not args.force:
        with rows_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
                done.add(row["item_id"])

    print(f"[load] {len(documents)} documents selected; {len(done)} already done")
    if args.dry_run:
        print("[dry-run] no model calls made")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    with rows_path.open("a", encoding="utf-8") as handle:
        for index, document in enumerate(documents, start=1):
            item_id = document["item_id"]
            if item_id in done:
                continue
            text_path = audit_dir / document["text_path"]
            text = text_path.read_text(encoding="utf-8")
            if len(text) > args.max_document_chars:
                raise SystemExit(
                    f"{item_id} has {len(text)} chars, exceeding "
                    f"--max-document-chars={args.max_document_chars}."
                )
            item_candidates = candidates_by_item.get(item_id, [])
            print(
                f"[{index}/{len(documents)}] {item_id}: "
                f"{len(item_candidates)} candidates, {len(text)} chars"
            )
            try:
                parsed = call_model(
                    client=client,
                    model=model,
                    document=document,
                    text=text,
                    candidates=item_candidates,
                    timeout_s=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    raw_dir=raw_dir,
                    repair_attempts=max(0, args.repair_attempts),
                    json_mode=not args.no_json_mode,
                )
            except Exception as error:
                error_row = {
                    "item_id": item_id,
                    "error": repr(error),
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                with errors_path.open("a", encoding="utf-8") as ef:
                    ef.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                print(f"[error] {item_id}: {error}")
                print(f"[error] raw bad responses, if any: {raw_dir}")
                if args.stop_on_error:
                    raise
                continue
            row = normalize_model_output(parsed, document, text, item_candidates)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            consolidated = consolidate(rows, audit_dir, args.provider, model)
            atomic_write_json(suggestions_path, consolidated)
            atomic_write_json(latest_path, consolidated)
            if args.sleep:
                time.sleep(args.sleep)

    consolidated = consolidate(rows, audit_dir, args.provider, model)
    atomic_write_json(suggestions_path, consolidated)
    atomic_write_json(latest_path, consolidated)
    print(f"[write] {suggestions_path}")
    print(f"[write] {latest_path}")
    print(json.dumps(consolidated["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
