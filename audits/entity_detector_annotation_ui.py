"""Local browser UI for the entity-detector human audit.

The UI writes directly to the CSV files produced by
``entity_detector_human_audit.py prepare``. Annotator A and B remain blind to
candidate provenance. The adjudication view initializes agreements
automatically and leaves disagreements unresolved.

Run:
  .venv/bin/python tools/entity_detector_annotation_ui.py \
    --audit-dir output/entity_detector_human_audit/er_human_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / "entity_detector_annotation_ui"
ANNOTATION_FILES = {
    "a": "annotator_a.csv",
    "b": "annotator_b.csv",
    "adjudicated": "adjudicated.csv",
}
ADDITION_FILES = {
    "a": "additions_a.csv",
    "b": "additions_b.csv",
    "adjudicated": "additions_adjudicated.csv",
}
VALID_LABELS = {"", "0", "1"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


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


def normalized_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "1"
    if text in {"0", "false", "no", "n"}:
        return "0"
    return ""


def addition_key(row: dict[str, Any]) -> str:
    return ":".join([
        str(row.get("item_id", "")),
        str(row.get("start", "")),
        str(row.get("end", "")),
    ])


class AuditStore:
    def __init__(self, audit_dir: Path, suggestions_path: Path | None = None) -> None:
        self.audit_dir = audit_dir.resolve()
        self.lock = threading.RLock()
        self.manifest = json.loads(
            (self.audit_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.document_fields, self.documents = read_csv(
            self.audit_dir / "documents.csv"
        )
        self.documents_by_id = {
            row["item_id"]: row for row in self.documents
        }
        self.text_by_id = {
            row["item_id"]: (
                self.audit_dir / row["text_path"]
            ).read_text(encoding="utf-8")
            for row in self.documents
        }
        self.master_fields, self.master_rows = read_csv(
            self.audit_dir / "candidates_master.csv"
        )
        self.master_by_id = {
            row["candidate_id"]: row for row in self.master_rows
        }
        self.suggestions_path = self.resolve_suggestions_path(suggestions_path)
        self.model_suggestions = self.load_model_suggestions()

    def resolve_suggestions_path(self, suggestions_path: Path | None) -> Path | None:
        if suggestions_path is not None:
            path = suggestions_path
            if not path.is_absolute():
                path = Path.cwd() / path
            return path
        default = self.audit_dir / "model_suggestions" / "latest_suggestions.json"
        return default if default.exists() else None

    def load_model_suggestions(self) -> dict[str, Any]:
        if not self.suggestions_path or not self.suggestions_path.exists():
            return {}
        try:
            data = json.loads(self.suggestions_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        by_item = data.get("by_item", {})
        return by_item if isinstance(by_item, dict) else {}

    def annotation_path(self, mode: str) -> Path:
        if mode not in ANNOTATION_FILES:
            raise ValueError(f"Invalid mode: {mode}")
        return self.audit_dir / ANNOTATION_FILES[mode]

    def addition_path(self, mode: str) -> Path:
        if mode not in ADDITION_FILES:
            raise ValueError(f"Invalid mode: {mode}")
        return self.audit_dir / ADDITION_FILES[mode]

    @property
    def review_path(self) -> Path:
        return self.audit_dir / "ui_review_state.json"

    @property
    def addition_review_path(self) -> Path:
        return self.audit_dir / "adjudication_addition_state.json"

    @property
    def model_suggestion_state_path(self) -> Path:
        return self.audit_dir / "ui_model_suggestion_state.json"

    def read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    def ensure_adjudication_files(self) -> None:
        with self.lock:
            output = self.annotation_path("adjudicated")
            if not output.exists():
                fields_a, rows_a = read_csv(self.annotation_path("a"))
                _, rows_b = read_csv(self.annotation_path("b"))
                by_b = {row["candidate_id"]: row for row in rows_b}
                adjudicated: list[dict[str, str]] = []
                for row_a in rows_a:
                    row = dict(row_a)
                    row_b = by_b.get(row["candidate_id"], {})
                    label_a = normalized_label(row_a.get("human_sensitive"))
                    label_b = normalized_label(row_b.get("human_sensitive"))
                    type_a = row_a.get("human_type", "").strip()
                    type_b = row_b.get("human_type", "").strip()
                    labels_agree = label_a != "" and label_a == label_b
                    types_agree = label_a != "1" or (
                        type_a != "" and type_a == type_b
                    )
                    if labels_agree and types_agree:
                        row["human_sensitive"] = label_a
                        row["human_type"] = type_a or row["proposed_type"]
                    else:
                        row["human_sensitive"] = ""
                        row["human_type"] = row["proposed_type"]
                    row["notes"] = ""
                    adjudicated.append(row)
                atomic_write_csv(output, fields_a, adjudicated)

            additions_output = self.addition_path("adjudicated")
            if not additions_output.exists():
                fields, _ = read_csv(self.addition_path("a"))
                atomic_write_csv(additions_output, fields, [])

    def annotations(self, mode: str) -> tuple[list[str], list[dict[str, str]]]:
        if mode == "adjudicated":
            self.ensure_adjudication_files()
            self.sync_new_agreements()
        return read_csv(self.annotation_path(mode))

    def additions(self, mode: str) -> tuple[list[str], list[dict[str, str]]]:
        if mode == "adjudicated":
            self.ensure_adjudication_files()
        return read_csv(self.addition_path(mode))

    def sync_new_agreements(self) -> None:
        """Fill unresolved adjudication rows when A and B now agree."""
        output = self.annotation_path("adjudicated")
        if not output.exists():
            return
        fields, adjudicated = read_csv(output)
        _, rows_a = read_csv(self.annotation_path("a"))
        _, rows_b = read_csv(self.annotation_path("b"))
        by_a = {row["candidate_id"]: row for row in rows_a}
        by_b = {row["candidate_id"]: row for row in rows_b}
        changed = False
        for row in adjudicated:
            if normalized_label(row.get("human_sensitive")) != "":
                continue
            row_a = by_a.get(row["candidate_id"], {})
            row_b = by_b.get(row["candidate_id"], {})
            label_a = normalized_label(row_a.get("human_sensitive"))
            label_b = normalized_label(row_b.get("human_sensitive"))
            type_a = row_a.get("human_type", "").strip()
            type_b = row_b.get("human_type", "").strip()
            if label_a == "" or label_a != label_b:
                continue
            if label_a == "1" and (not type_a or type_a != type_b):
                continue
            row["human_sensitive"] = label_a
            row["human_type"] = type_a or row["proposed_type"]
            changed = True
        if changed:
            atomic_write_csv(output, fields, adjudicated)

    def review_state(self) -> dict[str, dict[str, bool]]:
        state = self.read_json(self.review_path, {})
        return {
            mode: {
                str(item_id): bool(complete)
                for item_id, complete in state.get(mode, {}).items()
            }
            for mode in ANNOTATION_FILES
        }

    def model_suggestion_state(self) -> dict[str, dict[str, str]]:
        state = self.read_json(self.model_suggestion_state_path, {})
        return {
            mode: {
                str(suggestion_id): str(status)
                for suggestion_id, status in state.get(mode, {}).items()
            }
            for mode in ANNOTATION_FILES
        }

    def set_model_suggestion_state(self, payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", ""))
        suggestion_id = str(payload.get("suggestion_id", ""))
        status = str(payload.get("status", ""))
        if mode not in {"a", "b"}:
            raise ValueError("Model suggestions are only available in A/B mode")
        if status not in {"accepted", "ignored"}:
            raise ValueError("Invalid model suggestion status")
        if not suggestion_id:
            raise ValueError("Missing suggestion id")
        with self.lock:
            state = self.model_suggestion_state()
            state.setdefault(mode, {})[suggestion_id] = status
            atomic_write_json(self.model_suggestion_state_path, state)

    def adjudication_sources(
        self,
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        _, rows_a = self.annotations("a")
        _, rows_b = self.annotations("b")
        return (
            {row["candidate_id"]: row for row in rows_a},
            {row["candidate_id"]: row for row in rows_b},
        )

    def merged_additions(self) -> list[dict[str, Any]]:
        _, rows_a = self.additions("a")
        _, rows_b = self.additions("b")
        grouped: dict[str, dict[str, Any]] = {}
        for source, rows in (("A", rows_a), ("B", rows_b)):
            for row in rows:
                key = addition_key(row)
                current = grouped.setdefault(key, {
                    **row,
                    "key": key,
                    "sources": [],
                    "source_types": {},
                    "source_notes": {},
                })
                current["sources"].append(source)
                current["source_types"][source] = row.get("entity_type", "")
                current["source_notes"][source] = row.get("notes", "")

        state = self.read_json(self.addition_review_path, {})
        changed = False
        for key, row in grouped.items():
            if key not in state:
                source_types = set(row["source_types"].values())
                if set(row["sources"]) == {"A", "B"} and len(source_types) == 1:
                    state[key] = {
                        "included": True,
                        "entity_type": next(iter(source_types)),
                        "notes": "",
                    }
                    changed = True
                else:
                    state[key] = {
                        "included": None,
                        "entity_type": row.get("entity_type", ""),
                        "notes": "",
                    }
                    changed = True
            row["decision"] = state[key]
        if changed:
            atomic_write_json(self.addition_review_path, state)
            self.rebuild_adjudicated_additions(grouped, state)
        return sorted(
            grouped.values(),
            key=lambda row: (
                row["item_id"],
                int(row["start"]),
                int(row["end"]),
            ),
        )

    def rebuild_adjudicated_additions(
        self,
        grouped: dict[str, dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        if grouped is None:
            rows = self.merged_additions()
            grouped = {row["key"]: row for row in rows}
        if state is None:
            state = self.read_json(self.addition_review_path, {})
        fields, _ = self.additions("a")
        output: list[dict[str, str]] = []
        for key, decision in state.items():
            row = grouped.get(key)
            if not row or decision.get("included") is not True:
                continue
            output.append({
                "item_id": row["item_id"],
                "start": row["start"],
                "end": row["end"],
                "entity_text": row["entity_text"],
                "entity_type": (
                    decision.get("entity_type")
                    or row.get("entity_type")
                    or "OTHER_SENSITIVE"
                ),
                "notes": decision.get("notes", ""),
            })
        atomic_write_csv(self.addition_path("adjudicated"), fields, output)

    def bootstrap(self, mode: str) -> dict[str, Any]:
        with self.lock:
            fields, rows = self.annotations(mode)
            del fields
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped[row["item_id"]].append(row)
            reviews = self.review_state().get(mode, {})
            pending_additions: dict[str, int] = defaultdict(int)
            if mode == "adjudicated":
                for row in self.merged_additions():
                    if row["decision"].get("included") is None:
                        pending_additions[row["item_id"]] += 1

            documents = []
            for row in self.documents:
                candidates = grouped.get(row["item_id"], [])
                labeled = sum(
                    normalized_label(candidate.get("human_sensitive")) != ""
                    for candidate in candidates
                )
                pending = pending_additions.get(row["item_id"], 0)
                documents.append({
                    **row,
                    "candidate_total": len(candidates),
                    "candidate_labeled": labeled,
                    "addition_pending": pending,
                    "review_complete": reviews.get(row["item_id"], False),
                })
            return {
                "manifest": {
                    "run_name": self.manifest.get("run_name"),
                    "detector_profile": self.manifest.get("detector_profile"),
                    "n_sampled_documents": self.manifest.get(
                        "n_sampled_documents"
                    ),
                    "n_candidates": self.manifest.get("n_candidates"),
                },
                "mode": mode,
                "model_suggestions_loaded": bool(self.model_suggestions),
                "model_suggestions_path": (
                    str(self.suggestions_path) if self.suggestions_path else ""
                ),
                "documents": documents,
            }

    def document(self, mode: str, item_id: str) -> dict[str, Any]:
        with self.lock:
            if item_id not in self.documents_by_id:
                raise KeyError(item_id)
            _, rows = self.annotations(mode)
            candidates = [
                dict(row) for row in rows if row["item_id"] == item_id
            ]
            item_suggestions = self.model_suggestions.get(item_id, {})
            candidate_suggestions = item_suggestions.get("candidates", {})
            if isinstance(candidate_suggestions, dict):
                for row in candidates:
                    suggestion = candidate_suggestions.get(row["candidate_id"])
                    if suggestion:
                        row["model_suggestion"] = suggestion
            if mode == "adjudicated":
                source_a, source_b = self.adjudication_sources()
                for row in candidates:
                    candidate_id = row["candidate_id"]
                    a = source_a.get(candidate_id, {})
                    b = source_b.get(candidate_id, {})
                    row["annotator_a"] = {
                        "human_sensitive": normalized_label(
                            a.get("human_sensitive")
                        ),
                        "human_type": a.get("human_type", ""),
                        "notes": a.get("notes", ""),
                    }
                    row["annotator_b"] = {
                        "human_sensitive": normalized_label(
                            b.get("human_sensitive")
                        ),
                        "human_type": b.get("human_type", ""),
                        "notes": b.get("notes", ""),
                    }
                additions = [
                    row for row in self.merged_additions()
                    if row["item_id"] == item_id
                ]
            else:
                _, all_additions = self.additions(mode)
                additions = [
                    {**row, "key": addition_key(row)}
                    for row in all_additions
                    if row["item_id"] == item_id
                ]
            model_extras = []
            if mode in {"a", "b"}:
                existing_addition_keys = {
                    addition_key(row) for row in additions
                }
                states = self.model_suggestion_state().get(mode, {})
                for extra in item_suggestions.get("extras", []):
                    if not isinstance(extra, dict):
                        continue
                    if (
                        not extra.get("resolved")
                        or extra.get("overlaps_existing_candidate")
                        or extra.get("suggestion_id") in states
                    ):
                        continue
                    try:
                        key = ":".join([
                            item_id,
                            str(int(extra["start"])),
                            str(int(extra["end"])),
                        ])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if key in existing_addition_keys:
                        continue
                    model_extras.append(extra)

            return {
                "document": self.documents_by_id[item_id],
                "text": self.text_by_id[item_id],
                "candidates": candidates,
                "additions": additions,
                "model_extra_suggestions": model_extras,
                "review_complete": self.review_state()
                .get(mode, {})
                .get(item_id, False),
            }

    def save_candidate(self, payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", ""))
        candidate_id = str(payload.get("candidate_id", ""))
        label = normalized_label(payload.get("human_sensitive", ""))
        if mode not in ANNOTATION_FILES or candidate_id not in self.master_by_id:
            raise ValueError("Unknown mode or candidate")
        if label not in VALID_LABELS:
            raise ValueError("Invalid label")
        with self.lock:
            fields, rows = self.annotations(mode)
            found = False
            for row in rows:
                if row["candidate_id"] != candidate_id:
                    continue
                row["human_sensitive"] = label
                row["human_type"] = str(payload.get("human_type", "")).strip()
                row["notes"] = str(payload.get("notes", "")).strip()
                found = True
                break
            if not found:
                raise ValueError("Candidate not found in annotation sheet")
            atomic_write_csv(self.annotation_path(mode), fields, rows)

    def save_addition(self, payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", ""))
        if mode not in {"a", "b"}:
            raise ValueError("Direct additions are only available in A/B mode")
        item_id = str(payload.get("item_id", ""))
        if item_id not in self.text_by_id:
            raise ValueError("Unknown document")
        try:
            start = int(payload["start"])
            end = int(payload["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid offsets") from error
        text = self.text_by_id[item_id]
        if start < 0 or end <= start or end > len(text):
            raise ValueError("Offsets are outside the document")
        entity_text = text[start:end]
        entity_type = str(payload.get("entity_type", "")).strip()
        if not entity_type:
            raise ValueError("Entity type is required")
        new_row = {
            "item_id": item_id,
            "start": str(start),
            "end": str(end),
            "entity_text": entity_text,
            "entity_type": entity_type,
            "notes": str(payload.get("notes", "")).strip(),
        }
        with self.lock:
            fields, rows = self.additions(mode)
            key = addition_key(new_row)
            for row in rows:
                if addition_key(row) == key:
                    row.update(new_row)
                    break
            else:
                rows.append(new_row)
            atomic_write_csv(self.addition_path(mode), fields, rows)

    def delete_addition(self, payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", ""))
        key = str(payload.get("key", ""))
        if mode not in {"a", "b"}:
            raise ValueError("Only A/B additions can be deleted")
        with self.lock:
            fields, rows = self.additions(mode)
            remaining = [row for row in rows if addition_key(row) != key]
            atomic_write_csv(self.addition_path(mode), fields, remaining)

    def save_addition_decision(self, payload: dict[str, Any]) -> None:
        key = str(payload.get("key", ""))
        included = payload.get("included")
        if included not in {True, False, None}:
            raise ValueError("Invalid addition decision")
        with self.lock:
            grouped_rows = self.merged_additions()
            grouped = {row["key"]: row for row in grouped_rows}
            if key not in grouped:
                raise ValueError("Unknown addition")
            state = self.read_json(self.addition_review_path, {})
            state[key] = {
                "included": included,
                "entity_type": str(payload.get("entity_type", "")).strip(),
                "notes": str(payload.get("notes", "")).strip(),
            }
            atomic_write_json(self.addition_review_path, state)
            self.rebuild_adjudicated_additions(grouped, state)

    def save_review(self, payload: dict[str, Any]) -> None:
        mode = str(payload.get("mode", ""))
        item_id = str(payload.get("item_id", ""))
        if mode not in ANNOTATION_FILES or item_id not in self.documents_by_id:
            raise ValueError("Unknown mode or document")
        complete = bool(payload.get("complete"))
        with self.lock:
            state = self.review_state()
            state.setdefault(mode, {})[item_id] = complete
            atomic_write_json(self.review_path, state)


class AnnotationHandler(BaseHTTPRequestHandler):
    store: AuditStore

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[ui] {self.address_string()} {fmt % args}")

    def send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, error: Exception, status: int) -> None:
        self.send_json({"error": str(error)}, status)

    def serve_static(self, relative: str) -> None:
        relative = relative or "index.html"
        path = (STATIC_DIR / unquote(relative)).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/bootstrap":
                mode = query.get("mode", ["a"])[0]
                self.send_json(self.store.bootstrap(mode))
                return
            if parsed.path == "/api/document":
                mode = query.get("mode", ["a"])[0]
                item_id = query.get("item_id", [""])[0]
                self.send_json(self.store.document(mode, item_id))
                return
            relative = parsed.path.lstrip("/")
            self.serve_static(relative)
        except (ValueError, KeyError) as error:
            self.send_error_json(error, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - server safety net
            self.send_error_json(error, HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_payload(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid content length") from error
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self) -> None:
        actions = {
            "/api/candidate": self.store.save_candidate,
            "/api/addition": self.store.save_addition,
            "/api/addition/delete": self.store.delete_addition,
            "/api/addition/decision": self.store.save_addition_decision,
            "/api/model-suggestion": self.store.set_model_suggestion_state,
            "/api/review": self.store.save_review,
        }
        action = actions.get(urlparse(self.path).path)
        if action is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            action(self.read_payload())
            self.send_json({"ok": True})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_error_json(error, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - server safety net
            self.send_error_json(error, HTTPStatus.INTERNAL_SERVER_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument(
        "--suggestions",
        default="",
        help="Optional model suggestions JSON. Defaults to model_suggestions/latest_suggestions.json.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_dir = Path(args.audit_dir)
    if not audit_dir.is_absolute():
        audit_dir = Path.cwd() / audit_dir
    suggestions_path = Path(args.suggestions) if args.suggestions else None
    store = AuditStore(audit_dir, suggestions_path=suggestions_path)
    AnnotationHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"[ready] http://{args.host}:{args.port}")
    print(f"[audit] {audit_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] annotation UI")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
