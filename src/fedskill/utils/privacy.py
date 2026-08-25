"""Privacy sanitizer — entity scrubbing + lineage sanitization for shared skills.

Implements the three privacy mechanisms from §3.7:
  Mechanism 2: Entity scrubbing filter (regex + NER + blocklist)
  Mechanism 3: Lineage sanitization (strip provenance metadata)
"""

from __future__ import annotations

import re
from copy import deepcopy
from collections.abc import Callable, Iterable

from fedskill.models.skill import Skill

# ---------------------------------------------------------------------------
# Entity patterns (6 families)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("FILE_PATH", re.compile(
        r'(?<!\w)(?:/[\w.\-]+){2,}|(?:[A-Z]:\\[\w.\-\\]+)', re.IGNORECASE
    ), 0),
    ("URL", re.compile(
        r'https?://[^\s\'"<>)\],}]+', re.IGNORECASE
    ), 0),
    ("IP_ADDR", re.compile(
        r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b'
    ), 0),
    ("API_KEY", re.compile(
        r'(?:sk-[\w\-]{20,}|(?:api[_-]?key|token|bearer|password|secret)\s*[=:]\s*["\']?[\w\-]{16,})',
        re.IGNORECASE,
    ), 0),
    ("EMAIL", re.compile(
        r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'
    ), 0),
    ("USERNAME", re.compile(
        r'(?:user(?:name)?|login)\s*[=:]\s*["\']?[\w.\-]+',
        re.IGNORECASE,
    ), 0),
    ("PHONE", re.compile(
        r'\b(?:\+?1[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    ), 0),
    ("PERSON", re.compile(
        r'\b(?:customer|passenger|name|Mr\.|Mrs\.|Ms\.)\s+'
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
    ), 1),
]

_KEY_TYPES = {
    "first_name": "PERSON", "last_name": "PERSON", "full_name": "PERSON",
    "name": "PERSON", "email": "EMAIL", "email_address": "EMAIL",
    "phone": "PHONE", "phone_number": "PHONE", "mobile": "PHONE",
    "address": "ADDRESS", "address1": "ADDRESS", "address2": "ADDRESS",
    "street": "ADDRESS", "city": "LOCATION", "state": "LOCATION",
    "zip": "POSTAL_CODE", "zipcode": "POSTAL_CODE", "postal_code": "POSTAL_CODE",
    "country": "LOCATION", "dob": "DATE_OF_BIRTH", "date_of_birth": "DATE_OF_BIRTH",
    "user_id": "USER_ID", "customer_id": "CUSTOMER_ID", "account_id": "ACCOUNT_ID",
    "order_id": "ORDER_ID", "reservation_id": "RESERVATION_ID",
    "booking_id": "RESERVATION_ID", "ticket_id": "TICKET_ID",
    "transaction_id": "TRANSACTION_ID", "tracking_id": "TRACKING_ID",
    "payment_method_id": "PAYMENT_ID", "card_id": "PAYMENT_ID",
    "credit_card": "PAYMENT_ID", "username": "USERNAME", "login": "USERNAME",
    "api_key": "API_KEY", "token": "API_KEY", "bearer": "API_KEY",
    "password": "PASSWORD", "secret": "API_KEY", "url": "URL", "uri": "URL",
    "ip": "IP_ADDR", "ip_address": "IP_ADDR", "file_path": "FILE_PATH",
    "path": "FILE_PATH",
}
_KEY_ALTERNATION = "|".join(
    sorted((re.escape(key) for key in _KEY_TYPES), key=len, reverse=True)
)
_QUOTED_KV = re.compile(
    rf'(?P<keyquote>["\']?)(?P<key>{_KEY_ALTERNATION})(?P=keyquote)'
    rf'\s*[=:]\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
    re.IGNORECASE,
)
_UNQUOTED_KV = re.compile(
    rf'(?P<keyquote>["\']?)(?P<key>{_KEY_ALTERNATION})(?P=keyquote)'
    rf'\s*[=:]\s*(?P<value>[A-Za-z0-9_#@./:\\-]+)',
    re.IGNORECASE,
)

EntityDetector = Callable[[str], Iterable[tuple[int, int, str]]]

# Paths that should NOT be redacted (common system paths)
_SAFE_PATHS = {
    "/bin/bash", "/bin/sh", "/usr/bin/env", "/dev/null", "/tmp",
    "/etc/hosts", "/usr/local", "/home", "/root",
}


def _replace_group(match: re.Match, label: str, group: int | str = 0) -> str:
    value = match.group(group).strip().rstrip(".,;:)")
    if label == "FILE_PATH" and value in _SAFE_PATHS:
        return match.group(0)
    if len(value) <= 2:
        return match.group(0)
    start, end = match.span(group)
    relative_start = start - match.start()
    relative_end = end - match.start()
    return match.group(0)[:relative_start] + f"[{label}]" + match.group(0)[relative_end:]


def _apply_external_detector(text: str, entity_detector: EntityDetector | None) -> str:
    if entity_detector is None:
        return text
    spans = sorted(entity_detector(text), key=lambda span: (span[0], span[1]), reverse=True)
    result = text
    last_start = len(text)
    for start, end, label in spans:
        if not (0 <= start < end <= len(text)) or end > last_start:
            continue
        result = result[:start] + f"[{label}]" + result[end:]
        last_start = start
    return result


def _scrub_text(
    text: str,
    blocklist: set[str] | None = None,
    entity_detector: EntityDetector | None = None,
) -> str:
    """Replace sensitive entities in text with typed placeholders."""
    if not text:
        return text

    result = _apply_external_detector(text, entity_detector)

    # Structured values are replaced while their schema keys remain intact.
    for pattern in (_QUOTED_KV, _UNQUOTED_KV):
        result = pattern.sub(
            lambda match: _replace_group(
                match, _KEY_TYPES[match.group("key").lower()], "value"
            ),
            result,
        )

    for label, pattern, group in _PATTERNS:
        result = pattern.sub(
            lambda match, entity_type=label, capture=group: _replace_group(
                match, entity_type, capture
            ),
            result,
        )

    # Stage 2: Blocklist
    if blocklist:
        for term in blocklist:
            if len(term) >= 3:
                result = re.sub(
                    re.escape(term), "[REDACTED]", result, flags=re.IGNORECASE
                )

    return result


def scrub_skill(
    skill: Skill,
    blocklist: set[str] | None = None,
    entity_detector: EntityDetector | None = None,
) -> Skill:
    """Apply entity scrubbing to a skill's content fields.

    Returns a new Skill with sensitive entities replaced by placeholders.
    The original skill is not modified.
    """
    s = deepcopy(skill)
    s.name = _scrub_text(s.name, blocklist, entity_detector)
    s.description = _scrub_text(s.description, blocklist, entity_detector)
    s.when_to_use = _scrub_text(s.when_to_use, blocklist, entity_detector)
    s.procedure = _scrub_text(s.procedure, blocklist, entity_detector)
    s.examples = [_scrub_text(ex, blocklist, entity_detector) for ex in s.examples]
    for script in s.scripts:
        if "code" in script:
            script["code"] = _scrub_text(script["code"], blocklist, entity_detector)
        if "description" in script:
            script["description"] = _scrub_text(
                script["description"], blocklist, entity_detector
            )
    return s


def sanitize_lineage(skill: Skill) -> Skill:
    """Strip provenance metadata from a skill before sharing.

    Removes source_task, parent_ids, and resets tracking counters
    so the receiving backend cannot trace the skill's origin.
    """
    s = deepcopy(skill)
    s.source_task = None
    s.parent_ids = []
    s.inject_count = 0
    s.positive_count = 0
    s.negative_count = 0
    s.effectiveness = 0.0
    return s


def prepare_for_sharing(
    skill: Skill,
    blocklist: set[str] | None = None,
    min_level: int = 1,
    entity_detector: EntityDetector | None = None,
) -> Skill | None:
    """Full privacy pipeline: check level → scrub entities → sanitize lineage.

    Returns None if the skill's level is below min_level (i.e., should not
    be shared across the federation boundary).
    """
    if skill.level < min_level:
        return None
    scrubbed = scrub_skill(skill, blocklist, entity_detector)
    sanitized = sanitize_lineage(scrubbed)
    return sanitized
