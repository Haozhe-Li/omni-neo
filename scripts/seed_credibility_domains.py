"""
Manually seed domain-level `credibility` verdicts into Redis, bypassing the
LLM classifier for these specific domains from now on.

This is the patch half of the "arguable" label: `core/utils/source_credibility.py`
now teaches the classifier to recognize domains with a documented, independent
reliability problem, but that only catches a domain the first time an LLM call
happens to see it. Entries below take effect immediately, for every request,
without waiting on that. They land in the exact same cache key
(`omni:credibility:domain:<host>`) and TTL (`TTL_DOMAIN_VERDICT`, 365 days)
that classify_sources() itself writes to on an LLM "trusted"/"arguable"
verdict — this script is just a manual write to that same cache, not a
separate mechanism.

    python3 scripts/seed_credibility_domains.py
    python3 scripts/seed_credibility_domains.py --dry-run

Requires REDIS_URL, loaded from .env as main.py does. Every entry's `reason`
must cite a specific, checkable reliability problem — never a political
characterization — see the module docstring in source_credibility.py for why.
"""

import asyncio
import json
import sys

import dotenv

dotenv.load_dotenv()

from core.utils.redis_credibility import TTL_DOMAIN_VERDICT, credibility_redis  # noqa: E402

# domain -> one-sentence reason. Keep citing a specific, independently
# checkable reliability problem (fact-checker ratings, confirmed platform
# takedowns for coordinated inauthentic behavior, documented fabrication) —
# never a viewpoint or political alignment. See source_credibility.py's
# `_SYSTEM_PROMPT` for the same standard applied to the LLM classifier.
_ARGUABLE_DOMAINS: dict[str, str] = {
    "theepochtimes.com": (
        "The Epoch Times has been rated very low for factual reliability by "
        "independent media-monitoring organizations (e.g. NewsGuard), was "
        "found by Facebook to be running a large coordinated inauthentic-"
        "behavior ad network (banned in 2019), and has been documented by "
        "multiple news investigations amplifying QAnon conspiracy content "
        "and 2020 US election misinformation."
    ),
    "epochtimes.com": (
        "Epoch Times' Chinese/other-language edition; shares the same "
        "editorial operation and documented reliability record as "
        "theepochtimes.com (low independent fact-reliability ratings, a "
        "2019 Facebook takedown for coordinated inauthentic behavior, and "
        "documented misinformation amplification)."
    ),
    "epochtimes.com.tw": (
        "Epoch Times' Taiwan/Traditional-Chinese edition; shares the same "
        "editorial operation and documented reliability record as "
        "theepochtimes.com (low independent fact-reliability ratings, a "
        "2019 Facebook takedown for coordinated inauthentic behavior, and "
        "documented misinformation amplification)."
    ),
    "ntd.com": (
        "New Tang Dynasty Television is a sister outlet to The Epoch Times "
        "under the same media organization, sharing its editorial staff and "
        "the same documented pattern of low independent reliability ratings "
        "and misinformation amplification."
    ),
    "ntdtv.com": (
        "Legacy domain for New Tang Dynasty Television; same organization "
        "and documented reliability record as ntd.com."
    ),
    "renminbao.com": (
        "Renminbao (People's Report), a Falun Gong-affiliated advocacy "
        "outlet, frequently publishes news that has not been independently "
        "verified or fact-checked."
    ),
}


async def main(dry_run: bool) -> int:
    entries = {
        domain: json.dumps({"label": "arguable", "reason": reason})
        for domain, reason in _ARGUABLE_DOMAINS.items()
    }

    print(f"{'Would write' if dry_run else 'Writing'} {len(entries)} domain(s), "
          f"ttl={TTL_DOMAIN_VERDICT}s (~{TTL_DOMAIN_VERDICT // 86400}d):")
    for domain, reason in _ARGUABLE_DOMAINS.items():
        print(f"  {domain}: {reason}")

    if not dry_run:
        await credibility_redis.set_many(entries, TTL_DOMAIN_VERDICT)
        print("done.")
    else:
        print("(dry run — nothing was written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(dry_run="--dry-run" in sys.argv)))
