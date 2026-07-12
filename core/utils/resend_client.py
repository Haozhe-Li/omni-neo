"""Send the scheduled-task report notification email via Resend.

HTML body mirrors the site's own design tokens (app/globals.css in the
Next.js repo — teal accent #20B2AA, warm off-white background, 0.75rem
radius) so the email doesn't feel like a different product from the app it
came from. The full report itself still lives on its Pages URL (full
markdown, charts, everything already using that UI) — this email is a
notification card (title, summary, a CTA button), not a copy of the report;
the plain-text fallback carries the full body for text-only clients.
"""

from __future__ import annotations

import html as html_escape
import logging
import os
from datetime import datetime, timezone

import resend

logger = logging.getLogger(__name__)

resend.api_key = os.getenv("RESEND_API_KEY", "")

# Resend's shared sandbox sender — works without verifying a domain first.
# Switch to a verified address (e.g. "Omni Knows <noreply@omniknows.xyz>") by
# setting RESEND_FROM_EMAIL once a sending domain is verified in Resend.
_FROM = os.getenv("RESEND_FROM_EMAIL", "Omni Knows <onboarding@resend.dev>")

_SITE_URL = "https://omniknows.xyz"
_LOGO_URL = f"{_SITE_URL}/android-chrome-512x512.png"

# Colors lifted straight from app/globals.css :root (light theme — email
# clients that honor prefers-color-scheme get the .dark equivalents via the
# media query below; everyone else gets this as a sane, on-brand default).
_BG = "#f3f3ee"
_CARD = "#ffffff"
_FOREGROUND = "#1a1a1a"
_MUTED = "#6b6b6b"
_BORDER = "rgba(0,0,0,0.08)"
_ACCENT = "#20B2AA"

_BG_DARK = "#191a1a"
_CARD_DARK = "#222323"
_FOREGROUND_DARK = "#ffffff"
_MUTED_DARK = "#8b8b8b"
_BORDER_DARK = "rgba(255,255,255,0.08)"


def _build_html(title: str, summary: str, page_url: str) -> str:
    safe_title = html_escape.escape(title)
    safe_summary = html_escape.escape(summary).replace("\n", "<br>")
    year = datetime.now(timezone.utc).year

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
  @media (prefers-color-scheme: dark) {{
    .omni-bg {{ background-color: {_BG_DARK} !important; }}
    .omni-card {{ background-color: {_CARD_DARK} !important; border-color: {_BORDER_DARK} !important; }}
    .omni-fg {{ color: {_FOREGROUND_DARK} !important; }}
    .omni-muted {{ color: {_MUTED_DARK} !important; }}
  }}
</style>
</head>
<body class="omni-bg" style="margin:0;padding:0;background-color:{_BG};font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" class="omni-bg" style="background-color:{_BG};">
    <tr>
      <td align="center" style="padding:40px 20px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="max-width:560px;">
          <tr>
            <td style="padding-bottom:28px;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="vertical-align:middle;padding-right:8px;">
                    <img src="{_LOGO_URL}" width="24" height="24" alt="" style="display:block;border-radius:6px;">
                  </td>
                  <td class="omni-fg" style="vertical-align:middle;font-size:15px;font-weight:600;color:{_FOREGROUND};letter-spacing:-0.01em;">
                    Omni Knows
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td class="omni-card" style="background-color:{_CARD};border:1px solid {_BORDER};border-radius:16px;padding:36px 32px;">
              <p style="margin:0 0 10px 0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:{_ACCENT};">
                Scheduled Report
              </p>
              <h1 class="omni-fg" style="margin:0 0 16px 0;font-size:22px;line-height:1.3;font-weight:600;color:{_FOREGROUND};">
                {safe_title}
              </h1>
              <p class="omni-muted" style="margin:0 0 28px 0;font-size:14px;line-height:1.7;color:{_MUTED};">
                {safe_summary}
              </p>
              <table role="presentation" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="border-radius:10px;background-color:{_ACCENT};">
                    <a href="{page_url}" target="_blank" style="display:inline-block;padding:11px 22px;font-size:14px;font-weight:600;color:#ffffff;text-decoration:none;border-radius:10px;">
                      View Full Report &rarr;
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td style="padding:28px 4px 0 4px;">
              <p class="omni-muted" style="margin:0;font-size:12px;line-height:1.6;color:{_MUTED};">
                You're receiving this because you set up a scheduled research task on Omni Knows.<br>
                &copy; {year} <a href="{_SITE_URL}" style="color:{_MUTED};text-decoration:underline;">Omni Knows</a>. All rights reserved.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def send_report_email(
    to_email: str,
    title: str,
    summary: str,
    report_markdown: str,
    page_url: str | None = None,
) -> bool:
    """Send the report notification. HTML is the on-brand card + button most
    clients render; the plain-text part (Resend sends both — a standard
    multipart email) carries the full report body for text-only clients.
    Never raises — failure here shouldn't blow up a run that already
    succeeded; it stays inspectable via scheduled_task_runs regardless."""
    text_parts = [summary.strip(), ""]
    if page_url:
        text_parts += [f"View the full report: {page_url}", ""]
    text_parts += ["---", "", report_markdown.strip()]

    try:
        payload = {
            "from": _FROM,
            "to": [to_email],
            "subject": f"Your scheduled report: {title}",
            "text": "\n".join(text_parts),
        }
        if page_url:
            payload["html"] = _build_html(title, summary.strip(), page_url)
        resend.Emails.send(payload)
        return True
    except Exception as exc:
        logger.error(f"[resend_client] failed to send report email to {to_email!r}: {exc}")
        return False
