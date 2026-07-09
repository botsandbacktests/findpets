"""Match-alert emails.

When a reported sighting matches a lost pet, we notify people so they come back
to the site. Two audiences:

  * the lost pet's OWNER  — "a possible match was spotted near you"
  * the FINDER (reporter) — "the pet you spotted matches a lost pet"

PRIVACY: these emails deliberately contain **no** contact information for the
other party. That keeps the paywall intact — the owner still pays $9.99 on the
site to unlock the finder's contact details, and one person paying never reveals
anything to the other. The emails only carry a match score, a coarse distance,
and a link back to the site.

All sends are best-effort: failures are logged but never raised, so a mail
outage can't break the /api/sightings/match request.
"""
from __future__ import annotations

from .mailer import send_email

# Small reusable footer nudging recipients to whitelist us during domain warm-up.
_SPAM_TIP_HTML = (
    "<p style=\"color:#6b7280;font-size:13px\">Tip: add "
    "<b>noreply@tech956.com</b> to your contacts so FindMyPet alerts land in "
    "your inbox, not spam.</p>"
)


def _distance_phrase(distance_km: float) -> str:
    """Coarse, non-identifying distance text (we never reveal an exact spot)."""
    if distance_km <= 1:
        return "less than 1 km away"
    if distance_km <= 5:
        return "within a few km"
    if distance_km <= 16:
        return "within about 16 km"
    return f"about {round(distance_km)} km away"


def send_owner_alert(
    owner_email: str,
    pet_name: str,
    score_pct: float,
    distance_km: float,
    site_url: str,
    sighting_photo_urls: list[str] | None = None,
) -> bool:
    """Alert the lost pet's owner that a possible match was spotted.

    Contains NO finder contact info — drives the owner back to the site, where
    the existing unlock flow lets them pay to see the finder's details.
    """
    if not owner_email:
        return False

    pet_label = pet_name.strip() or "your pet"
    where = _distance_phrase(distance_km)
    login_url = f"{site_url}/app.html"

    # Embed the finder's actual sighting photo(s) so the owner can SEE it's their
    # pet. These are the spotted-animal images (owner already knows their own
    # pet's photos, so those wouldn't help). Contact info is still NOT included —
    # that stays behind the paywall.
    photo_urls = [u for u in (sighting_photo_urls or []) if u]
    if photo_urls:
        imgs_html = "".join(
            f'<img src="{u}" alt="Spotted animal" '
            f'style="max-width:280px;width:100%;border-radius:10px;'
            f'margin:6px 0;border:1px solid #e5e7eb" />'
            for u in photo_urls
        )
        photo_block_html = (
            f'<p style="margin:14px 0 4px;font-weight:600">Here\'s what was spotted:</p>'
            f'<div>{imgs_html}</div>'
        )
        photo_line_text = (
            "\n\nPhotos of the spotted animal are attached in this email "
            "(view it in HTML to see them):\n"
            + "\n".join(photo_urls)
        )
    else:
        photo_block_html = ""
        photo_line_text = ""

    subject = f"Possible match for {pet_label} spotted nearby"

    text = (
        f"Good news — someone just reported spotting an animal that looks like "
        f"{pet_label}.\n\n"
        f"Match score: {score_pct:.0f}%\n"
        f"Location: {where}"
        f"{photo_line_text}\n\n"
        f"Log in to FindMyPet to see the sighting photo and unlock the finder's "
        f"contact info so you can arrange a reunion:\n{login_url}\n\n"
        f"For your safety and privacy, we don't include the finder's contact "
        f"details in this email.\n\n"
        f"— FindMyPet"
    )
    html = (
        f"<p>Good news — someone just reported spotting an animal that looks like "
        f"<b>{pet_label}</b>.</p>"
        f"<p><b>Match score:</b> {score_pct:.0f}%<br>"
        f"<b>Location:</b> {where}</p>"
        f"{photo_block_html}"
        f"<p><a href=\"{login_url}\">Log in to FindMyPet</a> to see the sighting "
        f"photo and unlock the finder's contact info so you can arrange a reunion.</p>"
        f"<p style=\"color:#6b7280;font-size:13px\">For your safety and privacy, we "
        f"don't include the finder's contact details in this email.</p>"
        f"{_SPAM_TIP_HTML}"
        f"<p>— FindMyPet</p>"
    )
    return send_email(owner_email, subject, text, html)


def send_finder_alert(
    finder_email: str,
    score_pct: float,
    site_url: str,
) -> bool:
    """Thank the finder and let them know their sighting matched a lost pet.

    No owner contact info — just confirmation that the owner has been notified.
    """
    if not finder_email:
        return False

    login_url = f"{site_url}/app.html"
    subject = "The pet you spotted may match a lost pet"

    text = (
        f"Thank you for reporting a sighting on FindMyPet!\n\n"
        f"The animal you spotted is a possible match ({score_pct:.0f}%) for a pet "
        f"reported lost. We've notified the owner so they can follow up.\n\n"
        f"You can check your sighting any time here:\n{login_url}\n\n"
        f"Thanks for helping reunite a pet with its family.\n\n"
        f"— FindMyPet"
    )
    html = (
        f"<p>Thank you for reporting a sighting on <b>FindMyPet</b>!</p>"
        f"<p>The animal you spotted is a possible match "
        f"(<b>{score_pct:.0f}%</b>) for a pet reported lost. We've notified the "
        f"owner so they can follow up.</p>"
        f"<p><a href=\"{login_url}\">View your sighting</a></p>"
        f"<p>Thanks for helping reunite a pet with its family.</p>"
        f"{_SPAM_TIP_HTML}"
        f"<p>— FindMyPet</p>"
    )
    return send_email(finder_email, subject, text, html)
