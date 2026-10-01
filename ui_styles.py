"""Original branding and a single, narrowly scoped layout stylesheet.

Colors, typography, radii, and widget borders belong to config.toml. Native
Streamlit components render every case, document, status, and interactive UI.
"""
from html import escape

LOGO = '<svg viewBox="0 0 48 48" aria-hidden="true"><rect x="7" y="5" width="28" height="34" rx="5" fill="#EEF3E9" stroke="#244C3A" stroke-width="2"/><path d="M14 14h14M14 21h10M14 28h6" stroke="#244C3A" stroke-width="2" stroke-linecap="round"/><path d="M25 26h16v12H32l-6 5v-5h-1z" fill="#FFFCF6" stroke="#244C3A" stroke-width="2" stroke-linejoin="round"/><circle cx="30" cy="32" r="1" fill="#D69A4A"/><circle cx="36" cy="32" r="1" fill="#D69A4A"/></svg>'
MOTIF = '<svg class="cbc-motif" viewBox="0 0 550 140" aria-hidden="true"><path d="M25 119h500" stroke="#DED9CE"/><rect x="56" y="34" width="114" height="83" rx="9" fill="#EEF3E9" stroke="#DCE7D8"/><path d="M77 58h68M77 73h52M77 88h60" stroke="#244C3A" stroke-width="2" stroke-linecap="round"/><path d="M230 52l71-31 71 31zM239 60h125M246 107h110M236 117h130M258 65v36M286 65v36M314 65v36M342 65v36" fill="none" stroke="#244C3A" stroke-width="2"/><path d="M412 43h76v48h-35l-18 16V91h-23z" fill="#FFFCF6" stroke="#DCE7D8" stroke-width="2"/><path d="M427 60h43M427 73h30" stroke="#687068" stroke-width="2"/><circle cx="192" cy="62" r="5" fill="#D69A4A"/><path d="M179 63h38M383 63h19" stroke="#DED9CE" stroke-dasharray="3 4"/></svg>'

CSS = """<style>
/* One page-layout exception: Streamlit has no native main-padding setting. */
.stMainBlockContainer {padding:3.8rem 2rem 1.5rem;max-width:1280px}
.cbc-brand {display:flex;align-items:center;gap:12px;padding:0 0 8px}
.cbc-brand strong {font-size:22px;letter-spacing:-.035em;color:#18382C}
.cbc-brand small {display:block;font-size:12px;color:#687068;margin-top:2px}
.st-key-welcome-story {padding-top:18px}
.st-key-welcome-profile {padding:20px}
.st-key-progress-rail {padding-top:10px}
.st-key-agent-activity {padding-left:16px;border-left:1px solid #DED9CE}
.st-key-agent-activity p {font-size:13px;line-height:1.45}
.st-key-progress-rail p {font-size:13px}
.st-key-case-conversation {padding:0 4px}
.st-key-preparation-summary {padding:18px 22px;background:#EEF3E9;border-left:3px solid #244C3A;margin:12px 0 18px}
@media(max-width:850px) {
 .stMainBlockContainer {padding:2rem 1rem 1rem}
 .cbc-brand strong {font-size:18px}
 .st-key-agent-activity {padding-left:0;border-left:0}
}
</style>"""


def header(subtitle, profile=""):
    """Only the original brand mark uses HTML; dynamic labels stay escaped."""
    return (f'<div class="cbc-brand"><div><strong>Counsel Before Court</strong>'
            f'<small>{escape(subtitle)}</small></div></div>')
