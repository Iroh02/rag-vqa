"""Delete the backup-divider slide and everything after it (until end of deck)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "presentation" / "rag_vqa_open_book.html"

html = HTML.read_text(encoding="utf-8")

# Find the backup divider start
m = re.search(r"<!-- ═{50,}\n\s+BACKUP DIVIDER", html)
if not m:
    print("No backup divider found — already stripped or never present.")
else:
    end_marker = '</div><!-- end deck -->'
    end_idx = html.index(end_marker)
    stripped = html[:m.start()] + html[end_idx:]
    HTML.write_text(stripped, encoding="utf-8")
    print(f"Stripped {(end_idx - m.start())} chars of backup slides.")
