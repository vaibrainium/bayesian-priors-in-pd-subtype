"""Fill the subject report template (subject_report.html) with the per-subject data from subject_profiles.py."""

import json
from datetime import date
from pathlib import Path

from src.data_io import catalog

TEMPLATE = Path(__file__).with_name("subject_report.html")


def write_report(subjects, filename, min_rt_ms, subtype_bins, moca_cutoff, root="artifacts"):
    """Write the interactive report to <root>/<filename>; `subjects` are `subject_profiles.report_entry` dicts."""
    data = {"generated": date.today().isoformat(), "min_rt_ms": min_rt_ms, "subtype_bins": subtype_bins, "moca_cutoff": moca_cutoff, "subjects": subjects}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")  # safe inside a <script> element
    html = TEMPLATE.read_text().replace("/*__REPORT_DATA__*/null", payload)
    path = catalog.output_path(filename, root=root)
    catalog.write(html, path)
    print(f"saved {filename}")
    return path
