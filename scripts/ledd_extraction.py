"""Levodopa Equivalent Daily Dose (LEDD) extraction from the free-text `pd_meds` field.

Run:  .venv/bin/python scripts/ledd_extraction.py

Conversion factors follow Tomlinson et al. (2010), Mov Disord 25:2649-2653.
Output is a per-patient AUDIT TABLE intended for manual clinical verification, not a
finished variable. Rows flagged NO_SCHEDULE cannot be computed: the source records the
drug and tablet strength but no dose frequency, and daily dose is not inferable from
tablet strength alone.
"""
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROC = "/mnt/pd-data/processed"

# Tomlinson et al. 2010 LED conversion factors (mg drug -> mg levodopa equivalent)
FACTORS = {
    "levodopa_ir": 1.00,
    "levodopa_cr": 0.75,
    "rasagiline": 100.0,
    "selegiline_oral": 10.0,
    "rotigotine": 30.0,
    "pramipexole": 100.0,
    "ropinirole": 20.0,
    "amantadine": 1.00,
    "apomorphine": 10.0,
    "entacapone": None,      # adds 0.33 x total levodopa, handled separately
}
NO_LED = ["gabapentin", "artane", "trihexyphenidyl"]   # not dopaminergic

WORD_NUM = {"once": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _clean(s):
    s = str(s)
    s = s.encode("latin-1", "ignore").decode("utf-8", "ignore") if "\x83" in s else s
    return re.sub(r"[^\x20-\x7e]+", " ", s).strip()


def _freq(seg):
    """Doses per day from a schedule fragment; None if not stated."""
    seg = seg.lower()
    if re.search(r"night|bedtime|qhs|daily at bed", seg):
        if not re.search(r"\btimes\b", seg):
            return 1.0
    m = re.search(r"(\d+)\s*\(?\w*\)?\s*times?\s+(?:a\s+)?(?:day|daily)", seg)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(once|one|two|three|four|five|six)\s+times?\s+(?:a\s+)?(?:day|daily)", seg)
    if m:
        return float(WORD_NUM[m.group(1)])
    if re.search(r"\bbid\b", seg):
        return 2.0
    if re.search(r"\btid\b", seg):
        return 3.0
    if re.search(r"\bqid\b", seg):
        return 4.0
    if re.search(r"\b(qd|daily|per day)\b", seg):
        return 1.0
    return None


def _units(seg):
    """Tablets/capsules per dose; defaults to 1 only when a schedule exists."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:tab|tablet|cap|capsule)", seg.lower())
    return float(m.group(1)) if m else None


def parse(text):
    """-> (ledd, list of component dicts, flag set)

    Segments are scanned in order. A segment naming a drug opens a product slot; a
    segment carrying only a schedule fills the most recent UNFILLED slot. Slots are
    never overwritten, so a dose can never be silently dropped -- an unfilled slot
    at the end is reported as NO_SCHEDULE.
    """
    txt = _clean(text)
    segs = [s for s in re.split(r";|\r\n|\n", txt) if s.strip()]
    DRUGS = [("levodopa_ir", r"levodopa|sinemet"), ("rasagiline", r"rasagiline|azilect"),
             ("rotigotine", r"rotigotine|neupro"), ("pramipexole", r"pramipexole|mirapex"),
             ("ropinirole", r"ropinirole|requip"), ("amantadine", r"amantadine"),
             ("apomorphine", r"apomorphine"), ("selegiline_oral", r"selegiline"),
             ("entacapone", r"entacapone|comtan")]
    slots, flags = [], set()

    for seg in segs:
        low = seg.lower()
        if any(k in low for k in NO_LED):
            slots.append(dict(drug="(non-dopaminergic)", raw=seg.strip(), led=0.0, filled=True))
            continue

        kind = next((k for k, pat in DRUGS if re.search(pat, low)), None)
        f, un = _freq(seg), _units(seg)

        if kind is not None:
            if kind == "levodopa_ir" and re.search(r"\bcr\b", low):
                kind = "levodopa_cr"
            mg = None
            mm = (re.search(r"\d+\s*-\s*(\d+)\s*mg", low) if kind.startswith("levodopa")
                  else re.search(r"(\d+(?:\.\d+)?)\s*mg", low))
            if mm:
                mg = float(mm.group(1))
            elif kind.startswith("levodopa"):
                mg = 100.0
            slot = dict(drug=kind, raw=seg.strip(), mg=mg, units=un, freq=f, filled=False)
            if mg is None:
                flags.add("NO_STRENGTH")
            # MAOI / patch products are inherently once-daily
            if f is None and kind in ("rasagiline", "rotigotine", "selegiline_oral"):
                slot["freq"], slot["assumed_qd"] = 1.0, True
            if slot["freq"] is not None:
                slot["filled"] = True
            slots.append(slot)
            continue

        # schedule-only segment -> fill the most recent unfilled slot
        if f is not None:
            tgt = next((s_ for s_ in reversed(slots) if not s_["filled"]), None)
            if tgt is not None:
                tgt["freq"] = f
                if un is not None:
                    tgt["units"] = un
                tgt["filled"] = True
                tgt["raw"] += " | " + seg.strip()
            continue

    comps, ld_total = [], 0.0
    for s_ in slots:
        if s_["drug"] == "(non-dopaminergic)":
            comps.append(dict(drug=s_["drug"], raw=s_["raw"], led=0.0)); continue
        if s_["drug"] == "entacapone":
            flags.add("ENTACAPONE")
            comps.append(dict(drug=s_["drug"], raw=s_["raw"], led=np.nan)); continue
        if not s_["filled"] or s_.get("mg") is None:
            flags.add("NO_SCHEDULE" if not s_["filled"] else "NO_STRENGTH")
            comps.append(dict(drug=s_["drug"], raw=s_["raw"], mg=s_.get("mg"), led=np.nan)); continue
        units = s_["units"] if s_["units"] is not None else 1.0
        led = s_["mg"] * units * s_["freq"] * FACTORS[s_["drug"]]
        if s_["drug"].startswith("levodopa"):
            ld_total += led
        if s_.get("assumed_qd"):
            flags.add("ASSUMED_QD")
        comps.append(dict(drug=s_["drug"], raw=s_["raw"], mg=s_["mg"],
                          units=units, freq=s_["freq"], led=led))

    if "ENTACAPONE" in flags and ld_total:
        comps.append(dict(drug="entacapone_adj", raw="0.33 x total levodopa",
                          led=0.33 * ld_total))

    bad = {"NO_SCHEDULE", "NO_STRENGTH", "ENTACAPONE"} & flags
    total = np.nansum([c.get("led", np.nan) for c in comps]) if comps else np.nan
    return (np.nan if bad else total), comps, flags


def main():
    m = pd.read_csv(f"{PROC}/processed_metadata_all_data_accu_60.csv")
    u = m[m.pd_meds.notna()].drop_duplicates("subject_id").set_index("subject_id")
    rows = []
    for sid, r in u.iterrows():
        led, comps, flags = parse(r.pd_meds)
        rows.append(dict(subject_id=sid, LEDD=led,
                         flags=",".join(sorted(flags)) or "OK",
                         subtype=r.get("trem_vs_brady_type"),
                         detail=" + ".join(
                             f"{c['drug']}"
                             + (f" {c.get('mg')}mg x{c.get('units')} x{c.get('freq')}/d"
                                if c.get("freq") else "")
                             + (f" = {c['led']:.0f}" if not np.isnan(c.get("led", np.nan)) else " = ?")
                             for c in comps)))
    R = pd.DataFrame(rows).sort_values(["flags", "subject_id"])
    print("=" * 100)
    print("LEDD AUDIT TABLE  (Tomlinson et al. 2010 factors) -- VERIFY MANUALLY BEFORE USE")
    print("=" * 100)
    print(R.to_string(index=False, max_colwidth=68))
    ok = R[R.LEDD.notna()]
    print(f"\ncomputable: {len(ok)} / {len(R)} patients")
    if len(ok):
        print(f"LEDD range {ok.LEDD.min():.0f}-{ok.LEDD.max():.0f}, "
              f"median {ok.LEDD.median():.0f} mg/day")
        print("\nby subtype:")
        print(ok.groupby("subtype").LEDD.agg(["count", "median"]).to_string())
    print("\nNOT computable = drug and tablet strength recorded but NO dose frequency.")
    print("Daily dose is not inferable from tablet strength; these need source records.")
    R.to_csv(f"{PROC}/ledd_audit.csv", index=False)
    print(f"\nwritten: {PROC}/ledd_audit.csv")


if __name__ == "__main__":
    main()
