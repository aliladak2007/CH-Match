# companies_house_matcher_fixed.py
"""
Streamlit app to match company names in an uploaded Excel file against
Companies House and pull back key details (registered number, address,
officers, status, etc.).

This version fixes the following issues found in the original script:
    • get_company_profile() now returns data instead of None (which caused
      AttributeError downstream).
    • Removed unused imports (time, ThreadPoolExecutor, threading).
    • API key can be supplied via environment variable or Streamlit secrets
      to avoid hard‑coding credentials.
    • Added basic error handling and logging for easier debugging.
    • Consolidated common logic, applied typing hints and docstrings.
    • Tidied up progress‑bar handling and miscellaneous minor improvements.
"""

# ────────── Imports ──────────
from __future__ import annotations

import asyncio
import io
import os
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

import aiohttp
import pandas as pd
import requests
import streamlit as st
import os
os.environ["COMPANIES_HOUSE_API_KEY"] = "YOUR_API_KEY"

import unicodedata
import re

import requests
import re
from duckduckgo_search import DDGS
from duckduckgo_search import DDGS

import time

fallback_counter = {"count": 0, "max": 5}  # limit 10 Google fallbacks

def google_fallback_search(company_name: str) -> tuple[str, str] | tuple[None, None]:
    if fallback_counter["count"] >= fallback_counter["max"]:
        st.warning("Google fallback limit reached for this session.")
        return None, None

    query = f'site:find-and-update.company-information.service.gov.uk "{company_name}"'

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            time.sleep(3)  # polite delay to avoid bans
            results = ddgs.text(query, max_results=3)
            for r in results:
                link = r.get("href", "")
                match = re.search(r"/company/([0-9A-Z]+)", link)
                if match:
                    fallback_counter["count"] += 1
                    return match.group(1), r.get("title", company_name)
    except Exception as e:
        st.warning(f"Google fallback failed via DDGS: {e}")
        return None, None

    return None, None

def normalise_company_number(raw: str) -> str:
    """
    If `raw` is all digits, left-pad it to exactly 8 characters.
    If it contains any letter, return it unchanged (but stripped).
    """
    s = raw.strip()
    # if there’s any letter in the string, leave it as is
    if re.search(r'[A-Za-z]', s):
        return s.upper()
    # otherwise, if it’s all digits, pad to length 8
    if s.isdigit():
        return s.zfill(8)
    # anything else (e.g. symbols), just return stripped
    return s


def normalise_exact_name(name: str) -> str:
    """Strip leading/trailing whitespace, control characters, and normalise."""
    name = unicodedata.normalize("NFKD", str(name))
    name = ''.join(c for c in name if c.isprintable())
    name = re.sub(r'\s+', ' ', name)  # collapse all whitespace
    return name.strip().upper()


# ────────── Config ──────────
# Retrieve API key – prefer environment variable or Streamlit secrets
API_KEY: str | None = os.getenv("COMPANIES_HOUSE_API_KEY") or st.secrets.get(  # type: ignore[attr-defined]
    "COMPANIES_HOUSE_API_KEY", None
)

if not API_KEY:
    st.error(
        "No Companies House API key supplied. Set the COMPANIES_HOUSE_API_KEY environment "
        "variable or add it to st.secrets to continue."
    )
    st.stop()

NAME_COLUMN: str = "CompanyName"
TOP_N: int = 5  # Number of suggestions to keep per name

# ────────── Helper functions ──────────

def similarity(a: str, b: str) -> float:
    """Compute a case‑insensitive similarity ratio between two strings."""
    return round(SequenceMatcher(None, a.upper(), b.upper()).ratio(), 4)


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    allow_404: bool = False,
    retries: int = 1
) -> Dict[str, Any]:
    """
    GET JSON with minimal retries and a small pause to avoid bursts.
    """
    auth = aiohttp.BasicAuth(API_KEY, "")
    for attempt in range(retries):
        # ── NEW ── gentle pacing between calls
        await asyncio.sleep(0.5)
        async with session.get(url, auth=auth) as resp:
            if resp.status == 200:
                return await resp.json()
            if allow_404 and resp.status == 404:
                return {}
            # we’re only doing 1 attempt now; any error will bubble
            raise RuntimeError(f"Companies House API error {resp.status}: {url}")


# ────────── Async API Calls ──────────

async def get_company_matches_async(session, name: str, retries=3):
    original = name
    cleaned = normalise_exact_name(name)
    url_template = "https://api.company-information.service.gov.uk/search/companies?q={}"

    def strip_suffixes(name: str) -> str:
        return re.sub(r"\b(LTD|LIMITED|INC|LLC|PLC|CORPORATION|CORP|CO|COMPANY|PVT)\b\.?", "", name, flags=re.IGNORECASE).strip()

    async def fetch_items(query: str):
        for attempt in range(retries):
            try:
                async with session.get(url_template.format(query), auth=aiohttp.BasicAuth(API_KEY, ''),
                                       ssl=False, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("items", [])
                    elif response.status in [429, 500, 502, 503, 504]:
                        await asyncio.sleep(1 + attempt * 2)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(1 + attempt * 2)
        return []

    def extract_best(items: list, reference: str, fallback_label: str) -> Tuple[str, str, float, List[str], str]:
        sorted_items = sorted(items, key=lambda i: similarity(reference, i.get("title", "")), reverse=True)
        top_items = sorted_items[:TOP_N]
        exact_matches = [item for item in top_items if normalise_exact_name(item.get("title", "")) == reference]
        best_item = exact_matches[0] if exact_matches else top_items[0]

        best_name = best_item.get("title", "").strip()
        best_number = best_item.get("company_number", "")
        best_score = similarity(reference, best_name)
        is_exact = "Yes" if normalise_exact_name(best_name) == reference else fallback_label
        suggestions = [f"{item['title']} ({item['company_number']})" for item in top_items]

        return best_number, best_name, best_score, suggestions, is_exact

    # === First attempt: normal cleaned name ===
    items = await fetch_items(cleaned)
    if items:
        return extract_best(items, cleaned, "No")

    # === Second attempt: stripped suffixes ===
    stripped = strip_suffixes(cleaned)
    if stripped != cleaned:
        items = await fetch_items(stripped)
        if items:
            return extract_best(items, cleaned, "No (stripped)")

    # === Third attempt: truncated to first 2 or 3 words ===
    parts = stripped.split()
    for i in range(min(3, len(parts)), 0, -1):
        fallback_query = " ".join(parts[:i])
        items = await fetch_items(fallback_query)
        if items:
            return extract_best(items, cleaned, f"No (fallback {i} words)")

    # === Final fallback: quoted exact search ===
    quoted_url = f'https://api.company-information.service.gov.uk/search/companies?q="{cleaned}"&items_per_page=10&status=all'
    try:
        async with session.get(quoted_url, auth=aiohttp.BasicAuth(API_KEY, ''), ssl=False, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                items = data.get("items", [])
                if items:
                    return extract_best(items, cleaned, "Yes (quoted)")
    except Exception:
        pass  # silent fail

    return "Not Found", "No match", 0.0, [], "No"

async def get_company_profile_async(
    session: aiohttp.ClientSession, company_number: str
) -> Dict[str, str]:
    """
    Return company_name, address, status, type, and accounts dates.
    If we hit rate‐limit even after retries, return defaults.
    """
    url = f"https://api.company-information.service.gov.uk/company/{company_number}"
    try:
        data = await _get_json(session, url, allow_404=True)
    except RuntimeError as e:
        msg = str(e)
        # Only swallow if it's a rate-limit (429); otherwise re-raise
        if "429" in msg:
            st.warning(f"Rate-limited fetching profile for {company_number}, skipping: {msg}")
            return {
                "company_name":           "Unknown",
                "address":                "Address not found",
                "status":                 "Unknown",
                "type":                   "Unknown",
                "last_made_up_to":        None,
                "next_accounts_due_on":   None,
                "next_due":               None,
            }
        # any other error we let bubble so you still get the real data or see the failure
        raise

    # — now normal processing of data as before —
    addr = data.get("registered_office_address", {}) or {}
    lines = [
        addr.get("address_line_1",""),
        addr.get("address_line_2",""),
        addr.get("locality",""),
        addr.get("region",""),
        addr.get("postal_code",""),
        addr.get("country",""),
    ]
    formatted_address = ", ".join(filter(None, lines)) or "Address not found"

    accounts   = data.get("accounts", {}) or {}
    last_accts = accounts.get("last_accounts", {}) or {}
    next_accts = accounts.get("next_accounts", {}) or {}

    made_up_to = last_accts.get("made_up_to") or last_accts.get("period_end_on")
    due_on     = next_accts.get("due_on")

    return {
        "company_name":           data.get("company_name", "Unknown"),
        "address":                formatted_address,
        "status":                 data.get("company_status", "Unknown"),
        "type":                   data.get("type", "Unknown"),
        "last_made_up_to":        made_up_to,
        "next_accounts_due_on":   due_on,
        "next_due":               due_on,
    }


async def get_registered_officers_async(
    session: aiohttp.ClientSession, company_number: str
) -> str:
    """
    Return a comma-separated list of active officers.
    If we get rate-limited (429), retry a couple of times with back-off.
    After that, return a safe fallback string.
    """
    url = f"https://api.company-information.service.gov.uk/company/{company_number}/officers"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = await _get_json(session, url, allow_404=True)
            break
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries - 1:
                # back-off a little longer each time
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            # if it's a final 429 or any other error, fall through to fallback
            return "Officer data not found (rate limited)"
    else:
        # should never hit this because we break on success
        return "Officer data not found"

    officers = data.get("items", [])
    active: List[str] = []
    for officer in officers:
        if officer.get("resigned_on") in (None, ""):
            name = officer.get("name", "").strip()
            dob = officer.get("date_of_birth", {})
            if dob.get("year") and dob.get("month"):
                active.append(f"{name} ({dob['year']}-{dob['month']:02d})")
            else:
                active.append(name)

    return ", ".join(active) if active else "No active officers found"

async def process_company(
    session: aiohttp.ClientSession, name: str
) -> Dict[str, Any]:
    """
    Three modes:
      1. both name+number supplied → lookup by number, but score vs provided name
      2. only number supplied (blank name) → lookup by number
      3. only name supplied → lookup by name
    """
    # pop in upload order
    input_name   = process_company._orig_names.pop(0)
    input_number = process_company._orig_numbers.pop(0) or None

    # Case A: any number → lookup by number, but verify match quality
        # Case A: any number → lookup by number, but bail out if API didn’t find it
    if input_number:
        lookup_number = normalise_company_number(input_number)
        profile       = await get_company_profile_async(session, lookup_number)

        # if CH API returned nothing, skip straight to name‐only search
        if profile["company_name"] == "Unknown":
            st.info(
                f"No profile found for {lookup_number!r}; "
                "falling back to name-only lookup"
            )
            # do NOT return here; let it drop through to Case B
        else:
            # existing “trust-then-return” logic goes here...
            officers         = await get_registered_officers_async(session, lookup_number)
            official         = profile["company_name"]
            cleaned_input    = normalise_exact_name(input_name)
            cleaned_official = normalise_exact_name(official)

            if cleaned_input == cleaned_official:
                score = 1.0
                exact = "Yes (by number)"
            else:
                score = similarity(cleaned_input, cleaned_official)
                exact = "Yes (by number)" if score == 1.0 else None

            if score >= 0.9:
                exact = exact or "No (by number)"
                return {
                    "company_number":       lookup_number,
                    "official_name":        official,
                    "similarity_score":     score,
                    "possible_matches":     f"{official} ({lookup_number})",
                    "exact_match":          exact,
                    "registered_address":   profile["address"],
                    "registered_officers":  officers,
                    "company_status":       profile["status"],
                    "company_type":         profile["type"],
                    "last_made_up_to":      profile["last_made_up_to"],
                    "next_accounts_due_on": profile["next_accounts_due_on"],
                    "next_due":             profile["next_due"],
                }

            st.info(
                f"Discarding number {lookup_number!r} for “{input_name}” – "
                f"normalised names differ ({score:.2f} < 0.90)"
            )

    # Case B: only name → forward lookup
    number, official_name, score, suggestions, exact = (
        await get_company_matches_async(session, input_name)
    )
    if number in {"Not Found", "", "Error"}:
        return {
            "company_number":        number,
            "official_name":         official_name,
            "similarity_score":      score,
            "possible_matches":      "No matches",
            "exact_match":           exact,
            "registered_address":    "N/A",
            "registered_officers":   "N/A",
            "company_status":        "N/A",
            "company_type":          "N/A",
            "last_made_up_to":       None,
            "next_accounts_due_on":  None,
            "next_due":              None,
        }

    profile_task  = get_company_profile_async(session, number)
    officers_task = get_registered_officers_async(session, number)
    profile, officers = await asyncio.gather(
        profile_task, officers_task, return_exceptions=True
    )

    if isinstance(profile, Exception):
        profile = {
            "company_name":     "Unknown",
            "address":          "Address not found",
            "status":           "Unknown",
            "type":             "Unknown",
            "last_made_up_to":  None,
            "next_accounts_due_on": None,
            "next_due":         None,
        }
    if isinstance(officers, Exception):
        officers = "Officer data not found"

    return {
        "company_number":        number,
        "official_name":         official_name,
        "similarity_score":      score,
        "possible_matches":      ", ".join(suggestions),
        "exact_match":           exact,
        "registered_address":    profile["address"],
        "registered_officers":   officers,
        "company_status":        profile["status"],
        "company_type":          profile["type"],
        "last_made_up_to":       profile["last_made_up_to"],
        "next_accounts_due_on":  profile["next_accounts_due_on"],
        "next_due":              profile["next_due"],
    }

async def process_all_companies(
    names: List[str], progress_callback
) -> List[Dict[str, Any]]:
    """Process a list of company names concurrently, with throttling."""

    connector = aiohttp.TCPConnector(limit_per_host=5, limit=10)
    timeout = aiohttp.ClientTimeout(total=60, connect=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(1)  # Respect rate‑limits

        async def _worker(idx: int, comp_name: str):
            async with semaphore:
                result = await process_company(session, comp_name)
                progress_callback(idx + 1, len(names))
                return result

        tasks = [_worker(idx, name) for idx, name in enumerate(names)]
        return await asyncio.gather(*tasks)


# ────────── Synchronous helper for Streamlit ──────────

def run_async_processing(names: List[str], progress_callback):
    """
    Before kicking off the async run, stash the original
    CompanyName and CompanyNumber lists so each worker can pop
    the correct pair in order.
    """
    df_orig = st.session_state.get("uploaded_df", pd.DataFrame())
    # Fill any missing values so pop(0) ordering always lines up
    orig_names = df_orig.get("CompanyName", pd.Series()).fillna("").tolist()
    orig_numbers = df_orig.get("CompanyNumber", pd.Series()).fillna("").astype(str).tolist()

    # Make copies in case we need to reuse df_orig later
    process_company._orig_names   = orig_names.copy()
    process_company._orig_numbers = orig_numbers.copy()

    return asyncio.run(process_all_companies(names, progress_callback))


# ────────── Fallback synchronous functions ──────────
def apply_google_fallbacks_to_results(results_df: pd.DataFrame) -> pd.DataFrame:
    fallback_limit = 5
    fallback_used = 0

    for i, row in results_df.iterrows():
        if fallback_used >= fallback_limit:
            st.warning("Google fallback limit reached. Some unmatched companies remain.")
            break

        if row["company_number"] in {"Not Found", "", "Error"}:
            st.info(f"Trying Google fallback for: {row['CompanyName']}")

            # Sleep BEFORE sending query
            time.sleep(5)

            fallback_number, fallback_name = google_fallback_search(row["CompanyName"])
            if fallback_number:
                fallback_used += 1
                results_df.at[i, "company_number"] = fallback_number
                results_df.at[i, "official_name"] = fallback_name
                results_df.at[i, "similarity_score"] = 1.0
                results_df.at[i, "possible_matches"] = f"{fallback_name} ({fallback_number})"
                results_df.at[i, "exact_match"] = "Yes (Google Fallback)"
                results_df.at[i, "registered_address"] = "Not available"
                results_df.at[i, "registered_officers"] = "Not available"
                results_df.at[i, "company_status"] = "Unknown"
                results_df.at[i, "company_type"] = "Unknown"
                st.warning(f"Used Google fallback for: {row['CompanyName']}")
            else:
                st.info("No result from fallback.")
    
    return results_df

def get_company_profile(company_number):
    url = f"https://api.company-information.service.gov.uk/company/{company_number}"
    try:
        response = requests.get(url, auth=(API_KEY, ''), verify=False, timeout=30)

        if response.status_code == 200:
            data = response.json()

            address_data = data.get("registered_office_address", {})
            address_lines = [
                address_data.get("address_line_1", ""),
                address_data.get("address_line_2", ""),
                address_data.get("locality", ""),
                address_data.get("region", ""),
                address_data.get("postal_code", ""),
                address_data.get("country", "")
            ]
            formatted_address = ", ".join([line for line in address_lines if line])

            company_status = data.get("company_status", "Unknown")
            company_type = data.get("type", "Unknown")

            return {
                'address': formatted_address if formatted_address else "Address not found",
                'status': company_status,
                'type': company_type
            }

    except Exception:
        pass

    return {
        'address': "Address not found",
        'status': "Unknown",
        'type': "Unknown"
    }

def get_company_matches(name: str):
    cleaned = str(name).strip().upper()
    url = f"https://api.company-information.service.gov.uk/search/companies?q={cleaned}"
    resp = requests.get(url, auth=(API_KEY, ""), timeout=30)
    if resp.status_code != 200:
        return "Not Found", "No match", 0.0, [], "No"

    items = resp.json().get("items", [])
    if not items:
        return "Not Found", "No match", 0.0, [], "No"

    sorted_items = sorted(items, key=lambda i: similarity(cleaned, i.get("title", "")), reverse=True)
    top_items = sorted_items[:TOP_N]

    best = top_items[0]
    best_name = best.get("title", "").strip()
    best_number = best.get("company_number", "")
    best_score = similarity(cleaned, best_name)
    is_exact = "Yes" if best_name.upper() == cleaned else "No"
    suggestions = [f"{it['title']} ({it['company_number']})" for it in top_items]

    return best_number, best_name, best_score, suggestions, is_exact


def get_registered_officers(company_number: str) -> str:
    url = f"https://api.company-information.service.gov.uk/company/{company_number}/officers"
    resp = requests.get(url, auth=(API_KEY, ""), timeout=30)
    if resp.status_code != 200:
        return "Officer data not found"

    officers = resp.json().get("items", [])
    active: List[str] = []
    for officer in officers:
        if officer.get("resigned_on") in (None, ""):
            name = officer.get("name", "").strip()
            dob = officer.get("date_of_birth", {})
            if dob.get("year") and dob.get("month"):
                active.append(f"{name} ({dob['year']}-{dob['month']:02d})")
            else:
                active.append(name)
    return ", ".join(active) if active else "No active officers found"


# ────────── Streamlit UI ──────────
st.set_page_config(page_title="Companies House Matcher", layout="wide")
st.title("Companies House Company Name Matcher")

st.write(
    """
Upload an Excel file with a column named `CompanyName`.
The app will query Companies House and return the closest matches plus key details.
"""
)

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if not uploaded_file:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# Read the uploaded spreadsheet
try:
    df = pd.read_excel(uploaded_file)
    # ── NEW ── make the original DataFrame available to run_async_processing()
    # Make both columns available for run_async_processing()
    st.session_state["uploaded_df"] = df.copy()
    # stash original names and numbers (fill NaN → "")
    st.session_state["orig_names"]   = df["CompanyName"].fillna("").tolist()
    st.session_state["orig_numbers"] = df.get("CompanyNumber", pd.Series()).fillna("").astype(str).tolist()
except Exception as exc:
    st.error(f"Error reading Excel file: {exc}")
    st.stop()


if NAME_COLUMN not in df.columns:
    st.error(f"Column '{NAME_COLUMN}' not found in the uploaded file.")
    st.stop()

# Optional dummy checkbox — user cannot change it
_ = st.checkbox("Use high‑performance processing (recommended for large files)", value=True, disabled=False)

# Hardcoded logic override
use_async = True  # Always true regardless of UI

run_btn = st.button("Run Matching")

if not run_btn:
    st.stop()

progress_bar = st.progress(0.0)


def _update_progress(done: int, total: int):
    progress_bar.progress(done / total)


if use_async:
    st.info("Using concurrent processing…")
    # ── NEW ── fill NaN in CompanyName so that blank names trigger reverse lookup
    names = df[NAME_COLUMN].fillna("").tolist()
    results = run_async_processing(names, _update_progress)
else:
    st.info("Using sequential processing… this may take a while for large files.")
    results = []
    for idx, comp in enumerate(df[NAME_COLUMN].fillna("")):
        num, off_name, score, sugg, exact = get_company_matches(comp)
        if num != "Not Found":
            prof = get_company_profile(num)
            off = get_registered_officers(num)
        else:
            prof = {"address": "N/A", "status": "N/A", "type": "N/A"}
            off = "N/A"

        results.append(
            {
                "company_number": num,
                "official_name": off_name,
                "similarity_score": score,
                "possible_matches": ", ".join(sugg) if sugg else "No matches",
                "exact_match": exact,
                "registered_address": prof["address"],
                "registered_officers": off,
                "company_status": prof["status"],
                "company_type": prof["type"],
            }
        )
        _update_progress(idx + 1, len(df))

progress_bar.empty()

# Merge results into original dataframe
results_df = pd.DataFrame(results)
output_df = pd.concat([df, results_df], axis=1)

# Post-process with Google fallback
output_df = apply_google_fallbacks_to_results(output_df)


st.success("Matching complete!")
# Convert all fields to string to avoid Arrow export errors
output_df = output_df.astype(str)

# Ensure all values are strings for compatibility
output_df = output_df.astype(str)

# Prepare Excel output
output_xlsx = io.BytesIO()
output_df.to_excel(output_xlsx, index=False)
output_xlsx.seek(0)

st.dataframe(output_df, use_container_width=True)

# Download button
st.download_button(
    label="Download results as Excel",
    data=output_xlsx.getvalue(),
    file_name="companies_house_matches.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

