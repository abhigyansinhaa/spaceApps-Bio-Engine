import pandas as pd
import requests
import os
import time
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from threading import Lock
from tqdm import tqdm

# ================================
# Utility functions
# ================================

# Thread-safe counters for progress tracking
class ProgressCounter:
    def __init__(self):
        self.lock = Lock()
        self.success = 0
        self.failed = 0
        self.exists = 0
        self.skipped = 0
    
    def increment(self, status):
        with self.lock:
            if status == "success":
                self.success += 1
            elif status == "failed":
                self.failed += 1
            elif status == "exists":
                self.exists += 1
            elif status == "skipped":
                self.skipped += 1
    
    def get_stats(self):
        with self.lock:
            return {
                "success": self.success,
                "failed": self.failed,
                "exists": self.exists,
                "skipped": self.skipped
            }

def extract_pmc_id(url):
    match = re.search(r'/PMC(\d+)', url)
    return match.group(1) if match else None

def safe_request(url, headers, retries=3):
    """Request with retry + exponential backoff"""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30, stream=True)
            return r
        except requests.RequestException:
            if attempt < retries - 1:
                wait = (2 ** attempt) + random.random()
                time.sleep(wait)
    return None

def get_pdf_url_from_landing_page(landing_url):
    """Scrape landing page for PDF links or fallback to known PMC patterns"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/117.0.0.0 Safari/537.36"
    }
    try:
        response = safe_request(landing_url, headers)
        if not response or response.status_code != 200:
            return None
        html_content = response.text

        pdf_patterns = [
            r'href="(/pmc/articles/[^"]*\.pdf)"',
            r'href="(https://[^"]*\.pdf)"',
            r'href="([^"]*\.pdf)"'
        ]

        for pattern in pdf_patterns:
            pdf_matches = re.findall(pattern, html_content, re.IGNORECASE)
            if pdf_matches:
                pdf_link = pdf_matches[0]
                if pdf_link.startswith('/'):
                    pdf_link = 'https://pmc.ncbi.nlm.nih.gov' + pdf_link
                elif not pdf_link.startswith('http'):
                    pdf_link = landing_url.rstrip('/') + '/' + pdf_link
                return pdf_link

        # fallback
        pmc_id = extract_pmc_id(landing_url)
        if pmc_id:
            return f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/pdf/"

        return None
    except Exception:
        return None

def try_alternative_pdf_download(landing_url, filename, download_folder):
    """Try known PMC PDF endpoints"""
    pmc_id = extract_pmc_id(landing_url)
    if not pmc_id:
        return False

    alt_urls = [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/",
        f"https://europepmc.org/articles/PMC{pmc_id}?pdf=render",
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/{pmc_id}.pdf",
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/pdf',
    }

    for alt_url in alt_urls:
        r = safe_request(alt_url, headers)
        if r and r.status_code == 200 and b'%PDF' in r.content[:10]:
            filepath = os.path.join(download_folder, filename)
            try:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
                return True
            except IOError:
                return False
    return False

def download_pdf(pdf_url, filename, download_folder):
    """Download PDF and validate"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = safe_request(pdf_url, headers)
    if not r or r.status_code != 200:
        return False

    filepath = os.path.join(download_folder, filename)
    try:
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

        # Verify PDF
        with open(filepath, 'rb') as f:
            if f.read(4) != b'%PDF':
                os.remove(filepath)
                return False

        return True
    except IOError:
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def process_row(idx, row, url_column, download_folder, progress, pbar):
    """Handle one paper (thread-safe)"""
    try:
        landing_url = row[url_column]
        
        if pd.isna(landing_url) or not landing_url:
            progress.increment("skipped")
            pbar.update(1)
            return {"idx": idx, "url": None, "status": "skipped", "reason": "no url"}

        pmc_id = extract_pmc_id(str(landing_url))
        filename = f"PMC{pmc_id}.pdf" if pmc_id else f"paper_{idx}.pdf"
        filepath = os.path.join(download_folder, filename)

        # Check if already exists
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            progress.increment("exists")
            pbar.set_postfix({"✓": progress.success, "⊙": progress.exists, "✗": progress.failed}, refresh=False)
            pbar.update(1)
            return {"idx": idx, "url": landing_url, "status": "exists", "filename": filename}

        # Try landing page scraping first
        pdf_url = get_pdf_url_from_landing_page(landing_url)
        downloaded = download_pdf(pdf_url, filename, download_folder) if pdf_url else False

        # Try alternatives if failed
        if not downloaded:
            downloaded = try_alternative_pdf_download(landing_url, filename, download_folder)

        if downloaded:
            progress.increment("success")
            pbar.set_postfix({"✓": progress.success, "⊙": progress.exists, "✗": progress.failed}, refresh=False)
            pbar.update(1)
            return {"idx": idx, "url": landing_url, "status": "success", "filename": filename}
        else:
            progress.increment("failed")
            pbar.set_postfix({"✓": progress.success, "⊙": progress.exists, "✗": progress.failed}, refresh=False)
            pbar.update(1)
            return {"idx": idx, "url": landing_url, "status": "failed", "reason": "no valid pdf"}
    
    except Exception as e:
        progress.increment("failed")
        pbar.update(1)
        return {"idx": idx, "status": "error", "reason": str(e)}

# ================================
# Main
# ================================

def main():
    CSV_FILE = "SB_publication_PMC.csv"
    DOWNLOAD_FOLDER = "sample_publications"
    MAX_DOWNLOADS = 608
    WORKERS = 8  # 6-10 recommended for network-bound tasks

    Path(DOWNLOAD_FOLDER).mkdir(exist_ok=True)

    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✓ Loaded {len(df)} entries from CSV\n")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return

    # Detect URL column
    url_column = next((col for col in ['URL', 'url', 'Link', 'link', 'PMC_URL', 'pmc_url'] 
                       if col in df.columns), None)
    if not url_column:
        print(f"Available columns: {df.columns.tolist()}")
        print("✗ Error: Could not find URL column")
        return

    print(f"Using URL column: '{url_column}'")
    print(f"Download folder: '{DOWNLOAD_FOLDER}'")
    print(f"Max downloads: {MAX_DOWNLOADS}")
    print(f"Parallel workers: {WORKERS}")
    print("="*60)
    
    # Limit to MAX_DOWNLOADS
    df_to_process = df.head(MAX_DOWNLOADS)
    progress = ProgressCounter()
    
    start_time = time.time()
    results = []
    
    # Create progress bar
    with tqdm(total=len(df_to_process), 
              desc="Downloading PDFs", 
              unit="file",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(process_row, idx, row, url_column, DOWNLOAD_FOLDER, progress, pbar): idx
                for idx, row in df_to_process.iterrows()
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    progress.increment("failed")
                    pbar.update(1)
                    results.append({"idx": idx, "status": "error", "reason": str(e)})

    elapsed = time.time() - start_time

    # Summary statistics
    success = sum(1 for r in results if r["status"] == "success")
    exists = sum(1 for r in results if r["status"] == "exists")
    failed = [r for r in results if r["status"] == "failed"]
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  Total processed:    {len(results)}")
    print(f"  Newly downloaded:   {success}")
    print(f"  Already existed:    {exists}")
    print(f"  Failed downloads:   {len(failed)}")
    print(f"  Skipped (no URL):   {skipped}")
    print(f"  Errors:             {errors}")
    print(f"  Total available:    {success + exists}")
    print(f"  Success rate:       {(success/(len(results)-skipped-exists)*100) if (len(results)-skipped-exists) > 0 else 0:.1f}%")
    print(f"  Time elapsed:       {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    print(f"  Avg time per file:  {elapsed/len(results):.2f}s")
    print("="*60)

    # Save failed downloads
    if failed:
        failed_file = "failed_downloads.json"
        with open(failed_file, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\n✓ Failed URLs saved to: {failed_file}")
        print(f"  You can retry these by filtering the CSV")

    # Save complete results log
    results_file = "download_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Complete results saved to: {results_file}")

if __name__ == "__main__":
    main()