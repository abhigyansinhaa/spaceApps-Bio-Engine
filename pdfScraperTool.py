import pandas as pd
import requests
from urllib.parse import urlparse
import os
import time
from pathlib import Path
import re

def extract_pmc_id(url):
    """Extract PMC ID from the URL"""
    # Pattern to match PMC URLs and extract PMC ID
    match = re.search(r'/PMC(\d+)', url)
    if match:
        return match.group(1)
    return None

def get_pdf_url_from_landing_page(landing_url):
    """
    Convert PMC landing page URL to PDF URL
    Example: https://pmc.ncbi.nlm.nih.gov/articles/PMC4136787/ 
    -> https://pmc.ncbi.nlm.nih.gov/articles/PMC4136787/pdf/pone.0104830.pdf
    """
    try:
        # Get the landing page to extract the actual PDF filename
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(landing_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Look for PDF link in the HTML
        html_content = response.text
        
        # Look for PDF download links more specifically
        # Pattern 1: Look for links that contain "pdf" and end with .pdf
        pdf_patterns = [
            r'href="(/pmc/articles/[^"]*\.pdf)"',  # Relative PDF links
            r'href="(https://[^"]*\.pdf)"',        # Absolute PDF links
            r'href="([^"]*\.pdf)"'                 # Any PDF links
        ]
        
        for pattern in pdf_patterns:
            pdf_matches = re.findall(pattern, html_content, re.IGNORECASE)
            if pdf_matches:
                pdf_link = pdf_matches[0]
                # If it's a relative URL, make it absolute
                if pdf_link.startswith('/'):
                    pdf_link = 'https://pmc.ncbi.nlm.nih.gov' + pdf_link
                elif not pdf_link.startswith('http'):
                    # If it's just a filename like "pdf/pone.0104830.pdf", construct full URL
                    base_url = landing_url.rstrip('/')
                    pdf_link = f"{base_url}/{pdf_link}"
                print(f"Found PDF link in HTML: {pdf_link}")
                return pdf_link
        
        # If no PDF link found in HTML, try to construct it manually
        # Extract PMC ID and try common patterns
        pmc_match = re.search(r'/PMC(\d+)', landing_url)
        if pmc_match:
            pmc_id = pmc_match.group(1)
            base_url = landing_url.rstrip('/')
            
            # Common PDF filename patterns for PMC
            possible_pdf_urls = [
                f"{base_url}/pdf/",  # Try the PDF directory first
                f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/pdf/"
            ]
            
            # For each base URL, we'll try to find the actual PDF
            for base_pdf_url in possible_pdf_urls:
                try:
                    # Try to access the PDF directory page
                    pdf_dir_response = requests.get(base_pdf_url, headers=headers, timeout=10)
                    if pdf_dir_response.status_code == 200:
                        # Look for .pdf files in the directory listing or redirects
                        if 'pdf' in pdf_dir_response.headers.get('content-type', '').lower():
                            return base_pdf_url  # This IS the PDF
                        
                        # Check if it redirects to a PDF
                        if pdf_dir_response.url.endswith('.pdf'):
                            return pdf_dir_response.url
                            
                except:
                    continue
            
            print(f"Could not find PDF URL for PMC{pmc_id}")
            return None
        else:
            print(f"Could not extract PMC ID from: {landing_url}")
            return None
            
    except Exception as e:
        print(f"Error processing {landing_url}: {e}")
        return None

def try_alternative_pdf_download(landing_url, filename, download_folder):
    """Try alternative methods to get the PDF"""
    try:
        # Method 1: Try to get PDF through the PMC API or different endpoints
        pmc_match = re.search(r'/PMC(\d+)', landing_url)
        if not pmc_match:
            return False
        
        pmc_id = pmc_match.group(1)
        
        # Alternative PDF URLs to try
        alt_urls = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/",
            f"https://europepmc.org/articles/PMC{pmc_id}?pdf=render",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/{pmc_id}.pdf",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        for alt_url in alt_urls:
            print(f"Trying alternative URL: {alt_url}")
            try:
                response = requests.get(alt_url, headers=headers, timeout=30, allow_redirects=True)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' in content_type and len(response.content) > 10000:
                        filepath = os.path.join(download_folder, filename)
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        # Verify it's a real PDF
                        with open(filepath, 'rb') as f:
                            if f.read(4) == b'%PDF':
                                print(f"✓ Alternative download successful: {filename}")
                                return True
                        os.remove(filepath)
            except:
                continue
        
        return False
        
    except Exception as e:
        print(f"Alternative download failed: {e}")
        return False
    """Try different PDF URL patterns until we find one that works"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Extract PMC ID
    pmc_match = re.search(r'/PMC(\d+)', base_landing_url)
    if not pmc_match:
        return None
    
    pmc_id = pmc_match.group(1)
    base_url = base_landing_url.rstrip('/')
    
    # Try different PDF URL patterns
    pdf_url_patterns = [
        f"{base_url}/pdf/",  # Just the PDF directory
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/",  # Alternative base URL
        f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmc_id}/pdf/"  # Clean URL
    ]
    
    for pdf_url in pdf_url_patterns:
        try:
            # Try a HEAD request first to check if URL exists
            head_response = requests.head(pdf_url, headers=headers, timeout=10, allow_redirects=True)
            
            if head_response.status_code == 200:
                content_type = head_response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    print(f"Found working PDF URL: {head_response.url}")
                    return head_response.url
                    
        except requests.exceptions.RequestException:
            continue
    
    return None

def download_pdf(pdf_url, filename, download_folder):
    """Download PDF from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': pdf_url.replace('/pdf/', '/'),  # Set referer to the article page
        }
        
        print(f"Attempting to download: {pdf_url}")
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        # Debug: Check response details
        content_type = response.headers.get('content-type', '').lower()
        content_length = response.headers.get('content-length', 'unknown')
        print(f"Response content-type: {content_type}")
        print(f"Response content-length: {content_length}")
        print(f"Response status: {response.status_code}")
        print(f"Final URL after redirects: {response.url}")
        
        # Check if we got redirected to a different page (like login/paywall)
        if response.url != pdf_url and not response.url.endswith('.pdf'):
            print(f"✗ Got redirected to non-PDF URL: {response.url}")
            return False
        
        # Check if the response is actually a PDF
        if 'pdf' not in content_type and 'octet-stream' not in content_type:
            print(f"✗ Warning: {pdf_url} returned content-type: {content_type}")
            # Let's check the first few bytes to see what we actually got
            content_start = response.content[:200] if hasattr(response, 'content') else b''
            print(f"Content starts with: {content_start[:50]}")
            if b'<!DOCTYPE html' in content_start or b'<html' in content_start:
                print("✗ Received HTML instead of PDF (probably an error page)")
                return False
        
        filepath = os.path.join(download_folder, filename)
        total_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    total_size += len(chunk)
        
        # Verify the downloaded file
        file_size = os.path.getsize(filepath)
        print(f"Downloaded {file_size} bytes to {filename}")
        
        # Check if file is too small (likely an error page)
        if file_size < 10000:  # Less than 10KB is suspicious for a PDF
            # Read the first few bytes to check if it's actually a PDF
            with open(filepath, 'rb') as f:
                file_start = f.read(100)
                if not file_start.startswith(b'%PDF'):
                    print(f"✗ File doesn't start with PDF header. First bytes: {file_start[:50]}")
                    os.remove(filepath)  # Remove the invalid file
                    return False
                elif file_size < 5000:  # Even with PDF header, too small
                    print(f"✗ PDF file too small ({file_size} bytes), likely corrupted")
                    os.remove(filepath)
                    return False
        
        print(f"✓ Successfully downloaded: {filename} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {pdf_url}: {e}")
        return False

def main():
    # Configuration
    CSV_FILE = "SB_publication_PMC.csv"  # Replace with your CSV filename
    DOWNLOAD_FOLDER = "sample_publications"
    MAX_DOWNLOADS = 50
    DELAY_BETWEEN_REQUESTS = 2  # seconds
    
    # Create download folder
    Path(DOWNLOAD_FOLDER).mkdir(exist_ok=True)
    
    # Read CSV file
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Assuming your CSV has columns 'Title' and 'Link'
        # Adjust column names if different
        if 'Title' not in df.columns or 'Link' not in df.columns:
            print("Expected columns 'Title' and 'Link' not found.")
            print(f"Available columns: {list(df.columns)}")
            return
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Process first 50 rows
    successful_downloads = 0
    failed_downloads = 0
    
    for index, row in df.head(MAX_DOWNLOADS).iterrows():
        title = str(row['Title']).strip()
        link = str(row['Link']).strip()
        
        print(f"\nProcessing {index + 1}/{min(MAX_DOWNLOADS, len(df))}: {title[:50]}...")
        
        # Skip if link is invalid
        if not link or link == 'nan' or not link.startswith('http'):
            print(f"✗ Skipping invalid link: {link}")
            failed_downloads += 1
            continue
        
        # Get PDF URL
        pdf_url = get_pdf_url_from_landing_page(link)
        
        # If the first method didn't work, try the systematic approach
        if not pdf_url:
            print("Trying systematic PDF URL discovery...")
            pdf_url = find_working_pdf_url(link)
        
        if not pdf_url:
            print(f"✗ Could not find PDF URL for: {link}")
            failed_downloads += 1
            continue
        
        print(f"Found PDF URL: {pdf_url}")
        
        # Create safe filename
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:100]  # Limit length
        filename = f"{safe_title}.pdf"
        
        # Download PDF
        if download_pdf(pdf_url, filename, DOWNLOAD_FOLDER):
            successful_downloads += 1
        else:
            # Try alternative download methods
            print("Trying alternative download methods...")
            if try_alternative_pdf_download(link, filename, DOWNLOAD_FOLDER):
                successful_downloads += 1
            else:
                failed_downloads += 1
        
        # Add delay to be respectful to the server
        if index < MAX_DOWNLOADS - 1:  # Don't delay after the last request
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n=== Download Summary ===")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Total processed: {successful_downloads + failed_downloads}")
    print(f"PDFs saved to: {os.path.abspath(DOWNLOAD_FOLDER)}")

if __name__ == "__main__":
    main()