#!/usr/bin/env python3
"""
Download synthetic unknown faces for negative sampling.
Uses thispersondoesnotexist.com to get AI-generated faces.

Usage:
    python download_negatives.py [count]
    
Example:
    python download_negatives.py 50
"""

import os
import sys
import time
import requests

def download_faces(count=50, output_dir=None):
    """
    Download AI-generated faces from thispersondoesnotexist.com
    
    Args:
        count: Number of faces to download (default 50)
        output_dir: Output directory (default: /app/database/Unknown or ./database/Unknown)
    """
    # Determine output directory
    if output_dir is None:
        if os.path.exists("/app/database"):
            output_dir = "/app/database/Unknown"
        else:
            # Local development
            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "database", "Unknown")
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[DOWNLOAD] Downloading {count} unknown faces to: {output_dir}")
    
    success_count = 0
    failed_count = 0
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for i in range(count):
        try:
            # thispersondoesnotexist.com serves a new face on each request
            response = requests.get(
                "https://thispersondoesnotexist.com",
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200 and len(response.content) > 10000:
                # Save the image
                filepath = os.path.join(output_dir, f"unknown_{i:03d}.jpg")
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                success_count += 1
                print(f"[OK] [{i+1}/{count}] Downloaded: unknown_{i:03d}.jpg ({len(response.content)} bytes)")
            else:
                failed_count += 1
                print(f"[FAIL] [{i+1}/{count}] Failed: status={response.status_code}, size={len(response.content)}")
            
            # Rate limiting - be nice to the server
            time.sleep(0.5)
            
        except requests.exceptions.Timeout:
            failed_count += 1
            print(f"[FAIL] [{i+1}/{count}] Timeout")
        except requests.exceptions.RequestException as e:
            failed_count += 1
            print(f"[FAIL] [{i+1}/{count}] Error: {e}")
        except Exception as e:
            failed_count += 1
            print(f"[FAIL] [{i+1}/{count}] Unexpected error: {e}")
    
    print(f"\n[DONE] Download complete!")
    print(f"   Success: {success_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Location: {output_dir}")
    
    return success_count, failed_count


def main():
    # Parse command line arguments
    count = 50
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print(f"Invalid count: {sys.argv[1]}, using default 50")
    
    download_faces(count)


if __name__ == "__main__":
    main()
