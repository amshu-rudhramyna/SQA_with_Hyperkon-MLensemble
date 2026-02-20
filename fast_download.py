import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from pathlib import Path

# The eotdl CLI downloads a catalog.v2.parquet file
catalog_path = 'data/raw/HYPERVIEW2/catalog.v2.parquet'
target_dir = Path('data/raw/HYPERVIEW2')

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        return True # Skip
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        # eotdl assets are often just public S3/HTTP links if the metadata is already fetched.
        # We'll try a generic request. If it requires auth, we might need a token. Let's see.
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        return True
    except Exception as e:
        # Keep track if auth is strictly needed for the raw asset HTTP bins
        return False

def main():
    if not os.path.exists(catalog_path):
        print(f"Catalog not found at {catalog_path}. Make sure to run `eotdl datasets get HYPERVIEW2 -v 2` without --assets first.")
        return

    df = pd.read_parquet(catalog_path)
    
    # Extract asset URLs and relative paths from the parquet metadata
    tasks = []
    
    for _, row in df.iterrows():
        # Typically STAC parquet has an 'assets' column (JSON string or dict)
        # We need to parse it to find the actual download URL and the relative file name
        try:
            assets = row['assets']
            if isinstance(assets, str):
                assets_dict = json.loads(assets)
            else:
                assets_dict = assets
                
            # Usually STAC items have one main asset, or multiple.
            # In eotdl, the main data is often under a key like 'data', 'item', or just iterating values.
            for asset_key, asset_info in assets_dict.items():
                if isinstance(asset_info, dict) and 'href' in asset_info:
                    url = asset_info['href']
                    # Reconstruct path based on the STAC feature ID if possible, or from the URL
                    # In eotdl structure, it's usually inside folders matching the collection hierarchy
                    
                    # For simplicity, we just extract the last parts of the URL or use STAC ID
                    # Actually eotdl preserves original folder structure, let's try to extract relative path
                    
                    # The STAC item 'id' usually maps to the filename minus extension in some cases,
                    # but the URL has the real filename.
                    filename = url.split('/')[-1]
                    
                    # Figure out subfolder from ID or row properties
                    subfolder = ""
                    if 'id' in row:
                        if '/' in row['id']:
                            subfolder = os.path.dirname(row['id'])
                            
                    # Alternatively, eotdl STAC structure
                    # We will dump them flat for now into `data/raw/HYPERVIEW2/all_assets/` 
                    # OR try to maintain if we can parse the relative path.
                    # The CLI was downloading to `data/raw/HYPERVIEW2/test/hsi_satellite/...`
                    
                    # We'll just read where the URL points or rely on the item ID.
                    rel_path = row.get('id', filename)
                    if not rel_path.endswith(filename.split('.')[-1]):
                        rel_path += '.' + filename.split('.')[-1]
                        
                    out_path = target_dir / rel_path
                    tasks.append((url, out_path))
        except Exception as e:
            pass # Skip rows that don't match expected STAC asset format

    print(f"Discovered {len(tasks)} assets to download.")
    
    # Filter out already downloaded
    tasks = [t for t in tasks if not os.path.exists(t[1])]
    print(f"Remaining to download: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All downloads complete!")
        return

    # Use ThreadPoolExecutor to blast 50 concurrent requests
    success_count = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_url = {executor.submit(download_file, url, path): (url, path) for url, path in tasks}
        
        with tqdm(total=len(tasks), desc="Downloading Assets") as pbar:
            for future in as_completed(future_to_url):
                if future.result():
                    success_count += 1
                pbar.update(1)
                
    print(f"\nFinished. Successfully fetched {success_count}/{len(tasks)} files.")

if __name__ == "__main__":
    main()
