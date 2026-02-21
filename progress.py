import os
import time
import sys

def count_files(directory):
    total = 0
    for root, _, files in os.walk(directory):
        total += len(files)
    return total

target_dir = 'data/raw/HYPERVIEW2'
expected_total = 9409

print("Monitoring HYPERVIEW2 dataset download progress...\n")

last_printed_percent = -1

try:
    while True:
        if os.path.exists(target_dir):
            current = count_files(target_dir)
        else:
            current = 0
            
        current = min(current, expected_total)
        percent = (current / expected_total) * 100
        current_int_percent = int(percent)
        
        # Only print if we've crossed a 1% integer threshold
        if current_int_percent > last_printed_percent:
            bar_length = 40
            filled_length = int(bar_length * current // expected_total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% ({current}/{expected_total} files)   ')
            sys.stdout.flush()
            last_printed_percent = current_int_percent
        
        if current >= expected_total:
            print("\n\nDownload Complete!")
            break
            
        time.sleep(2)
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
