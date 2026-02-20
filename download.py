import sys
import io
import subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

if __name__ == '__main__':
    print("Starting download through subprocess...")
    process = subprocess.Popen(["eotdl", "datasets", "get", "HYPERVIEW2", "-v", "2", "-p", "data/raw/", "-f", "--assets"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False)
    
    for line in iter(process.stdout.readline, b''):
        # Write bytes as utf-8 safely, ignoring encoding issues
        sys.stdout.write(line.decode('utf-8', errors='replace'))
        sys.stdout.flush()

    process.wait()
    print(f"\nDownload finished with code {process.returncode}")
