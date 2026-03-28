import numpy as np
import os
from pathlib import Path

def inspect_shapes():
    base_dir = Path('c:/Users/One Page VR/Desktop/hypersoilnet2/data/raw/HYPERVIEW2/train')
    hsi_dir = base_dir / 'hsi_airborne'
    msi_dir = base_dir / 'msi_satellite'
    prisma_dir = base_dir / 'hsi_satellite'

    for name, d in [('HSI Airborne', hsi_dir), ('MSI Satellite', msi_dir), ('PRISMA', prisma_dir)]:
        if d.exists():
            files = list(d.glob('*.npz'))[:5]
            print(f"\n--- {name} ---")
            for f in files:
                data = np.load(f)
                arr = data['data'] if 'data' in data else data['arr_0']
                print(f"{f.name}: shape = {arr.shape}, dtype = {arr.dtype}")
        else:
            print(f"\n--- {name} (NOT FOUND) ---")

if __name__ == '__main__':
    inspect_shapes()
