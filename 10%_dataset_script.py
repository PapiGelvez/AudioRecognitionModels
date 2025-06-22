# Script para reorganizar el 10% de los archivos de audio desde el dataset original a uno con la siguiente estructura:
# SmallDataset, con carpetas 'Real' y 'Fake', y nombres de archivo únicos.
# Proyecto/
# ├── AUDIO/
# │   ├── FinalDataset_16khz/
# │   │   ├── Real/
# │   │   └── CycleGAN/, Diff/, etc.
# │   └── SmallDataset/  ← will be created
# │       ├── Real/      ← will be created
# │       └── Fake/      ← will be created
# └── 10%_dataset_script.py

import shutil
import random
from pathlib import Path


def sample_wavs_flat(src_dirs, dst_dir, sample_ratio=0.1):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    all_wavs = []
    for src in src_dirs:
        src_path = Path(src)
        all_wavs.extend(src_path.rglob("*.wav"))

    # Sampleo del 10%
    sample_size = int(len(all_wavs) * sample_ratio)
    sampled_files = random.sample(all_wavs, sample_size)

    for src_path in sampled_files:
        new_filename = "_".join(src_path.parts[-4:])
        dst_path = dst_dir / new_filename
        shutil.copy2(src_path, dst_path)


project_root = Path(__file__).resolve().parent

base_path = project_root / "AUDIO" / "FinalDataset_16khz"
destiny_path = project_root / "AUDIO" / "SmallDataset"

# Carpeta 'Real': 1 carpeta de origen
sample_wavs_flat([base_path / "Real"], destiny_path / "Real", sample_ratio=0.1)

# Carpeta 'Fake': 5 carpetas de origen
fake_sources = ["CycleGAN", "Diff", "StarGAN", "TTS", "TTS-VC"]
fake_paths = [base_path / folder for folder in fake_sources]
sample_wavs_flat(fake_paths, destiny_path / "Fake", sample_ratio=0.1)
