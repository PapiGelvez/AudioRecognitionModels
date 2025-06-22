# Script para reorganizar todos los archivos de audio desde el dataset original a uno con la siguiente estructura:
# FullDataset, con carpetas 'Real' y 'Fake', y nombres de archivo únicos.
# Proyecto/
# ├── AUDIO/
# │   ├── FinalDataset_16khz/
# │   │   ├── Real/
# │   │   └── CycleGAN/, Diff/, etc.
# │   └── FullDataset/  ← will be created
# │       ├── Real/      ← will be created
# │       └── Fake/      ← will be created
# └── full_dataset_script.py

import shutil
from pathlib import Path


def copy_all_wavs_flat(src_dirs, dst_dir):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    all_wavs = []
    for src in src_dirs:
        src_path = Path(src)
        all_wavs.extend(src_path.rglob("*.wav"))

    for src_path in all_wavs:
        new_filename = "_".join(src_path.parts[-4:])
        dst_path = dst_dir / new_filename
        shutil.copy2(src_path, dst_path)


project_root = Path(__file__).resolve().parent

base_path = project_root / "AUDIO" / "FinalDataset_16khz"
destiny_path = project_root / "AUDIO" / "FullDataset"

# Copy all 'Real' files
copy_all_wavs_flat([base_path / "Real"], destiny_path / "Real")

# Copy all 'Fake' files from the 5 subfolders
fake_sources = ["CycleGAN", "Diff", "StarGAN", "TTS", "TTS-VC"]
fake_paths = [base_path / folder for folder in fake_sources]
copy_all_wavs_flat(fake_paths, destiny_path / "Fake")
