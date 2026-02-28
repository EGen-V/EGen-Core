import os
import shutil
import re
from pathlib import Path

source_dir = Path('/home/erebus/Documents/EGen-Core/.temp_airllm/air_llm/airllm')
dest_dir = Path('/home/erebus/Documents/EGen-Core/egen_core/egen_core')

if not dest_dir.exists():
    dest_dir.mkdir(parents=True)

# Delete existing contents if any
for item in dest_dir.iterdir():
    if item.is_file():
        item.unlink()
    elif item.is_dir():
        shutil.rmtree(item)

# Walk source and copy/rebrand
for root, dirs, files in os.walk(source_dir):
    rel_path = Path(root).relative_to(source_dir)
    target_root = dest_dir / rel_path
    
    if not target_root.exists():
        target_root.mkdir(parents=True)
        
    for file in files:
        if not file.endswith('.py'):
            continue
            
        source_file = Path(root) / file
        
        # Rename file if it contains airllm
        dest_filename = file.replace('airllm', 'egen_core')
        dest_file = target_root / dest_filename
        
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Rebranding text replacements
        # We need to be careful with case
        content = content.replace('AirLLM', 'EGenCore')
        content = content.replace('AirLlm', 'EGenCore')
        content = content.replace('airllm', 'egen_core')
        content = content.replace('AIRLLM', 'EGEN_CORE')
        
        # Specific rename for base class
        content = content.replace('EGenCoreBaseModel', 'EGenCoreBaseModel')
        
        with open(dest_file, 'w', encoding='utf-8') as f:
            f.write(content)

print("Porting and rebranding complete.")
