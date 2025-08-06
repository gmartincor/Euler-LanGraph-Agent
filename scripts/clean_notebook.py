import json
import sys
import argparse
from pathlib import Path

def clean_notebook(notebook_path: Path) -> None:
    """Clean a notebook file in place, removing outputs and execution metadata."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Error reading {notebook_path}: {e}")
        sys.exit(1)
    
    # Clean each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            
            # Clean execution metadata
            if 'metadata' in cell:
                cell['metadata'].pop('execution', None)
                cell['metadata'].pop('scrolled', None)
    
    # Clean notebook-level metadata
    if 'metadata' in notebook:
        notebook['metadata'].pop('language_info', None)
        
        # Keep only essential kernelspec info
        if 'kernelspec' in notebook['metadata']:
            kernelspec = notebook['metadata']['kernelspec']
            essential_kernelspec = {
                k: v for k, v in kernelspec.items() 
                if k in ['display_name', 'language', 'name']
            }
            notebook['metadata']['kernelspec'] = essential_kernelspec
    
    # Write back the cleaned notebook
    try:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✅ Cleaned {notebook_path}")
    except Exception as e:
        print(f"❌ Error writing {notebook_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Clean Jupyter notebook outputs and metadata')
    parser.add_argument('notebook', help='Path to the notebook file')
    
    args = parser.parse_args()
    notebook_path = Path(args.notebook)
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        sys.exit(1)
    
    if not notebook_path.suffix == '.ipynb':
        print(f"❌ File is not a notebook: {notebook_path}")
        sys.exit(1)
    
    clean_notebook(notebook_path)

if __name__ == '__main__':
    main()
