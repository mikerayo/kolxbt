"""
Script temporal para actualizar imports después de reorganización
"""
import re
from pathlib import Path

# Mapeo de imports antiguos a nuevos
IMPORT_REPLACEMENTS = {
    r'from database import': 'from core.database import',
    r'from wallet_tracker import': 'from core.wallet_tracker import',
    r'from transaction_parser import': 'from core.transaction_parser import',
    r'from feature_engineering import': 'from core.feature_engineering import',
    r'from ml_models import': 'from core.ml_models import',
    r'from analyzer import': 'from core.analyzer import',
    r'from config import': 'from core.config import',
    r'from utils import': 'from core.utils import',
    r'from dexscreener_api import': 'from apis.dexscreener_api import',
    r'from bubblemaps_api import': 'from apis.bubblemaps_api import',
    r'from pumpfun_parser import': 'from apis.pumpfun_parser import',
}

def add_sys_path(content):
    """Añade sys.path si no está presente"""
    if 'sys.path.insert' in content:
        return content

    # Encuentra los imports
    lines = content.split('\n')
    import_end = 0

    # Busca el fin de los imports estándar
    for i, line in enumerate(lines):
        if i > 5 and line and not line.strip().startswith(('import', 'from', '#', '"""', "'''")):
            import_end = i
            break

    # Añade sys.path antes del primer import del proyecto
    for i, line in enumerate(lines):
        if 'from ' in line and any(old in line for old in IMPORT_REPLACEMENTS.keys()):
            # Añade antes de este import
            if 'from pathlib import Path' not in '\n'.join(lines[:i]):
                lines.insert(i, 'from pathlib import Path')
                lines.insert(i+1, '')
                i += 2
            if 'sys.path.insert' not in '\n'.join(lines[:i]):
                lines.insert(i, '# Add parent directory to path for imports')
                lines.insert(i+1, 'sys.path.insert(0, str(Path(__file__).parent.parent))')
                lines.insert(i+2, '')
            break

    return '\n'.join(lines)

def replace_imports(content):
    """Reemplaza imports antiguos con nuevos"""
    for old_pattern, new_import in IMPORT_REPLACEMENTS.items():
        content = re.sub(old_pattern, new_import, content)
    return content

def fix_file(filepath):
    """Fix imports in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has sys.path
        if 'sys.path.insert' in content and 'Path(__file__).parent.parent' in content:
            return False

        # Add sys.path
        content = add_sys_path(content)

        # Replace imports
        content = replace_imports(content)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

# Fix all Python files in subdirectories
base_dir = Path('.')

for subdir in ['processes', 'discovery', 'updaters', 'dashboard']:
    for py_file in (base_dir / subdir).glob('*.py'):
        if py_file.name != '__init__.py':
            print(f"Fixing {py_file}...")
            fix_file(py_file)

print("\n✓ Imports actualizados!")
