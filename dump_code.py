import os

def dump_code(startpath, file_out):
    with open(file_out, 'w', encoding='utf-8') as fout:
        for root, dirs, files in os.walk(startpath):
            dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__', 'models', 'data', 'assets', 'outputs']]
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            fout.write(f'{indent}{os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                if not file.endswith(('.png', '.pdf', '.db', '.pyc', '.resolved', '.tex', '.md', '.txt', '.csv', '.lock')):
                    fout.write(f'{subindent}{file}\n')

        fout.write("\n\n" + "="*80 + "\n\n")

        for root, dirs, files in os.walk(startpath):
            dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '__pycache__', 'models', 'data', 'assets', 'outputs']]
            for file in files:
                if not file.endswith(('.png', '.pdf', '.db', '.pyc', '.resolved', '.tex', '.md', '.txt', '.csv', '.lock', '.yaml', '.yml')):
                    filepath = os.path.join(root, file)
                    fout.write(f"--- FILE: {filepath} ---\n")
                    try:
                        with open(filepath, 'r', encoding='utf-8') as fin:
                            fout.write(fin.read())
                    except Exception as e:
                        fout.write(f"Error reading file: {e}\n")
                    fout.write("\n\n")

dump_code('.', 'all_code.txt')
