import os
import ast
import re

ROOT_DIR = r"c:\Users\athiy\Downloads\Semester-8\Personal Projects\HERALD"
OUTPUT_FILE = os.path.join(ROOT_DIR, "docs", "herald_documentation.tex")

# Directories to crawl
TARGET_DIRS = ["herald", "scripts", "data", "dashboard", "ml", "models", "assets", "docker", "tests", "outputs", "evidence"]
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".next"}

def escape_latex(text):
    if not text:
        return ""
    # Only escape outside of lstlisting environments
    # For simplicity, we just escape common characters in descriptions.
    chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}'
    }
    # Don't double escape already escaped chars
    for char, escaped in chars.items():
        if char == '\\': continue # handle slash carefully if needed
        text = text.replace(char, escaped)
    text = text.replace('\\', r'\textbackslash{}')
    # revert double escaped backslashes for textbackslash{}
    text = text.replace(r'\textbackslash{}textbackslash{}', r'\textbackslash{}')
    return text

def parse_python_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Could not read file: {e}"

    try:
        tree = ast.parse(content)
    except Exception as e:
        return f"Could not parse AST: {e}"

    docstring = ast.get_docstring(tree) or "No module docstring."
    
    imports = []
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(f"import {name.name}")
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                imports.append(f"from {node.module} import {name.name}")
        elif isinstance(node, ast.ClassDef):
            c_doc = ast.get_docstring(node) or "No class docstring."
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    m_doc = ast.get_docstring(item) or "No method docstring."
                    args = [a.arg for a in item.args.args]
                    methods.append({"name": item.name, "args": args, "doc": m_doc})
            classes.append({"name": node.name, "bases": bases, "doc": c_doc, "methods": methods})
        elif isinstance(node, ast.FunctionDef):
            # Only top-level functions (crude check: if parent is Module)
            pass

    # Top level functions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            f_doc = ast.get_docstring(node) or "No function docstring."
            args = [a.arg for a in node.args.args]
            functions.append({"name": node.name, "args": args, "doc": f_doc})

    out = []
    out.append(f"\\textbf{{Purpose:}} {escape_latex(docstring)}\\\\[1em]")
    
    if imports:
        out.append("\\textbf{Imports:}")
        out.append("\\begin{itemize}")
        for imp in imports:
            out.append(f"\\item \\texttt{{{escape_latex(imp)}}}")
        out.append("\\end{itemize}")

    for c in classes:
        out.append(f"\\subsection{{Class: {escape_latex(c['name'])}}}")
        bases_str = ", ".join(c['bases']) if c['bases'] else "None"
        out.append(f"\\textbf{{Inherits from:}} {escape_latex(bases_str)}\\\\")
        out.append(f"\\textbf{{Description:}} {escape_latex(c['doc'])}\\\\[1em]")
        
        if c['methods']:
            out.append("\\textbf{Methods:}")
            out.append("\\begin{itemize}")
            for m in c['methods']:
                args_str = ", ".join(m['args'])
                out.append(f"\\item \\texttt{{{escape_latex(m['name'])}({escape_latex(args_str)})}}: {escape_latex(m['doc'])}")
            out.append("\\end{itemize}")

    for f in functions:
        out.append(f"\\subsection{{Function: {escape_latex(f['name'])}}}")
        args_str = ", ".join(f['args'])
        out.append(f"\\textbf{{Signature:}} \\texttt{{{escape_latex(f['name'])}({escape_latex(args_str)})}}\\\\")
        out.append(f"\\textbf{{Description:}} {escape_latex(f['doc'])}\\\\[1em]")

    return "\n".join(out)

def generate_docs():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Preamble
        f.write("\\documentclass[12pt,a4paper]{report}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\usepackage{hyperref}\n")
        f.write("\\usepackage{listings}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage{fancyhdr}\n")
        f.write("\\usepackage{titlesec}\n")
        f.write("\\usepackage{tocloft}\n")
        f.write("\\usepackage{longtable}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage[T1]{fontenc}\n\n")
        f.write("\\title{HERALD Project Documentation}\n")
        f.write("\\author{Auto-Generated}\n")
        f.write("\\date{\\today}\n\n")
        f.write("\\begin{document}\n")
        f.write("\\maketitle\n")
        f.write("\\tableofcontents\n")
        
        # Project Overview
        f.write("\\chapter{Project Overview}\n")
        f.write("HERALD is a real-time Phishing Intelligence Platform designed to ingest, analyze, and detect malicious domains using a combination of lexical ML models, network enrichment, and visual analysis.\n")
        
        # Crawl directories
        for target in TARGET_DIRS:
            target_path = os.path.join(ROOT_DIR, target)
            if not os.path.exists(target_path): continue
            
            f.write(f"\\chapter{{{escape_latex(target.capitalize())} Directory}}\n")
            
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".dist-info")]
                for file in files:
                    if file.endswith(".pyc") or file.endswith(".joblib") or file.endswith(".png") or file.endswith(".pdf"):
                        continue
                    
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, ROOT_DIR)
                    f.write(f"\\section{{{escape_latex(rel_path)}}}\n")
                    
                    if file.endswith(".py"):
                        f.write(parse_python_file(filepath))
                    else:
                        f.write(f"\\textbf{{Type:}} Non-Python file ({escape_latex(file)})\n")
        
        # Appendix for Evidence
        f.write("\\chapter{Appendix}\n")
        evidence_path = os.path.join(ROOT_DIR, "evidence")
        if os.path.exists(evidence_path):
            f.write("\\section{Evidence Domains}\n")
            f.write("\\begin{itemize}\n")
            for item in os.listdir(evidence_path):
                f.write(f"\\item {escape_latex(item)}\n")
            f.write("\\end{itemize}\n")
            
        f.write("\\end{document}\n")

if __name__ == "__main__":
    print("Starting generation...")
    generate_docs()
    print("DOCUMENTATION COMPLETE - saved to docs/herald_documentation.tex")
