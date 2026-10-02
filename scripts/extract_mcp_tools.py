"""
extract_mcp_tools.py — Extrae los esquemas de las tools de los servidores MCP del fundador.

Para la v8 de Vertex-Core: necesitamos esquemas de herramientas reales y VARIADAS para
generar ejemplos de tool calling. Este script recorre los servidores MCP (Python FastMCP y
Node) y saca de cada tool: nombre, descripcion (docstring) y argumentos (con tipo).

Python (FastMCP): parsea el AST y toma las funciones decoradas con @mcp.tool / @app.tool /
@server.tool / @tool. Argumentos y tipos salen de la firma; la descripcion del docstring.
Node: extrae los bloques toolDefinitions / name+description+schema por regex ligero.

Salida: data/raw/mcp_tools_catalog.json  (lista de {server, name, description, parameters})

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\extract_mcp_tools.py
"""
import ast
import json
import re
from pathlib import Path

# (nombre_servidor, ruta_al_archivo). Solo los accesibles y con tools ricas/variadas.
PY_SERVERS = [
    ("vertex-net",        r"F:\vertex-net-mcp\server.py"),
    ("vertex-soc",        r"F:\vertex-soc-mcp\server.py"),
    ("host-forensics",    r"F:\vertex-host-forensics-mcp\server.py"),
    ("powershell",        r"F:\ps-mcp\server.py"),
    ("obd-car",           r"H:\mcp-car\server.py"),
    ("drone-tello",       r"H:\mcp-drone\tello_server.py"),
    ("plc",               r"H:\mcp-plc\plc_mcp.py"),
    ("amazon",            r"H:\amazon-scrapping\amazon_mcp_server.py"),
    ("hunting",           r"H:\chequeo-libro\mi-server-mcp\servidor_hunting.py"),
    ("inventario",        r"H:\chequeo-libro\mi-server-mcp\servidor_inventario.py"),
    ("correlacion",       r"H:\chequeo-libro\mi-server-mcp\servidor_correlacion.py"),
    ("vision",            r"F:\vision-core-mcp\mcp_server.py"),
]
NODE_SERVERS = [
    ("calculadora",  r"H:\mcp-calculator\server.js"),
    ("gestor-archivos", r"H:\mcp-book\manager-files\fs_server.js"),
    ("csv-db",       r"H:\mcp-book\database-csv\csv_server.js"),
    ("sqlite-db",    r"H:\mcp-book\database-sqlite\sqlite_server.js"),
    ("google-suite", r"H:\mcp-book\google-mcp\server.js"),
]

_TOOL_DECOS = {"tool"}  # coincide con mcp.tool, app.tool, server.tool, tool

# Mapea tipos de Python a JSON-schema
_PYT = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
        "list": "array", "dict": "object", "bytes": "string"}


def _anno_to_type(node):
    """Convierte una anotacion de tipo del AST a un tipo JSON-schema simple."""
    if node is None:
        return "string"
    if isinstance(node, ast.Name):
        return _PYT.get(node.id, "string")
    if isinstance(node, ast.Subscript):  # list[str], Optional[int]...
        base = getattr(node.value, "id", "")
        if base in ("list", "List"):
            return "array"
        if base in ("dict", "Dict"):
            return "object"
        if base in ("Optional", "Union"):
            return _anno_to_type(node.slice)
        return "string"
    if isinstance(node, ast.Constant) and node.value is None:
        return "string"
    return "string"


def _is_tool_decorator(dec):
    """True si el decorador es @algo.tool(...) o @tool(...)."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        return target.attr in _TOOL_DECOS
    if isinstance(target, ast.Name):
        return target.id in _TOOL_DECOS
    return False


def extract_python(server_name, path):
    tools = []
    src = Path(path).read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  [!] {server_name}: no se pudo parsear ({e}); se omite")
        return tools

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_tool_decorator(d) for d in node.decorator_list):
            continue
        doc = (ast.get_docstring(node) or "").strip().split("\n\n")[0]
        doc = " ".join(doc.split())  # colapsar espacios/saltos
        props, required = {}, []
        args = node.args
        defaults = dict(zip([a.arg for a in args.args][-len(args.defaults):], args.defaults)) \
            if args.defaults else {}
        for a in args.args:
            if a.arg in ("self", "cls"):
                continue
            props[a.arg] = {"type": _anno_to_type(a.annotation)}
            if a.arg not in defaults:
                required.append(a.arg)
        tools.append({
            "server": server_name,
            "name": node.name,
            "description": doc or f"Herramienta {node.name} del servidor {server_name}.",
            "parameters": {"type": "object", "properties": props, "required": required},
        })
    return tools


def extract_node(server_name, path):
    """Extractor ligero para Node: busca name/description y, si hay, inputSchema/schema."""
    tools = []
    src = Path(path).read_text(encoding="utf-8", errors="ignore")
    # Bloques con name: "..." y description: "..."
    for m in re.finditer(r'name:\s*["\']([\w\-]+)["\'][\s\S]{0,400}?description:\s*["\']([^"\']+)["\']', src):
        name, desc = m.group(1), m.group(2)
        tools.append({
            "server": server_name,
            "name": name,
            "description": desc.strip(),
            "parameters": {"type": "object", "properties": {}, "required": []},
        })
    return tools


def main():
    catalog = []
    print("== Servidores Python (FastMCP) ==")
    for name, path in PY_SERVERS:
        if not Path(path).exists():
            print(f"  [-] {name}: no existe {path}")
            continue
        t = extract_python(name, path)
        print(f"  [+] {name:16s} {len(t):3d} tools")
        catalog += t

    print("== Servidores Node ==")
    for name, path in NODE_SERVERS:
        if not Path(path).exists():
            print(f"  [-] {name}: no existe {path}")
            continue
        t = extract_node(name, path)
        print(f"  [+] {name:16s} {len(t):3d} tools")
        catalog += t

    out = Path(r"F:\vertex-lab\data\raw\mcp_tools_catalog.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[i] {len(catalog)} tools en total -> {out}")
    print("[i] Resumen por servidor:")
    by = {}
    for t in catalog:
        by[t["server"]] = by.get(t["server"], 0) + 1
    for s, n in sorted(by.items(), key=lambda x: -x[1]):
        print(f"    {s:16s} {n}")


if __name__ == "__main__":
    main()
