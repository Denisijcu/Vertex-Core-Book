"""
01e_toolcall_builder.py — Genera ejemplos de TOOL CALLING para la v8 de Vertex-Core.

Usa herramientas REALES de los servidores MCP del fundador (extraidas en
data/raw/mcp_tools_catalog.json), de dominios variados (red, forense, coche, dron, PLC,
Amazon, vision, Google...), para que el modelo aprenda el PATRON de tool calling y lo
generalice, en vez de memorizar unas pocas tools.

Cuatro casos (diagnostico del proyecto, ver docs/PLAN_v8.md):
  A) accion externa  -> el modelo emite tool_call, recibe resultado, informa lo REAL.
  B) sin herramienta -> pregunta conceptual, responde directo, SIN tool.
  C) la tool falla   -> resultado con error -> el modelo lo dice; NUNCA "ya lo hice".
  D) no inventar     -> "lo hiciste?" sin ejecutar / falta un argumento -> pide, no inventa.

CLAVE: se renderiza con el chat template real de Qwen3 (apply_chat_template con tools=...),
para que el formato <tool_call>/<tool_response> sea identico al de produccion.

Salida: data/raw/toolcall_examples.jsonl  (cada linea: {"text": "<conversacion renderizada>"})

Uso (desde F:\\vertex-lab):
  .\\venv\\Scripts\\python.exe scripts\\01e_toolcall_builder.py --preview
  .\\venv\\Scripts\\python.exe scripts\\01e_toolcall_builder.py --repeat 3
"""
import argparse
import json

from common import banner, load_config, rpath
from transformers import AutoTokenizer

CATALOG = rpath("data/raw/mcp_tools_catalog.json")

SYSTEM = (
    "Eres Vertex-Core, agente de Vertex Coders para tareas tecnicas y de ciberseguridad "
    "autorizada. Tienes herramientas disponibles. Cuando la tarea necesita un dato real o una "
    "accion (escanear, leer, ejecutar, enviar), LLAMA a la herramienta adecuada y espera su "
    "resultado. Solo afirmas que hiciste algo si recibiste el resultado de la herramienta. Si "
    "te falta un argumento obligatorio, lo pides; no lo inventes."
)


def load_tool(catalog, server, name):
    """Devuelve el esquema de una tool del catalogo, en formato OpenAI/function."""
    for t in catalog:
        if t["server"] == server and t["name"] == name:
            return {"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["parameters"]}}
    raise KeyError(f"tool no encontrada: {server}.{name}")


def _tc(name, args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]}


# ── Los ejemplos: (tools_que_se_ofrecen, [mensajes]) ────────────────────────────
# Se escriben a mano para que el comportamiento sea EXACTO. Variedad de dominios y casos.
def build_examples(cat):
    T = lambda s, n: load_tool(cat, s, n)
    ex = []

    # ===================== CASO A — accion externa (llama, recibe, informa) =====================
    # A sin argumentos:
    ex.append(([T("vertex-net", "get_network_interfaces")], [
        {"role": "user", "content": "Dame las interfaces de red de esta maquina."},
        _tc("get_network_interfaces", {}),
        {"role": "tool", "name": "get_network_interfaces",
         "content": '[{"InterfaceAlias":"Wi-Fi","IPv4Address":"192.168.1.20","state":"Up"},{"InterfaceAlias":"Ethernet","state":"Down"}]'},
        {"role": "assistant", "content": "Tienes dos interfaces: Wi-Fi activa con IP 192.168.1.20, y Ethernet desconectada."},
    ]))
    ex.append(([T("obd-car", "read_dtc")], [
        {"role": "user", "content": "Lee los codigos de fallo del carro."},
        _tc("read_dtc", {}),
        {"role": "tool", "name": "read_dtc", "content": '{"codes": ["P0301", "P0420"]}'},
        {"role": "assistant", "content": "El vehiculo reporta dos codigos: P0301 (fallo de encendido en cilindro 1) "
                                          "y P0420 (eficiencia del catalizador por debajo del umbral). ¿Quieres que te "
                                          "explique alguno en detalle con explain_dtc?"},
    ]))
    ex.append(([T("drone-tello", "takeoff"), T("drone-tello", "get_state")], [
        {"role": "user", "content": "Despega el dron."},
        _tc("takeoff", {}),
        {"role": "tool", "name": "takeoff", "content": '{"ok": true, "height_cm": 80}'},
        {"role": "assistant", "content": "Dron en el aire, estabilizado a 80 cm de altura."},
    ]))
    # A con argumentos:
    ex.append(([T("vertex-soc", "scan_subnet_full")], [
        {"role": "user", "content": "Escanea la subred 192.168.1.0/24 en los puertos 22,80,443."},
        _tc("scan_subnet_full", {"subnet": "192.168.1.0/24", "ports": "22,80,443"}),
        {"role": "tool", "name": "scan_subnet_full",
         "content": '{"hosts":[{"ip":"192.168.1.1","open":[80,443]},{"ip":"192.168.1.20","open":[22]}]}'},
        {"role": "assistant", "content": "Dos hosts vivos: 192.168.1.1 con 80 y 443 abiertos (router/web), y "
                                          "192.168.1.20 con 22 (SSH). El siguiente paso seria enumerar el HTTPS del router."},
    ]))
    ex.append(([T("obd-car", "explain_dtc")], [
        {"role": "user", "content": "Que significa el codigo P0420?"},
        _tc("explain_dtc", {"code": "P0420"}),
        {"role": "tool", "name": "explain_dtc",
         "content": '{"code":"P0420","desc":"Catalyst System Efficiency Below Threshold (Bank 1)"}'},
        {"role": "assistant", "content": "El P0420 indica que la eficiencia del sistema catalizador del banco 1 esta "
                                          "por debajo del umbral: suele ser el catalizador desgastado o un sensor de "
                                          "oxigeno defectuoso."},
    ]))
    ex.append(([T("plc", "recommend_setpoint")], [
        {"role": "user", "content": "Recomienda el setpoint para el tag TT-101 buscando 75 grados."},
        _tc("recommend_setpoint", {"tag": "TT-101", "valor_deseado": 75}),
        {"role": "tool", "name": "recommend_setpoint", "content": '{"tag":"TT-101","setpoint":74.2,"reason":"compensa inercia termica"}'},
        {"role": "assistant", "content": "Para TT-101 apuntando a 75 grados, el setpoint recomendado es 74.2, que "
                                          "compensa la inercia termica del proceso."},
    ]))
    ex.append(([T("amazon", "get_top_opportunities")], [
        {"role": "user", "content": "Dame las 3 mejores oportunidades de Amazon."},
        _tc("get_top_opportunities", {"limit": 3}),
        {"role": "tool", "name": "get_top_opportunities",
         "content": '[{"asin":"B0A","margin":0.42},{"asin":"B0B","margin":0.38},{"asin":"B0C","margin":0.35}]'},
        {"role": "assistant", "content": "Las 3 mejores por margen: B0A (42%), B0B (38%) y B0C (35%)."},
    ]))
    ex.append(([T("hunting", "find_suspicious_processes")], [
        {"role": "user", "content": "Busca procesos sospechosos en el equipo."},
        _tc("find_suspicious_processes", {}),
        {"role": "tool", "name": "find_suspicious_processes",
         "content": '[{"name":"svch0st.exe","pid":4210,"reason":"nombre similar a proceso legitimo"}]'},
        {"role": "assistant", "content": "Encontre uno sospechoso: svch0st.exe (PID 4210), con un nombre que imita "
                                          "svchost.exe legitimo. Vale la pena revisar su ruta y firma con forense."},
    ]))
    ex.append(([T("vision", "list_cameras")], [
        {"role": "user", "content": "Que camaras hay disponibles?"},
        _tc("list_cameras", {}),
        {"role": "tool", "name": "list_cameras", "content": '[{"id":"cam-01","name":"Entrada"},{"id":"cam-02","name":"Patio"}]'},
        {"role": "assistant", "content": "Hay dos camaras: cam-01 (Entrada) y cam-02 (Patio)."},
    ]))
    ex.append(([T("google-suite", "enviar_correo_template")], [
        {"role": "user", "content": "Envia un correo a ana@empresa.com con asunto 'Reunion' y el cuerpo 'Nos vemos el lunes'."},
        _tc("enviar_correo_template", {"to": "ana@empresa.com", "subject": "Reunion", "body": "Nos vemos el lunes"}),
        {"role": "tool", "name": "enviar_correo_template", "content": '{"sent": true, "to": "ana@empresa.com"}'},
        {"role": "assistant", "content": "Correo enviado a ana@empresa.com con el asunto 'Reunion'."},
    ]))

    # ===================== CASO B — sin herramienta (conceptual, responde directo) =====================
    ex.append(([T("vertex-net", "get_network_interfaces")], [
        {"role": "user", "content": "Que es una direccion MAC?"},
        {"role": "assistant", "content": "La direccion MAC es un identificador unico de 48 bits grabado en la tarjeta "
                                          "de red, que identifica el dispositivo a nivel de enlace (capa 2). Se escribe "
                                          "como seis pares hexadecimales, por ejemplo 40:23:43:C5:44:F3."},
    ]))
    ex.append(([T("vertex-soc", "scan_subnet_full")], [
        {"role": "user", "content": "Explicame que es una subred /24."},
        {"role": "assistant", "content": "Una /24 es una subred con mascara 255.255.255.0: los primeros 24 bits son la "
                                          "red y los ultimos 8 los hosts, o sea 254 direcciones utiles (256 menos la de "
                                          "red y la de broadcast)."},
    ]))
    ex.append(([T("plc", "read_alarms")], [
        {"role": "user", "content": "Que es un PLC?"},
        {"role": "assistant", "content": "Un PLC (Controlador Logico Programable) es una computadora industrial robusta "
                                          "que automatiza procesos: lee sensores, ejecuta logica y acciona salidas, en "
                                          "ciclos rapidos y con alta fiabilidad."},
    ]))
    ex.append(([T("amazon", "get_market_summary")], [
        {"role": "user", "content": "Cuanto es 15% de 240?"},
        {"role": "assistant", "content": "36."},
    ]))

    # ===================== CASO C — la tool falla (lo dice, NO finge exito) =====================
    ex.append(([T("drone-tello", "takeoff")], [
        {"role": "user", "content": "Despega el dron."},
        _tc("takeoff", {}),
        {"role": "tool", "name": "takeoff", "content": '{"ok": false, "error": "battery too low (12%)"}'},
        {"role": "assistant", "content": "No pude despegar: la herramienta reporto bateria demasiado baja (12%). Carga "
                                          "el dron antes de volar."},
    ]))
    ex.append(([T("vertex-soc", "scan_subnet_full")], [
        {"role": "user", "content": "Escanea la subred 10.0.0.0/24 en el puerto 445."},
        _tc("scan_subnet_full", {"subnet": "10.0.0.0/24", "ports": "445"}),
        {"role": "tool", "name": "scan_subnet_full", "content": '{"error": "network unreachable"}'},
        {"role": "assistant", "content": "El escaneo fallo: la red 10.0.0.0/24 es inalcanzable desde aqui. Verifica la "
                                          "conexion o la VPN al segmento objetivo."},
    ]))
    ex.append(([T("google-suite", "enviar_correo_template")], [
        {"role": "user", "content": "Manda el correo a juan@correo.com, asunto 'Hola', cuerpo 'prueba'."},
        _tc("enviar_correo_template", {"to": "juan@correo.com", "subject": "Hola", "body": "prueba"}),
        {"role": "tool", "name": "enviar_correo_template", "content": '{"sent": false, "error": "SMTP auth failed"}'},
        {"role": "assistant", "content": "No se envio el correo: la herramienta devolvio un error de autenticacion SMTP. "
                                          "Hay que revisar las credenciales del servidor de correo."},
    ]))

    # ===================== CASO D — no inventar (pregunta si lo hizo / falta argumento) =====================
    ex.append(([T("google-suite", "enviar_correo_template")], [
        {"role": "user", "content": "Ya enviaste el reporte al cliente?"},
        {"role": "assistant", "content": "No, todavia no lo he enviado: no he ejecutado enviar_correo_template en esta "
                                          "conversacion. Pasame el correo del cliente, el asunto y el cuerpo y lo envio."},
    ]))
    ex.append(([T("google-suite", "enviar_correo_template")], [
        {"role": "user", "content": "Mandale la propuesta por correo a Deloitte."},
        {"role": "assistant", "content": "Necesito la direccion de correo exacta del destinatario; no la voy a inventar. "
                                          "Pasame el email de tu contacto en Deloitte y te redacto y envio la propuesta."},
    ]))
    ex.append(([T("vertex-soc", "scan_subnet_full")], [
        {"role": "user", "content": "Escanea la subred."},
        {"role": "assistant", "content": "¿Que subred quieres escanear? Dame el rango (por ejemplo 192.168.1.0/24) y, si "
                                          "quieres, los puertos; no voy a asumir un objetivo por mi cuenta."},
    ]))
    ex.append(([T("obd-car", "explain_dtc")], [
        {"role": "user", "content": "Explicame el codigo de error."},
        {"role": "assistant", "content": "¿Cual codigo? Dame el codigo DTC exacto (por ejemplo P0420) y te lo explico. "
                                          "Si no lo tienes, puedo leerlos primero con read_dtc."},
    ]))

    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()

    cfg = load_config()
    base = cfg["model"]["base_model_name"]
    banner("01e - Ejemplos de tool calling (tools reales, formato Qwen3)")
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    print(f"[*] Catalogo: {len(cat)} tools. Cargando tokenizer de {base} ...")
    tok = AutoTokenizer.from_pretrained(base, cache_dir=cfg["model"].get("cache_dir"))

    examples = build_examples(cat)
    print(f"[*] {len(examples)} ejemplos base (se repiten x{a.repeat}).")

    if a.preview:
        tools, convo = examples[0]
        msgs = [{"role": "system", "content": SYSTEM}] + convo
        print("\n===== EJEMPLO RENDERIZADO =====\n")
        print(tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False))
        print("\n===== fin =====")
        return

    out = rpath("data/raw/toolcall_examples.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for _ in range(a.repeat):
            for tools, convo in examples:
                msgs = [{"role": "system", "content": SYSTEM}] + convo
                text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
    banner(f"{n} ejemplos de tool calling -> {out}")
    print("[i] Formato 'text' (ya renderizado). El 02_trainer.py debe saber leerlo (pendiente v8).")


if __name__ == "__main__":
    main()
