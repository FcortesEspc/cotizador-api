from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import os
import base64
import re

app = FastAPI(title="Cotizador Inteligente API", version="6.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """Eres un experto en presupuestos de construcción en México (CDMX), con 20 años de experiencia con Dacam Constructora y Hogar 911.

Genera cotizaciones profesionales por fases. Formato EXACTO:

FASE 1: [NOMBRE]  ▸  Semanas 1–X
──────────────────────────────────
1.01  [Concepto técnico detallado]
      Unidad: m²  |  Cant: X
      Materiales:   $X,XXX.00
      Mano de obra: $X,XXX.00
      Subtotal 1.01: $X,XXX.00

1.02  [Concepto]
      Unidad: pza  |  Cant: X
      Materiales:   $X,XXX.00
      Mano de obra: $X,XXX.00
      Subtotal 1.02: $X,XXX.00

      SUBTOTAL FASE 1
      Materiales:   $XX,XXX.00
      Mano de obra: $XX,XXX.00
      TOTAL FASE:   $XX,XXX.00

[Siguiente fase...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUMEN FINANCIERO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total costo directo:    $X,XXX,XXX.00
Gastos indirectos (10%): $XXX,XXX.00
Supervisión (5%):        $XXX,XXX.00
IVA (16%):               $XXX,XXX.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL COTIZACIÓN: $X,XXX,XXX.00 MXN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONDICIONES DE PAGO
  Anticipo 30%: $XXX,XXX.00
  Avances por etapa: 60%
  Finiquito 10%: $XXX,XXX.00

Tiempo de ejecución: X semanas

NOTAS:
  • [nota técnica 1]
  • [nota técnica 2]

TOTAL: $X,XXX,XXX.00 MXN

Reglas:
- Máximo 5 fases, 3-4 conceptos por fase (para no exceder tokens)
- Precios realistas CDMX 2024-2025
- Claves numeradas: 1.01, 1.02, 2.01...
- Separa siempre Materiales y Mano de Obra
- La última línea SIEMPRE: TOTAL: $X,XXX,XXX.00 MXN"""

PLANO_SYSTEM = """Eres arquitecto experto en análisis de planos en México. Responde SOLO con JSON:
{"tipo_plano":"texto","metros_cuadrados_totales":0,"niveles":1,"areas":{"sala_comedor":0,"cocina":0,"recamaras":0,"banos":0,"otros":0},"habitaciones":{"recamaras":0,"banos":0,"medios_banos":0},"tipo_construccion":"habitacional","trabajos_identificados":["lista"],"observaciones":"texto","confianza":"alta"}"""


class CotizacionRequest(BaseModel):
    tipos_trabajo: list[str] = []
    metros_cuadrados: float
    niveles: int = 1
    ubicacion: Optional[str] = "Ciudad de México"
    plazo: Optional[str] = "normal"
    descripcion: Optional[str] = ""
    servicios_adicionales: list[str] = []
    nombre_cliente: Optional[str] = ""
    nombre_obra: Optional[str] = ""

class CotizacionResponse(BaseModel):
    cotizacion: str
    total_estimado: Optional[str] = None
    metadata: dict


def build_prompt(req: CotizacionRequest) -> str:
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "construcción general"
    adicionales = ", ".join(req.servicios_adicionales) if req.servicios_adicionales else "ninguno"
    plazo_map = {"urgente": "Urgente (<1 semana)", "normal": "Normal (1-4 semanas)", "largo": "Largo (+1 mes)"}
    num = str(abs(hash(req.nombre_obra or "X")))[-6:].upper()
    return f"""Genera cotización completa para:

COTIZACIÓN No. {num} — Dacam Constructora / Hogar 911
Cliente: {req.nombre_cliente or "Por definir"}
Obra: {req.nombre_obra or "Por definir"}
Ubicación: {req.ubicacion}
Trabajo: {tipos}
m²: {req.metros_cuadrados}  |  Niveles: {req.niveles}
Plazo: {plazo_map.get(req.plazo, req.plazo)}
Servicios adicionales: {adicionales}
Descripción: {req.descripcion or "No proporcionada"}

Genera la cotización completa con máximo 5 fases y 3-4 conceptos por fase."""


def extract_total(text: str) -> Optional[str]:
    match = re.search(r'\*{0,2}TOTAL[^$\d]*\$?([\d,]+(?:\.\d{2})?)\s*MXN', text, re.IGNORECASE)
    return f"${match.group(1)} MXN" if match else None

def clean_json(raw: str) -> str:
    return raw.replace("```json","").replace("```","").strip()


@app.get("/")
def root():
    return {"api": "Cotizador Inteligente", "version": "6.0.0", "empresa": "Hogar 911 / Dacam Constructora",
            "endpoints": {"POST /cotizar": "Cotización completa", "POST /cotizar/stream": "Streaming",
                          "POST /cotizar/rapida": "Estimación rápida", "POST /cotizar/plano": "Analiza plano"}}

@app.get("/health")
def health():
    return {"status": "ok", "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/cotizar", response_model=CotizacionResponse)
def cotizar(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    if not req.tipos_trabajo and not req.descripcion:
        raise HTTPException(status_code=422, detail="Proporciona tipos_trabajo o descripcion")
    try:
        msg = client.messages.create(model=MODEL, max_tokens=8096, system=SYSTEM_PROMPT,
                                     messages=[{"role": "user", "content": build_prompt(req)}])
        texto = msg.content[0].text
        return CotizacionResponse(cotizacion=texto, total_estimado=extract_total(texto),
                                  metadata={"metros_cuadrados": req.metros_cuadrados,
                                            "tipos_trabajo": req.tipos_trabajo, "ubicacion": req.ubicacion,
                                            "tokens": msg.usage.output_tokens})
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")


@app.post("/cotizar/stream")
def cotizar_stream(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")

    def generate():
        try:
            with client.messages.stream(model=MODEL, max_tokens=8096, system=SYSTEM_PROMPT,
                                        messages=[{"role": "user", "content": build_prompt(req)}]) as stream:
                full = ""
                for text in stream.text_stream:
                    full += text
                    yield f"data: {json.dumps({'delta': text})}\n\n"
                yield f"data: {json.dumps({'done': True, 'total_estimado': extract_total(full)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/cotizar/rapida")
def cotizar_rapida(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "construcción general"
    prompt = f"""Estimación rápida para: {tipos}, {req.metros_cuadrados}m², {req.niveles} nivel(es), {req.ubicacion}.
Responde SOLO JSON: {{"rango_minimo":0,"rango_maximo":0,"moneda":"MXN","precio_por_m2_min":0,"precio_por_m2_max":0,"notas":"texto"}}"""
    try:
        msg = client.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
        return {"estimacion": json.loads(clean_json(msg.content[0].text)), "metros_cuadrados": req.metros_cuadrados}
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/cotizar/plano")
async def cotizar_plano(
    archivo: UploadFile = File(...),
    nombre_cliente: str = Form(default=""),
    nombre_obra: str = Form(default=""),
    ubicacion: str = Form(default="Ciudad de México"),
    plazo: str = Form(default="normal"),
    notas_adicionales: str = Form(default="")
):
    content_type = archivo.content_type or ""
    if content_type not in ["image/jpeg","image/jpg","image/png","image/webp","application/pdf"]:
        raise HTTPException(status_code=422, detail=f"Tipo no soportado: {content_type}")
    contenido = await archivo.read()
    if len(contenido) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="Archivo supera 10 MB")
    b64 = base64.standard_b64encode(contenido).decode("utf-8")
    try:
        doc = ({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":b64}}
               if content_type == "application/pdf"
               else {"type":"image","source":{"type":"base64","media_type":content_type,"data":b64}})
        msg = client.messages.create(model=MODEL, max_tokens=1024, system=PLANO_SYSTEM,
                                     messages=[{"role":"user","content":[doc,{"type":"text","text":"Analiza este plano. Solo JSON."}]}])
        analisis = json.loads(clean_json(msg.content[0].text))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="No se pudo interpretar el plano")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error IA: {str(e)}")

    req = CotizacionRequest(
        tipos_trabajo=analisis.get("trabajos_identificados", ["construcción"]),
        metros_cuadrados=analisis.get("metros_cuadrados_totales", 80) or 80,
        niveles=analisis.get("niveles", 1),
        ubicacion=ubicacion, plazo=plazo,
        descripcion=analisis.get("observaciones", ""),
        nombre_cliente=nombre_cliente, nombre_obra=nombre_obra
    )
    try:
        msg2 = client.messages.create(model=MODEL, max_tokens=8096, system=SYSTEM_PROMPT,
                                      messages=[{"role":"user","content":build_prompt(req)}])
        cotizacion = msg2.content[0].text
        return {"analisis_plano": analisis, "cotizacion": cotizacion,
                "total_estimado": extract_total(cotizacion),
                "metadata": {"archivo": archivo.filename, "metros_detectados": req.metros_cuadrados,
                             "confianza_analisis": analisis.get("confianza","media")}}
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error IA: {str(e)}")


# ── Endpoint multi-plano ──────────────────────────────────────────────────────

@app.post("/cotizar/planos")
async def cotizar_planos(
    nombre_cliente: str = Form(default=""),
    nombre_obra: str = Form(default=""),
    ubicacion: str = Form(default="Ciudad de México"),
    plazo: str = Form(default="normal"),
    notas_adicionales: str = Form(default=""),
    archivo_pb: Optional[UploadFile] = File(default=None),
    archivo_pa: Optional[UploadFile] = File(default=None),
    archivo_p2: Optional[UploadFile] = File(default=None),
    archivo_p3: Optional[UploadFile] = File(default=None),
    archivo_p4: Optional[UploadFile] = File(default=None),
):
    """Analiza hasta 5 planos (uno por nivel) y genera cotización consolidada."""
    archivos = [
        ("Planta Baja", archivo_pb),
        ("Planta Alta (Nivel 2)", archivo_pa),
        ("Nivel 3", archivo_p2),
        ("Nivel 4", archivo_p3),
        ("Nivel 5", archivo_p4),
    ]
    archivos_validos = [(nombre, f) for nombre, f in archivos if f and f.filename]

    if not archivos_validos:
        raise HTTPException(status_code=422, detail="Sube al menos un plano")

    allowed = ["image/jpeg","image/jpg","image/png","image/webp","application/pdf"]
    analisis_por_nivel = []

    # Analizar cada plano individualmente
    for nivel_nombre, archivo in archivos_validos:
        content_type = archivo.content_type or ""
        if content_type not in allowed:
            raise HTTPException(status_code=422, detail=f"Tipo no soportado en {nivel_nombre}: {content_type}")
        contenido = await archivo.read()
        if len(contenido) > 10 * 1024 * 1024:
            raise HTTPException(status_code=422, detail=f"{nivel_nombre} supera 10 MB")
        b64 = base64.standard_b64encode(contenido).decode("utf-8")
        doc = ({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":b64}}
               if content_type == "application/pdf"
               else {"type":"image","source":{"type":"base64","media_type":content_type,"data":b64}})
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=1024, system=PLANO_SYSTEM,
                messages=[{"role":"user","content":[doc,{"type":"text","text":f"Analiza este plano de {nivel_nombre}. Solo JSON."}]}]
            )
            analisis = json.loads(clean_json(msg.content[0].text))
            analisis["nivel"] = nivel_nombre
            analisis_por_nivel.append(analisis)
        except (json.JSONDecodeError, anthropic.APIError) as e:
            analisis_por_nivel.append({"nivel": nivel_nombre, "error": str(e), "metros_cuadrados_totales": 0})

    # Consolidar análisis
    metros_total = sum(a.get("metros_cuadrados_totales", 0) or 0 for a in analisis_por_nivel)
    niveles_count = len(analisis_por_nivel)
    todos_trabajos = []
    for a in analisis_por_nivel:
        todos_trabajos.extend(a.get("trabajos_identificados", []))
    trabajos_unicos = list(dict.fromkeys(todos_trabajos))

    resumen_niveles = "\n".join([
        f"- {a['nivel']}: {a.get('metros_cuadrados_totales', 0)} m², "
        f"{a.get('habitaciones', {}).get('recamaras', 0)} rec, "
        f"{a.get('habitaciones', {}).get('banos', 0)} baños"
        f"{' (ERROR: '+a['error']+')' if 'error' in a else ''}"
        for a in analisis_por_nivel
    ])

    plazo_map = {"urgente": "Urgente", "normal": "Normal (1-4 semanas)", "largo": "Largo (+1 mes)"}
    num = str(abs(hash(nombre_obra or "X")))[-6:].upper()

    prompt_consolidado = f"""Genera cotización completa para proyecto de {niveles_count} nivel(es):

COTIZACIÓN No. {num} — Dacam Constructora / Hogar 911
Cliente: {nombre_cliente or "Por definir"}
Obra: {nombre_obra or "Por definir"}
Ubicación: {ubicacion}
Plazo: {plazo_map.get(plazo, plazo)}
Notas: {notas_adicionales or "Ninguna"}

ANÁLISIS POR NIVEL:
{resumen_niveles}

TOTALES CONSOLIDADOS:
- Metros cuadrados totales: {metros_total} m²
- Número de niveles: {niveles_count}
- Trabajos identificados: {', '.join(trabajos_unicos) if trabajos_unicos else 'construcción general'}

Genera cotización completa con máximo 5 fases considerando TODOS los niveles.
Cada fase debe desglosar el trabajo por nivel cuando aplique.
Máximo 3-4 conceptos por fase."""

    try:
        msg = client.messages.create(model=MODEL, max_tokens=8096, system=SYSTEM_PROMPT,
                                     messages=[{"role":"user","content":prompt_consolidado}])
        cotizacion = msg.content[0].text
        return {
            "analisis_por_nivel": analisis_por_nivel,
            "consolidado": {
                "metros_totales": metros_total,
                "niveles": niveles_count,
                "trabajos": trabajos_unicos
            },
            "cotizacion": cotizacion,
            "total_estimado": extract_total(cotizacion),
            "metadata": {
                "archivos_procesados": len(archivos_validos),
                "niveles_analizados": [a["nivel"] for a in analisis_por_nivel]
            }
        }
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error IA: {str(e)}")
