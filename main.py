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

app = FastAPI(
    title="Cotizador Inteligente API",
    description="API para generación de cotizaciones de construcción con IA — Hogar 911 / Dacam",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """Eres un experto en presupuestos de construcción y demolición en México (Ciudad de México), con más de 20 años de experiencia trabajando con contratistas de alto nivel como Dacam Constructora y Hogar 911.

Generas cotizaciones profesionales en formato estructurado por FASES usando EXACTAMENTE este formato:

FASE X: [NOMBRE DESCRIPTIVO]  ▸  Semanas X–X
──────────────────────────────────────────────────────
X.01  [Descripción técnica detallada del concepto]
      Unidad: [m²/m³/pza/lote/jornal]  |  Cant: X
      Materiales:   $X,XXX.00
      Mano de obra: $X,XXX.00
      ─────────────────────────────
      Subtotal X.01:  $X,XXX.00

      ══════════════════════════════════════
      SUBTOTAL FASE X
      Materiales:   $XX,XXX.00
      Mano de obra: $XX,XXX.00
      TOTAL FASE:   $XX,XXX.00
      ══════════════════════════════════════

Reglas:
- Usa precios REALISTAS para CDMX 2024-2025
- Cada concepto tiene clave numerada (1.01, 1.02, 2.01...)
- Siempre separa Materiales y Mano de Obra
- Sé técnico y específico en cada concepto
- NO incluyas encabezado ni resumen financiero — solo las fases indicadas"""

RESUMEN_PROMPT = """Eres un experto en presupuestos de construcción en México (Dacam Constructora / Hogar 911).

Dado el desglose de fases de una cotización, genera el resumen financiero final con EXACTAMENTE este formato:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUMEN FINANCIERO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUBTOTAL COSTO DIRECTO DE OBRA
  Materiales:             $X,XXX,XXX.00
  Mano de obra:           $X,XXX,XXX.00
  ─────────────────────────────────────
  Total costo directo:    $X,XXX,XXX.00

Administración y Gastos Indirectos (10%): $XXX,XXX.00
Dirección y Supervisión de Obra (5%):     $XXX,XXX.00
IVA (16%):                                $XXX,XXX.00

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL COTIZACIÓN:  $X,XXX,XXX.00 MXN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONDICIONES DE PAGO
  ▸ Anticipo:   30% al inicio — $XXX,XXX.00
  ▸ 1er avance: 25% al término de cimentación
  ▸ 2do avance: 25% al término de estructura
  ▸ 3er avance: 10% al término de instalaciones
  ▸ Finiquito:  10% a la entrega — $XXX,XXX.00

Tiempo de ejecución: X semanas
Validez de cotización: 30 días naturales

NOTAS TÉCNICAS IMPORTANTES:
  • [Nota técnica 1]
  • [Nota técnica 2]
  • [Nota técnica 3]

TOTAL: $X,XXX,XXX.00 MXN

Calcula los totales sumando los subtotales de cada fase del desglose proporcionado.
La última línea SIEMPRE debe ser: TOTAL: $X,XXX,XXX.00 MXN"""

PLANO_SYSTEM = """Eres un arquitecto e ingeniero civil experto en análisis de planos arquitectónicos en México.
Responde SOLO con JSON válido sin markdown:
{
  "tipo_plano": "descripción",
  "metros_cuadrados_totales": 0,
  "niveles": 1,
  "areas": {"sala_comedor": 0, "cocina": 0, "recamaras": 0, "banos": 0, "estacionamiento": 0, "otros": 0},
  "habitaciones": {"recamaras": 0, "banos": 0, "medios_banos": 0, "estacionamientos": 0},
  "tipo_construccion": "habitacional/comercial/industrial",
  "trabajos_identificados": ["lista"],
  "observaciones": "observaciones técnicas",
  "confianza": "alta/media/baja"
}"""


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


def build_header(req: CotizacionRequest) -> str:
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "no especificado"
    plazo_map = {"urgente": "Urgente (menos de 1 semana)", "normal": "Normal (1–4 semanas)", "largo": "Proyecto largo (más de 1 mes)"}
    return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COTIZACIÓN DE PROYECTO No. {str(hash(req.nombre_obra or 'X'))[-6:].upper()}
Dacam Constructora / Hogar 911
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cliente:          {req.nombre_cliente or "Por definir"}
Obra / Proyecto:  {req.nombre_obra or "Por definir"}
Dirección:        {req.ubicacion}
Tipo de proyecto: {tipos}
Metros cuadrados: {req.metros_cuadrados} m²  |  Niveles: {req.niveles}
Plazo requerido:  {plazo_map.get(req.plazo, req.plazo)}
Servicios adicionales: {", ".join(req.servicios_adicionales) if req.servicios_adicionales else "Ninguno"}
Descripción:      {req.descripcion or "Proyecto de construcción/remodelación"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESGLOSE POR FASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def build_fases_prompt(req: CotizacionRequest, fases: str, inicio: int, fin: int) -> str:
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "construcción general"
    adicionales = ", ".join(req.servicios_adicionales) if req.servicios_adicionales else "ninguno"
    return f"""Genera ÚNICAMENTE las fases {inicio} a {fin} de la cotización para:

PROYECTO:
- Tipo: {tipos}
- Metros cuadrados: {req.metros_cuadrados} m²
- Niveles: {req.niveles}
- Ubicación: {req.ubicacion}
- Descripción: {req.descripcion or "No proporcionada"}
- Servicios adicionales: {adicionales}

FASES TOTALES DEL PROYECTO: {fases}

Genera SOLO las fases {inicio} a {fin} con su desglose completo de conceptos, materiales y mano de obra.
NO incluyas encabezado, NO incluyas resumen financiero. Solo las fases indicadas."""

def determine_fases(req: CotizacionRequest) -> list:
    """Determina las fases según el tipo de trabajo y tamaño."""
    tipos = " ".join(req.tipos_trabajo).lower()
    m2 = req.metros_cuadrados

    if "demolicion" in tipos or "cascajo" in tipos:
        return ["FASE 1: PREPARACIÓN Y PROTECCIONES",
                "FASE 2: DEMOLICIÓN ESTRUCTURAL",
                "FASE 3: RETIRO Y ACARREO DE CASCAJO",
                "FASE 4: LIMPIEZA FINAL Y ENTREGA"]

    if "remodelacion" in tipos or "acabados" in tipos:
        return ["FASE 1: DEMOLICIÓN PARCIAL Y PREPARACIÓN",
                "FASE 2: OBRA CIVIL Y ALBAÑILERÍA",
                "FASE 3: INSTALACIONES",
                "FASE 4: ACABADOS INTERIORES",
                "FASE 5: PINTURA, CANCELERÍA Y DETALLES FINALES"]

    if m2 <= 80:
        return ["FASE 1: PRELIMINARES Y CIMENTACIÓN",
                "FASE 2: ESTRUCTURA Y MUROS",
                "FASE 3: INSTALACIONES",
                "FASE 4: ACABADOS Y ENTREGA"]

    if req.niveles >= 3 or m2 >= 200:
        return ["FASE 1: TERRACERÍAS Y CIMENTACIÓN",
                "FASE 2: ESTRUCTURA PRINCIPAL",
                "FASE 3: MUROS Y ALBAÑILERÍA",
                "FASE 4: INSTALACIONES HIDROSANITARIAS Y GAS",
                "FASE 5: INSTALACIONES ELÉCTRICAS",
                "FASE 6: ACABADOS INTERIORES",
                "FASE 7: FACHADA, EXTERIORES Y ENTREGA"]

    return ["FASE 1: PRELIMINARES Y CIMENTACIÓN",
            "FASE 2: ESTRUCTURA DE CONCRETO",
            "FASE 3: MUROS Y ALBAÑILERÍA",
            "FASE 4: INSTALACIONES HIDROSANITARIAS Y ELÉCTRICAS",
            "FASE 5: ACABADOS, PINTURA Y ENTREGA"]

def extract_total(text: str) -> Optional[str]:
    match = re.search(r'TOTAL:\s*\$?([\d,]+(?:\.\d{2})?)\s*MXN', text, re.IGNORECASE)
    return f"${match.group(1)} MXN" if match else None

def clean_json(raw: str) -> str:
    return raw.replace("```json", "").replace("```", "").strip()


@app.get("/")
def root():
    return {"api": "Cotizador Inteligente", "version": "4.0.0", "empresa": "Hogar 911 / Dacam Constructora",
            "endpoints": {"POST /cotizar": "Cotización completa por fases (multi-llamada)",
                          "POST /cotizar/stream": "Cotización streaming",
                          "POST /cotizar/rapida": "Estimación rápida",
                          "POST /cotizar/plano": "Analiza plano con IA",
                          "GET /health": "Estado"}}

@app.get("/health")
def health():
    return {"status": "ok", "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/cotizar", response_model=CotizacionResponse)
def cotizar(req: CotizacionRequest):
    """Genera cotización completa dividida en grupos de fases para evitar cortes."""
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    if not req.tipos_trabajo and not req.descripcion:
        raise HTTPException(status_code=422, detail="Proporciona al menos tipos_trabajo o descripcion")

    fases = determine_fases(req)
    fases_str = "\n".join([f"  {f}" for f in fases])
    n = len(fases)

    # Dividir fases en grupos de 3
    grupos = []
    for i in range(0, n, 3):
        grupos.append(fases[i:i+3])

    cotizacion_completa = build_header(req)

    try:
        # Generar cada grupo de fases
        for gi, grupo in enumerate(grupos):
            inicio_idx = gi * 3 + 1
            fin_idx = min(inicio_idx + len(grupo) - 1, n)
            prompt = build_fases_prompt(req, fases_str, inicio_idx, fin_idx)
            msg = client.messages.create(
                model=MODEL, max_tokens=8096, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            cotizacion_completa += msg.content[0].text + "\n\n"

        # Generar resumen financiero final
        resumen_prompt = f"""Basándote en el siguiente desglose de fases, genera el resumen financiero completo.
Suma todos los subtotales de cada fase para calcular el costo directo total.

DESGLOSE DE FASES:
{cotizacion_completa}

Genera el resumen financiero con los totales correctos calculados."""

        msg_resumen = client.messages.create(
            model=MODEL, max_tokens=2048, system=RESUMEN_PROMPT,
            messages=[{"role": "user", "content": resumen_prompt}]
        )
        cotizacion_completa += msg_resumen.content[0].text

        return CotizacionResponse(
            cotizacion=cotizacion_completa,
            total_estimado=extract_total(cotizacion_completa),
            metadata={"metros_cuadrados": req.metros_cuadrados, "tipos_trabajo": req.tipos_trabajo,
                      "ubicacion": req.ubicacion, "fases": fases, "grupos_generados": len(grupos) + 1}
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")


@app.post("/cotizar/stream")
def cotizar_stream(req: CotizacionRequest):
    """Genera cotización completa con streaming — envía cada fase conforme se genera."""
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")

    def generate():
        try:
            fases = determine_fases(req)
            fases_str = "\n".join([f"  {f}" for f in fases])
            n = len(fases)
            grupos = []
            for i in range(0, n, 3):
                grupos.append(fases[i:i+3])

            header = build_header(req)
            yield f"data: {json.dumps({'delta': header})}\n\n"

            full_text = header

            # Generar fases por grupos
            for gi, grupo in enumerate(grupos):
                inicio_idx = gi * 3 + 1
                fin_idx = min(inicio_idx + len(grupo) - 1, n)
                prompt = build_fases_prompt(req, fases_str, inicio_idx, fin_idx)

                with client.messages.stream(
                    model=MODEL, max_tokens=8096, system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    chunk = ""
                    for text in stream.text_stream:
                        chunk += text
                        full_text += text
                        yield f"data: {json.dumps({'delta': text})}\n\n"
                    full_text += "\n\n"
                    yield f"data: {json.dumps({'delta': '\n\n'})}\n\n"

            # Generar resumen financiero
            sep = "\n"
            yield f"data: {json.dumps({'delta': sep})}\n\n"

            resumen_prompt = f"""Basándote en el siguiente desglose de fases, genera el resumen financiero.
Suma todos los TOTAL FASE para calcular el costo directo total.

{full_text}

Genera solo el resumen financiero con totales calculados correctamente."""

            with client.messages.stream(
                model=MODEL, max_tokens=2048, system=RESUMEN_PROMPT,
                messages=[{"role": "user", "content": resumen_prompt}]
            ) as stream:
                resumen = ""
                for text in stream.text_stream:
                    resumen += text
                    full_text += text
                    yield f"data: {json.dumps({'delta': text})}\n\n"

            yield f"data: {json.dumps({'done': True, 'total_estimado': extract_total(full_text)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/cotizar/rapida")
def cotizar_rapida(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "construcción general"
    prompt = f"""Dame SOLO una estimación rápida de precio para:
- Trabajo: {tipos}
- m²: {req.metros_cuadrados}
- Niveles: {req.niveles}
- Ubicación: {req.ubicacion}
- Plazo: {req.plazo}

Responde SOLO con JSON sin markdown:
{{"rango_minimo": 0, "rango_maximo": 0, "moneda": "MXN", "precio_por_m2_min": 0, "precio_por_m2_max": 0, "notas": "texto breve"}}"""
    try:
        message = client.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
        estimacion = json.loads(clean_json(message.content[0].text))
        return {"estimacion": estimacion, "metros_cuadrados": req.metros_cuadrados}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error al parsear estimación")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")


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
    allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp", "application/pdf"]
    if content_type not in allowed:
        raise HTTPException(status_code=422, detail=f"Tipo no soportado: {content_type}.")

    contenido = await archivo.read()
    if len(contenido) > 10 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="El archivo supera 10 MB.")

    b64 = base64.standard_b64encode(contenido).decode("utf-8")

    try:
        doc_content = ({"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}
                       if content_type == "application/pdf"
                       else {"type": "image", "source": {"type": "base64", "media_type": content_type, "data": b64}})

        msg_analisis = client.messages.create(
            model=MODEL, max_tokens=1024, system=PLANO_SYSTEM,
            messages=[{"role": "user", "content": [doc_content, {"type": "text", "text":
                "Analiza este plano y extrae toda la información técnica. Responde SOLO con el JSON indicado."}]}]
        )
        analisis = json.loads(clean_json(msg_analisis.content[0].text))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="No se pudo interpretar el plano.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA al analizar plano: {str(e)}")

    # Usar el endpoint /cotizar con los datos del plano
    req = CotizacionRequest(
        tipos_trabajo=analisis.get("trabajos_identificados", ["construcción"]),
        metros_cuadrados=analisis.get("metros_cuadrados_totales", 80) or 80,
        niveles=analisis.get("niveles", 1),
        ubicacion=ubicacion,
        plazo=plazo,
        descripcion=analisis.get("observaciones", ""),
        nombre_cliente=nombre_cliente,
        nombre_obra=nombre_obra
    )

    try:
        resultado = cotizar(req)
        return {"analisis_plano": analisis, "cotizacion": resultado.cotizacion,
                "total_estimado": resultado.total_estimado,
                "metadata": {"archivo": archivo.filename,
                             "metros_detectados": req.metros_cuadrados,
                             "trabajos_detectados": req.tipos_trabajo,
                             "confianza_analisis": analisis.get("confianza", "media")}}
    except HTTPException as e:
        raise e
