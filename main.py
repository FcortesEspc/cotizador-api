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
    version="3.0.0"
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

Generas cotizaciones profesionales en formato estructurado por FASES, usando EXACTAMENTE este formato:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COTIZACIÓN DE PROYECTO No. [NUMERO]
Dacam Constructora / Hogar 911
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cliente:          [NOMBRE CLIENTE]
Obra / Proyecto:  [NOMBRE OBRA]
Dirección:        [UBICACION]
Tipo de proyecto: [TIPO DE TRABAJO]
Descripción:      [DESCRIPCION TECNICA COMPLETA Y PROFESIONAL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESGLOSE POR FASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FASE 1: [NOMBRE DESCRIPTIVO]  ▸  Semanas 1–X
──────────────────────────────────────────
1.01  [Descripción técnica detallada del concepto]
      Unidad: [m²/m³/pza/lote/jornal]  |  Cant: X
      Materiales:  $X,XXX.00
      Mano de obra: $X,XXX.00
      ─────────────────────────────
      Subtotal 1.01:  $X,XXX.00

1.02  [Descripción técnica detallada]
      Unidad: [unidad]  |  Cant: X
      Materiales:  $X,XXX.00
      Mano de obra: $X,XXX.00
      ─────────────────────────────
      Subtotal 1.02:  $X,XXX.00

      ══════════════════════════════
      SUBTOTAL FASE 1
      Materiales:   $XX,XXX.00
      Mano de obra: $XX,XXX.00
      TOTAL FASE:   $XX,XXX.00
      ══════════════════════════════

[Repite para todas las fases necesarias — mínimo 3, máximo 8]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUMEN FINANCIERO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUBTOTAL COSTO DIRECTO DE OBRA
  Materiales:             $X,XXX,XXX.00
  Mano de obra:           $X,XXX,XXX.00
  ─────────────────────────────────────
  Total costo directo:    $X,XXX,XXX.00

Administración y Gastos Indirectos (10%): $XXX,XXX.00
Dirección y Supervisión de Obra:          $XXX,XXX.00
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
  • [Nota técnica relevante 1]
  • [Nota técnica relevante 2]
  • [Nota técnica relevante 3]

TOTAL: $X,XXX,XXX.00 MXN

Reglas estrictas:
- Usa precios REALISTAS para CDMX 2024-2025
- Divide siempre en FASES lógicas y secuenciales
- Cada concepto tiene clave numerada (1.01, 1.02, 2.01...)
- Siempre separa Materiales y Mano de Obra
- Sé técnico y específico — describe exactamente qué incluye cada concepto
- La última línea SIEMPRE debe ser exactamente: TOTAL: $X,XXX,XXX.00 MXN"""

PLANO_SYSTEM = """Eres un arquitecto e ingeniero civil experto en análisis de planos arquitectónicos en México.
Tu tarea es analizar planos de construcción (plantas, cortes, fachadas) y extraer información técnica precisa.

Responde SOLO con JSON válido sin markdown ni texto adicional:
{
  "tipo_plano": "descripción del tipo de plano",
  "metros_cuadrados_totales": 0,
  "niveles": 1,
  "areas": {"sala_comedor": 0, "cocina": 0, "recamaras": 0, "banos": 0, "estacionamiento": 0, "otros": 0},
  "habitaciones": {"recamaras": 0, "banos": 0, "medios_banos": 0, "estacionamientos": 0},
  "tipo_construccion": "habitacional/comercial/industrial",
  "trabajos_identificados": ["lista", "de", "trabajos"],
  "observaciones": "observaciones técnicas importantes",
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


def build_prompt(req: CotizacionRequest) -> str:
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "no especificado"
    adicionales = ", ".join(req.servicios_adicionales) if req.servicios_adicionales else "ninguno"
    plazo_map = {"urgente": "Urgente (menos de 1 semana)", "normal": "Normal (1–4 semanas)", "largo": "Proyecto largo (más de 1 mes)"}
    plazo_texto = plazo_map.get(req.plazo, req.plazo)
    return f"""Genera una cotización profesional completa en el formato de fases indicado para:

DATOS DEL PROYECTO:
- Cliente: {req.nombre_cliente or "Por definir"}
- Nombre de obra: {req.nombre_obra or "Por definir"}
- Tipo de trabajo: {tipos}
- Metros cuadrados: {req.metros_cuadrados} m²
- Niveles: {req.niveles}
- Ubicación: {req.ubicacion}
- Plazo requerido: {plazo_texto}
- Servicios adicionales solicitados: {adicionales}
- Descripción detallada: {req.descripcion or "No proporcionada"}

Genera la cotización completa con todas las fases, desglose de materiales y mano de obra, y el resumen financiero."""


def extract_total(text: str) -> Optional[str]:
    match = re.search(r'TOTAL:\s*\$?([\d,]+(?:\.\d{2})?)\s*MXN', text, re.IGNORECASE)
    return f"${match.group(1)} MXN" if match else None


def clean_json(raw: str) -> str:
    return raw.replace("```json", "").replace("```", "").strip()


@app.get("/")
def root():
    return {"api": "Cotizador Inteligente", "version": "3.0.0", "empresa": "Hogar 911 / Dacam Constructora",
            "endpoints": {"POST /cotizar": "Cotización por fases (JSON)", "POST /cotizar/stream": "Cotización con streaming (SSE)",
                          "POST /cotizar/rapida": "Estimación rápida", "POST /cotizar/plano": "Analiza plano con IA", "GET /health": "Estado"}}


@app.get("/health")
def health():
    return {"status": "ok", "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/cotizar", response_model=CotizacionResponse)
def cotizar(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    if not req.tipos_trabajo and not req.descripcion:
        raise HTTPException(status_code=422, detail="Proporciona al menos tipos_trabajo o descripcion")
    try:
        message = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                                         messages=[{"role": "user", "content": build_prompt(req)}])
        texto = message.content[0].text
        return CotizacionResponse(cotizacion=texto, total_estimado=extract_total(texto),
                                  metadata={"metros_cuadrados": req.metros_cuadrados, "tipos_trabajo": req.tipos_trabajo,
                                            "ubicacion": req.ubicacion, "input_tokens": message.usage.input_tokens,
                                            "output_tokens": message.usage.output_tokens})
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")


@app.post("/cotizar/stream")
def cotizar_stream(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")

    def generate():
        try:
            with client.messages.stream(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                                        messages=[{"role": "user", "content": build_prompt(req)}]) as stream:
                full_text = ""
                for text in stream.text_stream:
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
        raise HTTPException(status_code=422, detail=f"Tipo no soportado: {content_type}. Usa JPG, PNG, WEBP o PDF.")

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
        raise HTTPException(status_code=500, detail="No se pudo interpretar el plano. Intenta con una imagen más clara.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA al analizar plano: {str(e)}")

    metros = analisis.get("metros_cuadrados_totales", 0) or 80
    trabajos = analisis.get("trabajos_identificados", ["construcción"])
    plazo_map = {"urgente": "Urgente", "normal": "Normal (1–4 semanas)", "largo": "Proyecto largo (+1 mes)"}

    cotizacion_prompt = f"""Genera una cotización profesional completa en formato de FASES basada en este análisis de plano:

ANÁLISIS DEL PLANO:
- Tipo: {analisis.get('tipo_plano', 'No especificado')}
- Metros totales: {metros} m²
- Niveles: {analisis.get('niveles', 1)}
- Tipo construcción: {analisis.get('tipo_construccion', 'habitacional')}
- Trabajos identificados: {', '.join(trabajos)}
- Áreas: {json.dumps(analisis.get('areas', {}), ensure_ascii=False)}
- Habitaciones: {json.dumps(analisis.get('habitaciones', {}), ensure_ascii=False)}
- Observaciones: {analisis.get('observaciones', 'Ninguna')}

DATOS DEL CLIENTE:
- Cliente: {nombre_cliente or 'Por definir'}
- Obra: {nombre_obra or 'Por definir'}
- Ubicación: {ubicacion}
- Plazo: {plazo_map.get(plazo, plazo)}
- Notas: {notas_adicionales or 'Ninguna'}

Genera cotización completa con fases, desglose de materiales y mano de obra, y resumen financiero."""

    try:
        msg_cot = client.messages.create(model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                                         messages=[{"role": "user", "content": cotizacion_prompt}])
        cotizacion_texto = msg_cot.content[0].text
        return {"analisis_plano": analisis, "cotizacion": cotizacion_texto,
                "total_estimado": extract_total(cotizacion_texto),
                "metadata": {"archivo": archivo.filename, "metros_detectados": metros,
                             "trabajos_detectados": trabajos, "confianza_analisis": analisis.get("confianza", "media")}}
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA al generar cotización: {str(e)}")
