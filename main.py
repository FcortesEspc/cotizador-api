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
    version="2.0.0"
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

SYSTEM_PROMPT = """Eres un experto en presupuestos de construcción y demolición en México (Ciudad de México), con más de 20 años de experiencia.
Generas cotizaciones profesionales, detalladas y realistas en pesos mexicanos (MXN).

Siempre incluyes:
1. Resumen ejecutivo del proyecto
2. Desglose de conceptos con:
   - Descripción del concepto
   - Unidad (m², m³, pza, viaje, jornal, etc.)
   - Cantidad
   - Precio unitario (MXN)
   - Importe total por concepto
3. Subtotal de materiales
4. Subtotal de mano de obra
5. Subtotal de equipos/herramienta
6. IVA (16%)
7. Total general
8. Condiciones de pago (anticipo, avances, finiquito)
9. Tiempo estimado de ejecución
10. Notas y consideraciones importantes

Usas precios actualizados para CDMX 2024-2025.
Al final siempre incluyes una línea con el formato exacto: TOTAL: $X,XXX.XX MXN"""

PLANO_SYSTEM = """Eres un arquitecto e ingeniero civil experto en análisis de planos arquitectónicos en México.
Tu tarea es analizar planos de construcción (plantas, cortes, fachadas) y extraer información técnica precisa.

Al analizar un plano debes identificar:
1. Tipo de plano (planta baja, planta alta, fachada, corte, etc.)
2. Metros cuadrados totales de construcción
3. Metros cuadrados por área (sala, cocina, recámaras, baños, etc.)
4. Número de niveles/pisos
5. Número de recámaras, baños, estacionamientos
6. Tipo de construcción (habitacional, comercial, industrial)
7. Trabajos identificados necesarios
8. Observaciones técnicas importantes

Responde SOLO con JSON válido sin markdown ni texto adicional."""


# ── Modelos ───────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(req: CotizacionRequest) -> str:
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "no especificado"
    adicionales = ", ".join(req.servicios_adicionales) if req.servicios_adicionales else "ninguno"
    plazo_map = {
        "urgente": "Urgente (menos de 1 semana)",
        "normal": "Normal (1–4 semanas)",
        "largo": "Proyecto largo (más de 1 mes)"
    }
    plazo_texto = plazo_map.get(req.plazo, req.plazo)
    return f"""Genera una cotización profesional completa para el siguiente proyecto de construcción:

DATOS DEL PROYECTO:
- Cliente: {req.nombre_cliente or "Por definir"}
- Nombre de obra: {req.nombre_obra or "Por definir"}
- Tipo de trabajo: {tipos}
- Metros cuadrados: {req.metros_cuadrados} m²
- Niveles: {req.niveles}
- Ubicación: {req.ubicacion}
- Plazo requerido: {plazo_texto}
- Servicios adicionales: {adicionales}
- Descripción detallada: {req.descripcion or "No proporcionada"}

Genera la cotización completa siguiendo el formato profesional estándar.
Sé específico con cantidades y precios unitarios realistas para CDMX."""


def extract_total(text: str) -> Optional[str]:
    match = re.search(r'TOTAL:\s*\$?([\d,]+(?:\.\d{2})?)\s*MXN', text, re.IGNORECASE)
    if match:
        return f"${match.group(1)} MXN"
    return None


def clean_json(raw: str) -> str:
    return raw.replace("```json", "").replace("```", "").strip()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api": "Cotizador Inteligente",
        "version": "2.0.0",
        "empresa": "Hogar 911 / Dacam Constructora",
        "endpoints": {
            "POST /cotizar": "Cotización completa (JSON)",
            "POST /cotizar/stream": "Cotización con streaming (SSE)",
            "POST /cotizar/rapida": "Estimación rápida de precio",
            "POST /cotizar/plano": "Analiza plano y genera cotización",
            "GET /health": "Estado de la API"
        }
    }


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
        message = client.messages.create(
            model=MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(req)}]
        )
        texto = message.content[0].text
        return CotizacionResponse(
            cotizacion=texto, total_estimado=extract_total(texto),
            metadata={"metros_cuadrados": req.metros_cuadrados, "tipos_trabajo": req.tipos_trabajo,
                      "ubicacion": req.ubicacion, "input_tokens": message.usage.input_tokens,
                      "output_tokens": message.usage.output_tokens}
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")


@app.post("/cotizar/stream")
def cotizar_stream(req: CotizacionRequest):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")

    def generate():
        try:
            with client.messages.stream(
                model=MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(req)}]
            ) as stream:
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
    prompt = f"""Dame SOLO una estimación rápida de precio (rango mínimo-máximo) para:
- Trabajo: {tipos}
- Metros cuadrados: {req.metros_cuadrados} m²
- Niveles: {req.niveles}
- Ubicación: {req.ubicacion}
- Plazo: {req.plazo}

Responde SOLO con JSON válido, sin texto adicional, sin markdown:
{{"rango_minimo": 00000, "rango_maximo": 00000, "moneda": "MXN", "precio_por_m2_min": 000, "precio_por_m2_max": 000, "notas": "texto breve"}}"""
    try:
        message = client.messages.create(model=MODEL, max_tokens=300,
                                         messages=[{"role": "user", "content": prompt}])
        estimacion = json.loads(clean_json(message.content[0].text))
        return {"estimacion": estimacion, "metros_cuadrados": req.metros_cuadrados}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error al parsear estimación de IA")
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
    """Analiza un plano (imagen o PDF) y genera cotización automáticamente."""

    # Validar tipo de archivo
    content_type = archivo.content_type or ""
    allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp", "application/pdf"]
    if content_type not in allowed:
        raise HTTPException(status_code=422,
                            detail=f"Tipo de archivo no soportado: {content_type}. Usa JPG, PNG, WEBP o PDF.")

    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    contenido = await archivo.read()
    if len(contenido) > MAX_SIZE:
        raise HTTPException(status_code=422, detail="El archivo supera el límite de 10 MB.")

    b64 = base64.standard_b64encode(contenido).decode("utf-8")

    # Paso 1: Analizar el plano con visión
    try:
        if content_type == "application/pdf":
            doc_content = {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": b64}
            }
        else:
            doc_content = {
                "type": "image",
                "source": {"type": "base64", "media_type": content_type, "data": b64}
            }

        analysis_prompt = """Analiza este plano arquitectónico y extrae toda la información técnica posible.

Responde SOLO con este JSON (sin markdown, sin texto extra):
{
  "tipo_plano": "descripción del tipo de plano",
  "metros_cuadrados_totales": 0,
  "niveles": 1,
  "areas": {
    "sala_comedor": 0,
    "cocina": 0,
    "recamaras": 0,
    "banos": 0,
    "estacionamiento": 0,
    "otros": 0
  },
  "habitaciones": {
    "recamaras": 0,
    "banos": 0,
    "medios_banos": 0,
    "estacionamientos": 0
  },
  "tipo_construccion": "habitacional/comercial/industrial",
  "trabajos_identificados": ["lista", "de", "trabajos"],
  "observaciones": "observaciones técnicas importantes",
  "confianza": "alta/media/baja"
}"""

        msg_analisis = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=PLANO_SYSTEM,
            messages=[{"role": "user", "content": [doc_content, {"type": "text", "text": analysis_prompt}]}]
        )

        analisis = json.loads(clean_json(msg_analisis.content[0].text))

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="No se pudo interpretar el plano. Intenta con una imagen más clara.")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA al analizar plano: {str(e)}")

    # Paso 2: Generar cotización basada en el análisis
    metros = analisis.get("metros_cuadrados_totales", 0)
    if metros <= 0:
        metros = 80  # fallback si no se detectó

    trabajos = analisis.get("trabajos_identificados", ["construcción"])
    tipos_str = ", ".join(trabajos)

    plazo_map = {"urgente": "Urgente (menos de 1 semana)", "normal": "Normal (1–4 semanas)", "largo": "Proyecto largo (más de 1 mes)"}
    plazo_texto = plazo_map.get(plazo, plazo)

    cotizacion_prompt = f"""Genera una cotización profesional completa basada en el siguiente análisis de plano arquitectónico:

ANÁLISIS DEL PLANO:
- Tipo de plano: {analisis.get('tipo_plano', 'No especificado')}
- Metros cuadrados totales: {metros} m²
- Niveles: {analisis.get('niveles', 1)}
- Tipo de construcción: {analisis.get('tipo_construccion', 'habitacional')}
- Trabajos identificados: {tipos_str}
- Áreas: {json.dumps(analisis.get('areas', {}), ensure_ascii=False)}
- Habitaciones: {json.dumps(analisis.get('habitaciones', {}), ensure_ascii=False)}
- Observaciones del plano: {analisis.get('observaciones', 'Ninguna')}

DATOS DEL CLIENTE:
- Cliente: {nombre_cliente or 'Por definir'}
- Obra: {nombre_obra or 'Por definir'}
- Ubicación: {ubicacion}
- Plazo: {plazo_texto}
- Notas adicionales: {notas_adicionales or 'Ninguna'}

Genera la cotización completa con desglose por áreas y conceptos.
Al final incluye: TOTAL: $X,XXX.XX MXN"""

    try:
        msg_cotizacion = client.messages.create(
            model=MODEL, max_tokens=2500, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": cotizacion_prompt}]
        )
        cotizacion_texto = msg_cotizacion.content[0].text

        return {
            "analisis_plano": analisis,
            "cotizacion": cotizacion_texto,
            "total_estimado": extract_total(cotizacion_texto),
            "metadata": {
                "archivo": archivo.filename,
                "metros_detectados": metros,
                "trabajos_detectados": trabajos,
                "confianza_analisis": analisis.get("confianza", "media")
            }
        }

    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA al generar cotización: {str(e)}")
