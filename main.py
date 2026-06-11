from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import os

app = FastAPI(
    title="Cotizador Inteligente API",
    description="API para generación de cotizaciones de construcción con IA — Hogar 911 / Dacam",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    import re
    match = re.search(r'TOTAL:\s*\$?([\d,]+(?:\.\d{2})?)\s*MXN', text, re.IGNORECASE)
    if match:
        return f"${match.group(1)} MXN"
    return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api": "Cotizador Inteligente",
        "version": "1.0.0",
        "empresa": "Hogar 911 / Dacam Constructora",
        "endpoints": {
            "POST /cotizar": "Genera cotización completa (respuesta JSON)",
            "POST /cotizar/stream": "Genera cotización con streaming (SSE)",
            "POST /cotizar/rapida": "Estimación rápida de precio",
            "GET /health": "Estado de la API"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/cotizar", response_model=CotizacionResponse)
def cotizar(req: CotizacionRequest):
    """Genera una cotización completa. Devuelve JSON con el texto completo."""
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    if not req.tipos_trabajo and not req.descripcion:
        raise HTTPException(status_code=422, detail="Proporciona al menos tipos_trabajo o descripcion")

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(req)}]
        )
        texto = message.content[0].text
        total = extract_total(texto)

        return CotizacionResponse(
            cotizacion=texto,
            total_estimado=total,
            metadata={
                "metros_cuadrados": req.metros_cuadrados,
                "tipos_trabajo": req.tipos_trabajo,
                "ubicacion": req.ubicacion,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")


@app.post("/cotizar/stream")
def cotizar_stream(req: CotizacionRequest):
    """Genera cotización con Server-Sent Events (streaming en tiempo real)."""
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")

    def generate():
        try:
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(req)}]
            ) as stream:
                full_text = ""
                for text in stream.text_stream:
                    full_text += text
                    yield f"data: {json.dumps({'delta': text})}\n\n"

                total = extract_total(full_text)
                yield f"data: {json.dumps({'done': True, 'total_estimado': total})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/cotizar/rapida")
def cotizar_rapida(req: CotizacionRequest):
    """Estimación rápida de rango de precio sin desglose completo."""
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
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        # Limpiar posibles backticks
        raw = raw.replace("```json", "").replace("```", "").strip()
        estimacion = json.loads(raw)
        return {"estimacion": estimacion, "metros_cuadrados": req.metros_cuadrados}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error al parsear estimación de IA")
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")
