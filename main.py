from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import anthropic
import json
import os
import base64
import re
import hashlib
import hmac
import time
import asyncpg

app = FastAPI(title="Cotizador Inteligente API", version="7.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "cotizador_hogar911_2026")
MODEL = "claude-sonnet-4-5"
security = HTTPBearer(auto_error=False)

# ── Auth ──────────────────────────────────────────────────────────────────────

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def make_token(user_id: int, email: str, nombre: str, rol: str) -> str:
    payload = json.dumps({"id": user_id, "email": email, "nombre": nombre, "rol": rol, "exp": int(time.time()) + 86400 * 7})
    encoded = base64.b64encode(payload.encode()).decode()
    sig = hmac.new(JWT_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{sig}"

def verify_token(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            raise ValueError()
        encoded, sig = parts
        expected = hmac.new(JWT_SECRET.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError()
        payload = json.loads(base64.b64decode(encoded).decode())
        if payload.get("exp", 0) < time.time():
            raise ValueError("expired")
        return payload
    except:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Token requerido")
    return verify_token(credentials.credentials)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un experto en presupuestos de construcción en México (CDMX), con Dacam Constructora y Hogar 911 — Construyendo Confianza.

Genera cotizaciones profesionales por fases. Formato EXACTO:

FASE 1: [NOMBRE]  ▸  Semanas 1–X
──────────────────────────────────
1.01  [Concepto técnico detallado]
      Unidad: m²  |  Cant: X
      Materiales:   $X,XXX.00
      Mano de obra: $X,XXX.00
      Subtotal 1.01: $X,XXX.00

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
  Avances: 60%
  Finiquito 10%: $XXX,XXX.00

Tiempo de ejecución: X semanas

NOTAS:
  • [nota 1]
  • [nota 2]

TOTAL: $X,XXX,XXX.00 MXN

Reglas:
- Máximo 5 fases, 3-4 conceptos por fase
- Precios realistas CDMX 2024-2025
- Claves numeradas: 1.01, 2.01...
- Separa Materiales y Mano de Obra
- Última línea SIEMPRE: TOTAL: $X,XXX,XXX.00 MXN"""

PLANO_SYSTEM = """Eres arquitecto experto en análisis de planos en México. Responde SOLO con JSON:
{"tipo_plano":"texto","metros_cuadrados_totales":0,"niveles":1,"areas":{"sala_comedor":0,"cocina":0,"recamaras":0,"banos":0,"otros":0},"habitaciones":{"recamaras":0,"banos":0,"medios_banos":0},"tipo_construccion":"habitacional","trabajos_identificados":["lista"],"observaciones":"texto","confianza":"alta"}"""

# ── Modelos ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

class UsuarioCreate(BaseModel):
    nombre: str
    email: str
    password: str
    rol: Optional[str] = "vendedor"

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

class CotizacionGuardar(BaseModel):
    nombre_cliente: Optional[str] = ""
    nombre_obra: Optional[str] = ""
    tipos_trabajo: list[str] = []
    metros_cuadrados: Optional[float] = 0
    ubicacion: Optional[str] = ""
    total_estimado: Optional[str] = ""
    contenido: str

class CotizacionResponse(BaseModel):
    cotizacion: str
    total_estimado: Optional[str] = None
    metadata: dict

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(req: CotizacionRequest) -> str:
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "construcción general"
    adicionales = ", ".join(req.servicios_adicionales) if req.servicios_adicionales else "ninguno"
    plazo_map = {"urgente": "Urgente", "normal": "Normal (1-4 semanas)", "largo": "Largo (+1 mes)"}
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

Genera cotización completa con máximo 5 fases y 3-4 conceptos por fase."""

def extract_total(text: str) -> Optional[str]:
    match = re.search(r'\*{0,2}TOTAL[^$\d]*\$?([\d,]+(?:\.\d{2})?)\s*MXN', text, re.IGNORECASE)
    return f"${match.group(1)} MXN" if match else None

def clean_json(raw: str) -> str:
    return raw.replace("```json","").replace("```","").strip()

# ── Endpoints públicos ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"api": "Cotizador Inteligente", "version": "7.0.0", "empresa": "Hogar 911 / Dacam Constructora"}

@app.get("/health")
def health():
    return {"status": "ok", "anthropic_configured": bool(os.environ.get("ANTHROPIC_API_KEY")), "db": bool(DATABASE_URL)}

# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/login")
async def login(req: LoginRequest):
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DB no configurada")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        user = await conn.fetchrow("SELECT * FROM usuarios_cotizador WHERE email=$1 AND activo=true", req.email)
        if not user:
            raise HTTPException(status_code=401, detail="Usuario no encontrado")
        ph = hash_pw(req.password)
        stored = user["password_hash"]
        if stored != ph and not stored.startswith("$2b$"):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        if stored.startswith("$2b$"):
            await conn.execute("UPDATE usuarios_cotizador SET password_hash=$1 WHERE id=$2", ph, user["id"])
        token = make_token(user["id"], user["email"], user["nombre"], user["rol"])
        return {"token": token, "usuario": {"id": user["id"], "nombre": user["nombre"], "email": user["email"], "rol": user["rol"]}}
    finally:
        await conn.close()

@app.post("/auth/usuarios")
async def crear_usuario(req: UsuarioCreate, current_user: dict = Depends(get_current_user)):
    if current_user.get("rol") != "admin":
        raise HTTPException(status_code=403, detail="Solo admins")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        user = await conn.fetchrow(
            "INSERT INTO usuarios_cotizador (nombre, email, password_hash, rol) VALUES ($1,$2,$3,$4) RETURNING id, nombre, email, rol",
            req.nombre, req.email, hash_pw(req.password), req.rol
        )
        return dict(user)
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=400, detail="El email ya existe")
    finally:
        await conn.close()

@app.get("/auth/usuarios")
async def listar_usuarios(current_user: dict = Depends(get_current_user)):
    if current_user.get("rol") != "admin":
        raise HTTPException(status_code=403, detail="Solo admins")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        users = await conn.fetch("SELECT id, nombre, email, rol, activo, created_at FROM usuarios_cotizador ORDER BY created_at DESC")
        return [dict(u) for u in users]
    finally:
        await conn.close()

@app.delete("/auth/usuarios/{user_id}")
async def desactivar_usuario(user_id: int, current_user: dict = Depends(get_current_user)):
    if current_user.get("rol") != "admin":
        raise HTTPException(status_code=403, detail="Solo admins")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("UPDATE usuarios_cotizador SET activo=false WHERE id=$1", user_id)
        return {"ok": True}
    finally:
        await conn.close()

# ── Cotizaciones en nube ──────────────────────────────────────────────────────

@app.get("/cotizaciones")
async def listar_cotizaciones(current_user: dict = Depends(get_current_user)):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if current_user.get("rol") == "admin":
            rows = await conn.fetch("SELECT id, nombre_cliente, nombre_obra, tipos_trabajo, metros_cuadrados, ubicacion, total_estimado, creado_por, created_at FROM cotizaciones ORDER BY created_at DESC")
        else:
            rows = await conn.fetch("SELECT id, nombre_cliente, nombre_obra, tipos_trabajo, metros_cuadrados, ubicacion, total_estimado, creado_por, created_at FROM cotizaciones WHERE creado_por=$1 ORDER BY created_at DESC", current_user.get("email"))
        return [dict(r) for r in rows]
    finally:
        await conn.close()

@app.post("/cotizaciones/guardar")
async def guardar_cotizacion(req: CotizacionGuardar, current_user: dict = Depends(get_current_user)):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow(
            "INSERT INTO cotizaciones (nombre_cliente, nombre_obra, tipos_trabajo, metros_cuadrados, ubicacion, total_estimado, contenido, creado_por) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id",
            req.nombre_cliente, req.nombre_obra, req.tipos_trabajo, req.metros_cuadrados,
            req.ubicacion, req.total_estimado, req.contenido, current_user.get("email")
        )
        return {"ok": True, "id": row["id"]}
    finally:
        await conn.close()

@app.get("/cotizaciones/{cot_id}")
async def obtener_cotizacion(cot_id: int, current_user: dict = Depends(get_current_user)):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow("SELECT * FROM cotizaciones WHERE id=$1", cot_id)
        if not row:
            raise HTTPException(status_code=404, detail="No encontrada")
        return dict(row)
    finally:
        await conn.close()

@app.delete("/cotizaciones/{cot_id}")
async def eliminar_cotizacion(cot_id: int, current_user: dict = Depends(get_current_user)):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("DELETE FROM cotizaciones WHERE id=$1", cot_id)
        return {"ok": True}
    finally:
        await conn.close()

# ── Cotizar endpoints (con auth) ──────────────────────────────────────────────

@app.post("/cotizar", response_model=CotizacionResponse)
def cotizar(req: CotizacionRequest, current_user: dict = Depends(get_current_user)):
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
                                            "tipos_trabajo": req.tipos_trabajo, "ubicacion": req.ubicacion})
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error de IA: {str(e)}")

@app.post("/cotizar/stream")
def cotizar_stream(req: CotizacionRequest, current_user: dict = Depends(get_current_user)):
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
def cotizar_rapida(req: CotizacionRequest, current_user: dict = Depends(get_current_user)):
    if req.metros_cuadrados <= 0:
        raise HTTPException(status_code=422, detail="metros_cuadrados debe ser mayor a 0")
    tipos = ", ".join(req.tipos_trabajo) if req.tipos_trabajo else "construcción general"
    prompt = f"""Estimación rápida: {tipos}, {req.metros_cuadrados}m², {req.niveles} nivel(es), {req.ubicacion}.
JSON solo: {{"rango_minimo":0,"rango_maximo":0,"moneda":"MXN","precio_por_m2_min":0,"precio_por_m2_max":0,"notas":"texto"}}"""
    try:
        msg = client.messages.create(model=MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
        return {"estimacion": json.loads(clean_json(msg.content[0].text)), "metros_cuadrados": req.metros_cuadrados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/cotizar/plano")
async def cotizar_plano(
    archivo: UploadFile = File(...),
    nombre_cliente: str = Form(default=""),
    nombre_obra: str = Form(default=""),
    ubicacion: str = Form(default="Ciudad de México"),
    plazo: str = Form(default="normal"),
    notas_adicionales: str = Form(default=""),
    current_user: dict = Depends(get_current_user)
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
