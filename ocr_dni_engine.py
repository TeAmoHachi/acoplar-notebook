# ocr_dni_engine.py
"""
Motor OCR para DNI Peruano (Azul y Electrónico)
Compatible con PaddleOCR 2.x
Versión: 3.1 FINAL CORREGIDA - Parser híbrido (MRZ + texto)
Autor: TeAmoHachi
Fecha: 2025-11-14
"""

import cv2
import re
import os
from datetime import datetime
from paddleocr import PaddleOCR
import numpy as np

# ============================================================
# 1) INICIALIZACIÓN DEL OCR
# ============================================================
ocr_engine = None
ocr_error_message = None

try:
    print("🔧 Inicializando PaddleOCR 2.x...")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='es',
        use_gpu=False,
        show_log=False
    )
    print("✅ PaddleOCR 2.x inicializado correctamente")
except Exception as e:
    ocr_error_message = f"Error inicializando OCR: {str(e)}"
    print(f"❌ {ocr_error_message}")

# ============================================================
# 2) CORRECCIÓN DE ERRORES OCR
# ============================================================
def corregir_fecha_ocr(fecha_str: str) -> str:
    """Corrige fechas mal leídas por el OCR"""
    if not fecha_str or len(fecha_str) != 8:
        return None
    try:
        dia = fecha_str[0:2]
        mes = fecha_str[2:4]
        anio = fecha_str[4:8]
        
        if dia.startswith('3') and int(dia) > 31:
            dia = '0' + dia[1]
        if mes == '19':
            mes = '10'
        if anio == '2062':
            anio = '2002'
        elif anio == '2919':
            anio = '2019'
        
        fecha = datetime(int(anio), int(mes), int(dia))
        return f"{dia}/{mes}/{anio}"
    except:
        return None

def separar_nombres_pegados(nombre: str) -> str:
    """
    Separa nombres comunes pegados.
    Ejemplos:
    - JORGELUIS → JORGE LUIS
    - MARIAISABEL → MARIA ISABEL
    - JUANCARLOS → JUAN CARLOS
    """
    nombres_comunes = [
        'JORGE', 'LUIS', 'CARLOS', 'JUAN', 'JOSE', 'MARIA', 
        'ISABEL', 'ROSA', 'ANA', 'CARMEN', 'MONICA', 'MILAGROS',
        'PATRICIA', 'VICTORIA', 'ELENA', 'SANDRA', 'DIANA', 'ADRIAN',
        'PEDRO', 'PABLO', 'MIGUEL', 'ANGEL', 'ANTONIO', 'MANUEL',
        'RICARDO', 'ROBERTO', 'DANIEL', 'DAVID', 'FRANCISCO'
    ]
    
    for nombre_comun in nombres_comunes:
        if nombre.startswith(nombre_comun) and len(nombre) > len(nombre_comun):
            resto = nombre[len(nombre_comun):]
            # Verificar que el resto también sea un nombre común O tenga más de 3 letras
            if resto in nombres_comunes or len(resto) >= 3:
                resultado = f"{nombre_comun} {resto}"
                print(f"🔧 Separando nombres: {nombre} → {resultado}")
                return resultado
    
    return nombre

# ============================================================
# 3) PARSER PRINCIPAL (HÍBRIDO: MRZ + TEXTO)
# ============================================================
def parsear_dni(texto_ocr: str) -> dict:
    """
    Extrae datos del DNI de forma ROBUSTA:
    - Intenta MRZ primero (más confiable)
    - Si falla, busca en TEXTO con múltiples patrones
    - Tolerante a marcas, deterioro, reflejos
    """
    datos = {}
    lineas = [l.strip() for l in texto_ocr.split('\n') if l.strip()]
    
    print(f"📊 Parser recibió {len(lineas)} líneas de texto")
    
    # ═══════════════════════════════════════════════════════
    # 1️⃣ DNI (múltiples estrategias)
    # ═══════════════════════════════════════════════════════
    
    # Estrategia 1: MRZ (más confiable)
    mrz_dni_match = re.search(r'PER(\d{8})', texto_ocr)
    if mrz_dni_match:
        datos["dni"] = mrz_dni_match.group(1)
        print(f"✅ DNI detectado (MRZ): {datos['dni']}")
    
    # Estrategia 2: Buscar "DNI" seguido de 8 dígitos
    if "dni" not in datos:
        dni_match = re.search(r'DNI\s*(\d{8})', texto_ocr, re.IGNORECASE)
        if dni_match:
            datos["dni"] = dni_match.group(1)
            print(f"✅ DNI detectado (TEXTO): {datos['dni']}")
    
    # Estrategia 3: Buscar 8 dígitos solos (con guion o sin él)
    if "dni" not in datos:
        dni_solo = re.search(r'\b(\d{8})[-\s]?\d?\b', texto_ocr)
        if dni_solo:
            datos["dni"] = dni_solo.group(1)
            print(f"✅ DNI detectado (FALLBACK): {datos['dni']}")
    
    # ═══════════════════════════════════════════════════════
    # 2️⃣ APELLIDOS (buscar palabras grandes en mayúsculas)
    # ═══════════════════════════════════════════════════════
    
    apellidos_encontrados = []
    
    # Buscar cerca de palabras clave: "Apellido", "Primer", "Segundo"
    for i, linea in enumerate(lineas):
        if re.search(r'(Primer|Segundo|Apellido)', linea, re.IGNORECASE):
            # Revisar las siguientes 5 líneas
            for j in range(i+1, min(i+6, len(lineas))):
                candidato = lineas[j].strip()
                
                # Ignorar fechas, DNIs, códigos
                if re.match(r'^\d{6,}$', candidato):
                    continue
                
                # Ignorar líneas muy cortas o con símbolos
                if len(candidato) < 3 or re.search(r'[^A-ZÁÉÍÓÚÑ\s]', candidato):
                    continue
                
                # Debe ser MAYÚSCULAS (apellidos)
                if re.match(r'^[A-ZÁÉÍÓÚÑ]+$', candidato):
                    if candidato not in apellidos_encontrados:  # Evitar duplicados
                        apellidos_encontrados.append(candidato)
                        print(f"✅ Apellido detectado: {candidato}")
    
    # Asignar apellidos (primeros 2)
    if len(apellidos_encontrados) >= 1:
        datos["apellido_paterno"] = apellidos_encontrados[0]
    if len(apellidos_encontrados) >= 2:
        datos["apellido_materno"] = apellidos_encontrados[1]
    
    # ═══════════════════════════════════════════════════════
    # 3️⃣ NOMBRES (MRZ + fallback a texto)
    # ═══════════════════════════════════════════════════════
    
    # Estrategia 1: MRZ (línea 3: APELLIDO<<NOMBRES<<)
    mrz_nombres = re.search(r'^[A-Z]{3,}<<([A-Z<]+)$', texto_ocr, re.MULTILINE)
    if mrz_nombres:
        nombres_raw = mrz_nombres.group(1).replace('<', ' ').strip()
        # Filtrar si los nombres no son solo espacios
        if nombres_raw and len(nombres_raw) > 2:
            datos["nombres"] = nombres_raw
            print(f"✅ Nombres (MRZ): {nombres_raw}")
    
    # Estrategia 2: Buscar "Pre Nombres" o "Nombres" en el texto
    if "nombres" not in datos:
        for i, linea in enumerate(lineas):
            if re.search(r'(Pre\s*)?Nombres', linea, re.IGNORECASE):
                # Revisar las siguientes 3 líneas
                for j in range(i+1, min(i+4, len(lineas))):
                    candidato = lineas[j].strip()
                    
                    # Ignorar fechas, DNIs
                    if re.match(r'^\d{6,}$', candidato):
                        continue
                    
                    # Debe tener al menos 4 caracteres y ser mayúsculas
                    if len(candidato) >= 4 and re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', candidato):
                        # Verificar que NO sea un apellido ya detectado
                        if candidato not in apellidos_encontrados:
                            # 🔥 NUEVO: Separar nombres pegados
                            candidato = separar_nombres_pegados(candidato)
                            datos["nombres"] = candidato.strip()
                            print(f"✅ Nombres (TEXTO): {candidato}")
                            break
                
                if "nombres" in datos:
                    break
    
    # Estrategia 3: Buscar palabras de 2+ términos en mayúsculas (ej: JORGE LUIS)
    if "nombres" not in datos:
        for linea in lineas:
            # Buscar líneas con 2+ palabras en mayúsculas (al menos 4 letras cada una)
            match = re.match(r'^([A-ZÁÉÍÓÚÑ]{4,}\s+[A-ZÁÉÍÓÚÑ]{4,})$', linea.strip())
            if match and linea.strip() not in apellidos_encontrados:
                datos["nombres"] = match.group(1)
                print(f"✅ Nombres (PATRÓN): {match.group(1)}")
                break
    
    # ═══════════════════════════════════════════════════════
    # 4️⃣ FECHA DE NACIMIENTO (robusta con corrección OCR)
    # ═══════════════════════════════════════════════════════
    
    fecha_encontrada = False
    
    # Buscar cerca de "Nacimiento"
    for i, linea in enumerate(lineas):
        if re.search(r'Nacimiento', linea, re.IGNORECASE):
            # Revisar las siguientes 5 líneas
            for j in range(i, min(i + 6, len(lineas))):
                # Buscar fechas de 8 dígitos (DDMMAAAA)
                match = re.search(r'\b(\d{8})\b', lineas[j])
                if match:
                    fecha_raw = match.group(1)
                    fecha_corregida = corregir_fecha_ocr(fecha_raw)
                    
                    if fecha_corregida:
                        dia, mes, anio = fecha_corregida.split('/')
                        try:
                            fecha_nac = datetime(int(anio), int(mes), int(dia))
                            edad = (datetime.now() - fecha_nac).days // 365
                            
                            # Validar edad razonable (0-120 años)
                            if 0 <= edad <= 120:
                                datos["fecha_nacimiento"] = fecha_corregida
                                datos["fecha_nacimiento_iso"] = f"{anio}-{mes}-{dia}"
                                datos["edad"] = edad
                                print(f"✅ Fecha nacimiento: {fecha_corregida} (Edad: {edad})")
                                fecha_encontrada = True
                                break
                        except Exception as e:
                            print(f"⚠️ Error validando fecha {fecha_corregida}: {e}")
            
            if fecha_encontrada:
                break
    
    # ═══════════════════════════════════════════════════════
    # 5️⃣ SEXO (MRZ + fallback a texto)
    # ═══════════════════════════════════════════════════════
    
    # Estrategia 1: MRZ (patrón: DDMMAA[M/F]AAAMMDD)
    mrz_sexo_match = re.search(r'(\d{6})([MF])(\d{7})', texto_ocr)
    if mrz_sexo_match:
        sexo = mrz_sexo_match.group(2)
        datos["sexo"] = sexo
        datos["sexo_completo"] = "MASCULINO" if sexo == "M" else "FEMENINO"
        print(f"✅ Sexo (MRZ): {datos['sexo_completo']}")
    
    # Estrategia 2: Buscar "Sexo" seguido de M o F
    if "sexo" not in datos:
        for i, linea in enumerate(lineas):
            if re.search(r'Sexo', linea, re.IGNORECASE):
                for j in range(i+1, min(i+3, len(lineas))):
                    if lineas[j] in ['M', 'F', 'MASCULINO', 'FEMENINO']:
                        sexo = 'M' if lineas[j] in ['M', 'MASCULINO'] else 'F'
                        datos["sexo"] = sexo
                        datos["sexo_completo"] = "MASCULINO" if sexo == "M" else "FEMENINO"
                        print(f"✅ Sexo (TEXTO): {datos['sexo_completo']}")
                        break
                break
    
        # ═══════════════════════════════════════════════════════
        # 6️⃣ ESTADO CIVIL (ULTRA ROBUSTO - múltiples estrategias)
        # ═══════════════════════════════════════════════════════
    
        # Estrategia 1: Buscar cerca de "Estado Civil" (con tolerancia a errores OCR)
        for i, linea in enumerate(lineas):
            # Tolerar: Estado Civil, Estado Civit, Estado Ciwil, Estado Civu, etc.
            if re.search(r'Estado\s*Ci[vwtu][a-z]*', linea, re.IGNORECASE):
                print(f"🔍 Buscando estado civil después de: {linea}")
            
                # Revisar las siguientes 8 líneas (más alcance)
                for j in range(i + 1, min(i + 9, len(lineas))):
                    linea_candidata = lineas[j].strip()
                
                    # Buscar letra sola (S, C, D, V)
                    if linea_candidata in ['S', 'C', 'D', 'V']:
                        mapa = {
                            'S': 'SOLTERO',
                            'C': 'CASADO',
                            'D': 'DIVORCIADO',
                            'V': 'VIUDO'
                        }
                        datos["estado_civil"] = mapa[linea_candidata]
                        print(f"✅ Estado civil (letra sola): {datos['estado_civil']}")
                        break
                
                    # Buscar letra dentro de una línea corta (ej: "S " o " S")
                    if len(linea_candidata) <= 3 and linea_candidata.upper() in ['S', 'C', 'D', 'V']:
                        mapa = {
                            'S': 'SOLTERO',
                            'C': 'CASADO',
                            'D': 'DIVORCIADO',
                            'V': 'VIUDO'
                        }
                        datos["estado_civil"] = mapa[linea_candidata.upper()]
                        print(f"✅ Estado civil (línea corta): {datos['estado_civil']}")
                        break
                
                    # Buscar patrón: "Estado Civil: S" o "Civil S" (todo junto)
                    match_junto = re.search(r'\b([SCDV])\b', linea_candidata)
                    if match_junto:
                        mapa = {
                            'S': 'SOLTERO',
                            'C': 'CASADO',
                            'D': 'DIVORCIADO',
                            'V': 'VIUDO'
                        }
                        datos["estado_civil"] = mapa[match_junto.group(1)]
                        print(f"✅ Estado civil (patrón): {datos['estado_civil']}")
                        break
            
                if "estado_civil" in datos:
                    break
    
        # Estrategia 2: Buscar "Sexo" y "Estado Civil" en la MISMA línea o líneas consecutivas
        if "estado_civil" not in datos:
            for i, linea in enumerate(lineas):
                # Patrón: "Sexo Estado Civil" seguido de "M S" o "F C"
                if re.search(r'(Sexo|Estado)', linea, re.IGNORECASE):
                    # Buscar en las siguientes 3 líneas
                    for j in range(i + 1, min(i + 4, len(lineas))):
                        # Patrón: línea con 1-3 letras que contenga S, C, D o V
                        if len(lineas[j]) <= 3:
                            for letra in ['S', 'C', 'D', 'V']:
                                if letra in lineas[j].upper():
                                    mapa = {
                                        'S': 'SOLTERO',
                                        'C': 'CASADO',
                                        'D': 'DIVORCIADO',
                                        'V': 'VIUDO'
                                    }
                                    datos["estado_civil"] = mapa[letra]
                                    print(f"✅ Estado civil (cerca de Sexo): {datos['estado_civil']}")
                                    break
                    
                        if "estado_civil" in datos:
                            break
                
                    if "estado_civil" in datos:
                        break
    
        # Estrategia 3: Buscar en TODO el texto (último recurso)
        if "estado_civil" not in datos:
            # Buscar líneas de 1 solo carácter con S, C, D, V
            for linea in lineas:
                if len(linea) == 1 and linea in ['S', 'C', 'D', 'V']:
                    # Verificar que esté cerca de palabras clave
                    idx = lineas.index(linea)
                    contexto = ' '.join(lineas[max(0, idx-3):idx+1])
                
                    if re.search(r'(Estado|Civil|Sexo)', contexto, re.IGNORECASE):
                        mapa = {
                            'S': 'SOLTERO',
                            'C': 'CASADO',
                            'D': 'DIVORCIADO',
                            'V': 'VIUDO'
                        }
                        datos["estado_civil"] = mapa[linea]
                        print(f"✅ Estado civil (fallback): {datos['estado_civil']}")
                        break
    
    # ═══════════════════════════════════════════════════════
    # 7️⃣ NOMBRE COMPLETO (construcción inteligente)
    # ═══════════════════════════════════════════════════════
    
    # Orden correcto: NOMBRES + APELLIDO_PATERNO + APELLIDO_MATERNO
    if all(k in datos for k in ["nombres", "apellido_paterno", "apellido_materno"]):
        datos["nombre_completo"] = f"{datos['nombres']} {datos['apellido_paterno']} {datos['apellido_materno']}"
    elif all(k in datos for k in ["nombres", "apellido_paterno"]):
        datos["nombre_completo"] = f"{datos['nombres']} {datos['apellido_paterno']}"
    elif "apellido_paterno" in datos and "apellido_materno" in datos:
        # Si solo hay apellidos (caso raro), úsalos
        datos["nombre_completo"] = f"{datos['apellido_paterno']} {datos['apellido_materno']}"
    
    if "nombre_completo" in datos:
        print(f"✅ Nombre completo: {datos['nombre_completo']}")
    else:
        print(f"⚠️ No se pudo construir nombre completo")
    
    datos["tipo_documento"] = "DNI"
    
    return datos

# ============================================================
# 4) FUNCIÓN PRINCIPAL
# ============================================================
def extraer_datos_dni(imagen_path: str) -> dict:
    """Función principal que extrae todos los datos del DNI"""
    if ocr_engine is None:
        return {"error": "OCR no inicializado"}
    
    try:
        print(f"\n{'='*70}")
        print(f"📸 PROCESANDO FRENTE DEL DNI")
        print(f"📸 Archivo: {os.path.basename(imagen_path)}")
        print(f"{'='*70}")
        
        img = cv2.imread(imagen_path)
        if img is None:
            return {"error": "No se pudo cargar la imagen"}
        
        print(f"✅ Imagen cargada: {img.shape[1]}x{img.shape[0]} px")
        
        print("🤖 Ejecutando OCR con imagen ORIGINAL...")
        resultado = ocr_engine.ocr(img, cls=True)
        
        if not resultado or not resultado[0]:
            return {"error": "No se pudo extraer texto"}
        
        texto_completo = '\n'.join([bloque[1][0] for bloque in resultado[0]])
        
        print(f"\n{'='*70}")
        print(f"📄 TEXTO EXTRAÍDO ({len(texto_completo)} caracteres):")
        print(f"{'='*70}")
        print(texto_completo[:800])
        print(f"{'='*70}\n")
        
        datos = parsear_dni(texto_completo)
        
        if not datos.get("dni"):
            return {"error": "No se pudo detectar el DNI"}
        
        print(f"\n{'='*70}")
        print(f"✅ EXTRACCIÓN EXITOSA")
        print(f"{'='*70}")
        print(f"   DNI: {datos.get('dni')}")
        print(f"   Nombre: {datos.get('nombre_completo', 'N/A')}")
        print(f"   Fecha Nac: {datos.get('fecha_nacimiento', 'N/A')}")
        print(f"   Edad: {datos.get('edad', 'N/A')} años")
        print(f"   Sexo: {datos.get('sexo_completo', 'N/A')}")
        print(f"   Estado Civil: {datos.get('estado_civil', 'N/A')}")
        print(f"{'='*70}\n")
        
        return datos
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Error procesando DNI: {str(e)}"}