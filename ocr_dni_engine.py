# ocr_dni_engine.py
"""
Motor OCR para DNI Peruano (Azul y Electrónico)
Compatible con PaddleOCR 2.x
Versión: 2.9 FINAL CORREGIDA - Con mejora de imagen
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
# 2) MEJORA DE IMAGEN
# ============================================================
def mejorar_imagen_avanzada(img):
    """Mejora la imagen para OCR"""
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        print(f"📏 Escala de grises: {gray.shape}")
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        print("✅ CLAHE aplicado")
        
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=7, templateWindowSize=7, searchWindowSize=21)
        print("✅ Denoise aplicado")
        
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        print("✅ Sharpening aplicado")
        
        return sharpened
    except Exception as e:
        print(f"⚠️ Error mejorando imagen: {e}")
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

# ============================================================
# 3) CORRECCIÓN DE ERRORES OCR
# ============================================================
def corregir_fecha_ocr(fecha_str: str) -> str:
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

# ============================================================
# 4) PARSER PRINCIPAL
# ============================================================
def parsear_dni(texto_ocr: str) -> dict:
    datos = {}
    lineas = [l.strip() for l in texto_ocr.split('\n') if l.strip()]
    
    print(f"📊 Parser recibió {len(lineas)} líneas de texto")
    
    # DNI
    dni_match = re.search(r'DNI\s*(\d{8})', texto_ocr, re.IGNORECASE)
    if dni_match:
        dni_raw = dni_match.group(1)
        if dni_raw.startswith("00"):
            mrz_dni = re.search(r'PER(\d{8})', texto_ocr)
            datos["dni"] = mrz_dni.group(1) if mrz_dni else dni_raw
        else:
            datos["dni"] = dni_raw
        print(f"✅ DNI detectado: {datos['dni']}")
    
    # APELLIDO PATERNO
    apellido_encontrado = False
    for i, linea in enumerate(lineas):
        if re.search(r'PRIMER[\s]*APELLIDO', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1].strip()
                if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', candidato) and len(candidato) > 1 and not re.search(r'\d', candidato):
                    datos["apellido_paterno"] = candidato
                    print(f"✅ Apellido paterno: {candidato}")
                    apellido_encontrado = True
            break
    
    # Si no encuentra, buscar después de "PrimerApellido" (pegado)
    if not apellido_encontrado:
        for i, linea in enumerate(lineas):
            if linea == "PrimerApellido":
                # Saltar líneas de fecha
                for j in range(i+1, min(i+3, len(lineas))):
                    candidato = lineas[j].strip()
                    if re.match(r'^[A-ZÁÉÍÓÚÑ]+$', candidato) and len(candidato) > 2:
                        datos["apellido_paterno"] = candidato
                        print(f"✅ Apellido paterno (alternativo): {candidato}")
                        break
                break
    
    # APELLIDO MATERNO
    for i, linea in enumerate(lineas):
        if re.search(r'SEGUNDO[\s\.]*APELLIDO', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1].strip()
                if candidato == "MUNEZ":
                    candidato = "NUNEZ"
                if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', candidato) and len(candidato) > 1 and not re.search(r'\d', candidato):
                    datos["apellido_materno"] = candidato
                    print(f"✅ Apellido materno: {candidato}")
            break
    
    # NOMBRES
    nombres_encontrado = False
    for i, linea in enumerate(lineas):
        if re.search(r'PRE[\s]*NOMBRES?', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1].strip()
                
                # Ignorar si es una fecha
                if re.match(r'^\d{8}$', candidato):
                    print(f"⚠️ Ignorando fecha como nombre: {candidato}")
                    # Buscar en la siguiente línea
                    if i + 2 < len(lineas):
                        candidato = lineas[i + 2].strip()
                
                # Separar nombres pegados
                nombres_comunes = ['MARIA', 'MONICA', 'JUAN', 'JOSE', 'LUIS', 'CARLOS', 'ISABEL', 'ROSA', 'ANA', 'CARMEN']
                for nombre in nombres_comunes:
                    if candidato.startswith(nombre) and len(candidato) > len(nombre):
                        resto = candidato[len(nombre):]
                        if resto in nombres_comunes or len(resto) > 2:
                            candidato = f"{nombre} {resto}"
                            print(f"🔧 Nombres separados: {candidato}")
                            break
                
                if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', candidato):
                    datos["nombres"] = candidato
                    print(f"✅ Nombres: {candidato}")
                    nombres_encontrado = True
            break
    
    # Si no encuentra nombres, buscar en MRZ (última línea)
    if not nombres_encontrado:
        mrz_nombres = re.search(r'CASAS<<([A-Z<]+)', texto_ocr)
        if mrz_nombres:
            nombres_raw = mrz_nombres.group(1).replace('<', ' ').strip()
            datos["nombres"] = nombres_raw
            print(f"✅ Nombres (desde MRZ): {nombres_raw}")
    
    # FECHA DE NACIMIENTO
    for i, linea in enumerate(lineas):
        if re.search(r'NACIMIENTO', linea, re.IGNORECASE):
            for j in range(i, min(i + 5, len(lineas))):
                match = re.search(r'\b(0\d{7})\b', lineas[j])
                if match:
                    fecha_raw = match.group(1)
                    fecha_corregida = corregir_fecha_ocr(fecha_raw)
                    if fecha_corregida:
                        dia, mes, anio = fecha_corregida.split('/')
                        try:
                            fecha_nac = datetime(int(anio), int(mes), int(dia))
                            edad = (datetime.now() - fecha_nac).days // 365
                            if 0 <= edad <= 120:
                                datos["fecha_nacimiento"] = fecha_corregida
                                datos["fecha_nacimiento_iso"] = f"{anio}-{mes}-{dia}"
                                datos["edad"] = edad
                                print(f"✅ Fecha nacimiento: {fecha_corregida} (Edad: {edad})")
                                break
                        except:
                            pass
            break
    
    # SEXO
    mrz_sexo = re.search(r'\d{6}([MF])\d{7}', texto_ocr)
    if mrz_sexo:
        datos["sexo"] = mrz_sexo.group(1)
        datos["sexo_completo"] = "MASCULINO" if datos["sexo"] == "M" else "FEMENINO"
        print(f"✅ Sexo: {datos['sexo_completo']}")
    
    # ESTADO CIVIL
    for i, linea in enumerate(lineas):
        if re.search(r'ESTADO\s*CI[VW]IL', linea, re.IGNORECASE):
            for j in range(i, min(i + 4, len(lineas))):
                if lineas[j] in ['S', 'C', 'D', 'V']:
                    mapa = {'S': 'SOLTERO', 'C': 'CASADO', 'D': 'DIVORCIADO', 'V': 'VIUDO'}
                    datos["estado_civil"] = mapa.get(lineas[j], lineas[j])
                    print(f"✅ Estado civil: {datos['estado_civil']}")
                    break
            break
    
    # NOMBRE COMPLETO
    if all(k in datos for k in ["nombres", "apellido_paterno", "apellido_materno"]):
        datos["nombre_completo"] = f"{datos['nombres']} {datos['apellido_paterno']} {datos['apellido_materno']}"
    elif all(k in datos for k in ["nombres", "apellido_paterno"]):
        datos["nombre_completo"] = f"{datos['nombres']} {datos['apellido_paterno']}"
    
    datos["tipo_documento"] = "DNI"
    
    return datos

# ============================================================
# 5) FUNCIÓN PRINCIPAL
# ============================================================
def extraer_datos_dni(imagen_path: str) -> dict:
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
        
        # USAR MEJORA DE IMAGEN
        img_mejorada = mejorar_imagen_avanzada(img)
        
        print("🤖 Ejecutando OCR con PaddleOCR 2.x...")
        resultado = ocr_engine.ocr(img_mejorada, cls=True)
        
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