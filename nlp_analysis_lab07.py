import os
import time
import json
import re
import concurrent.futures
import pandas as pd
import requests
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- CONFIGURACIÓN DE APIs ---

# 1. GEMINI (Google) - Para LinkedIn
_model_gemini = None

def _get_gemini_model(api_key):
    global _model_gemini
    if _model_gemini is None:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        }

        system_instruction = (
            "Eres un experto en análisis de sentimientos para redes sociales. "
            "Devuelve SOLO JSON válido con: "
            "sentimiento (Positivo/Negativo/Neutro), explicacion (breve razón). "
        )

        # ### CAMBIO IMPORTANTE 1: Usamos el modelo LITE que permite 4000 RPM (peticiones x minuto)
        # Si este falla, prueba con "gemini-2.0-flash-lite-preview-02-05" o "gemini-1.5-flash"
        _model_gemini = genai.GenerativeModel(
            model_name="gemini-flash-latest", 
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
    return _model_gemini

def analyze_with_gemini(text):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return "ERROR: No API Key", 0

    start_time = time.time()
    try:
        model = _get_gemini_model(api_key)
        
        # ### CAMBIO IMPORTANTE 2: Eliminamos el sleep de 4 segundos.
        # Con el modelo Lite, el limite es tan alto que no hace falta esperar tanto.
        # time.sleep(4)  <-- ELIMINADO
        
        response = model.generate_content(
            f"""Analiza este comentario de LinkedIn: "{text}" """
        )
        
        result = response.text.replace('```json', '').replace('```', '').strip()
        end_time = time.time()
        return result, end_time - start_time
    except Exception as e:
        # Solo dormimos un poco si hay un error real (ej. saturación)
        time.sleep(1) 
        return f"ERROR: {str(e)}", time.time() - start_time

# 2. GROQ - Para X (Twitter)
def analyze_with_groq(text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return "ERROR: No API Key", 0

    start_time = time.time()
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de sentimientos. Responde SOLO en JSON."},
                {"role": "user", "content": f"""Analiza este tweet: "{text}".
                Formato JSON esperado: {{"sentimiento": "Positivo|Negativo|Neutro", "explicacion": "breve razón"}}"""}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = completion.choices[0].message.content
        end_time = time.time()
        return result, end_time - start_time
    except Exception as e:
        return f"ERROR: {str(e)}", time.time() - start_time

# 3. OPENROUTER - Para Facebook
def analyze_with_openrouter(text):
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"
    if not api_key: return "ERROR: No OpenRouter API Key", 0

    start_time = time.time()
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-distill-llama-70b", 
            messages=[
                {"role": "system", "content": "Eres un analista de opiniones. Responde siempre en JSON puro."},
                {"role": "user", "content": f"""Analiza el sentimiento de este post de Facebook: "{text}".
                JSON: {{"sentimiento": "Positivo/Negativo/Neutro", "explicacion": "breve razón"}}"""}
            ],
            extra_body={
                "headers": {
                    "HTTP-Referer": "https://localhost", 
                    "X-Title": "NLP Lab 07",
                }
            },
            stream=False
        )
        result = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
        end_time = time.time()
        return result, end_time - start_time
    except Exception as e:
        return f"ERROR: {str(e)}", time.time() - start_time

# 4. CLOUDFLARE WORKERS AI - Para Instagram
def analyze_with_cloudflare(text):
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token: return "ERROR: No API Credentials", 0

    start_time = time.time()
    try:
        model = "@cf/meta/llama-3-8b-instruct"
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        prompt = f"""Analiza el sentimiento de este comentario de Instagram: "{text}".
        Responde estrictamente con un objeto JSON: {{"sentimiento": "Positivo|Negativo|Neutro", "explicacion": "razon muy breve (max 10 palabras)"}}"""
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a sentiment analysis bot. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json()
        
        if response_json.get('success'):
            result = response_json['result']['response'].strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].strip()
            
            end_time = time.time()
            return result, end_time - start_time
        else:
            return f"ERROR: {response_json.get('errors')}", time.time() - start_time
            
    except Exception as e:
        return f"ERROR: {str(e)}", time.time() - start_time


# --- CONTROLADOR CENTRAL ---

def process_row(row):
    """Procesa una fila del DataFrame y selecciona el modelo según el origen."""
    origin = str(row['origen']).lower().strip()
    text = str(row['texto'])
    
    if len(text) < 5: 
        return None 

    result_raw = "{}"
    duration = 0
    model_used = "Unknown"

    # Mapeo según Hoja de Ruta (Lab 07)
    if 'linkedin' in origin:
        result_raw, duration = analyze_with_gemini(text)
        model_used = "Gemini (LinkedIn)"
    elif 'x' in origin or 'twitter' in origin:
        result_raw, duration = analyze_with_groq(text)
        model_used = "Groq (X)"
    elif 'facebook' in origin or 'fb' in origin:
        result_raw, duration = analyze_with_openrouter(text)
        model_used = "OpenRouter (Facebook)"
    elif 'instagram' in origin or 'ig' in origin:
        result_raw, duration = analyze_with_cloudflare(text)
        model_used = "Cloudflare (Instagram)"
    else:
        result_raw, duration = analyze_with_gemini(text)
        model_used = "Gemini (Default)"

    import re
    import ast

    def robust_json_parse(text):
        """Intenta parsear JSON de multiples formas."""
        text = text.strip()
        
        # 0. Limpieza básica de caracteres problematicos
        text = text.replace("“", '"').replace("”", '"') # Smart quotes
        
        # Estrategia 1: JSON Directo
        try:
            return json.loads(text)
        except:
            pass

        # Estrategia 2: Regex para extraer bloque JSON
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
        except:
            pass
            
        # Estrategia 3: Python Eval (para single quotes o propiedades sin comillas a veces)
        try:
            # Intentar encontrar estructura de dict si regex fallo o json fallo
            match = re.search(r"\{.*\}", text, re.DOTALL)
            clean_text = match.group() if match else text
            return ast.literal_eval(clean_text)
        except:
            pass
            
        return None

    # Intentar parsear
    parsed = robust_json_parse(result_raw)

    if parsed and isinstance(parsed, dict):
        sentiment = parsed.get("sentimiento", "Desconocido")
        explanation = parsed.get("explicacion", "Sin explicación")
    else:
        sentiment = "Error Parsing"
        explanation = result_raw 

    return {
        "texto_original": text,
        "origen": origin,
        "modelo": model_used,
        "sentimiento": sentiment,
        "explicacion": explanation,
        "tiempo_ejecucion": round(duration, 4)
    }

def main():
    print("🚀 INICIANDO ANÁLISIS NLP PARALELO (LAB 07) - MODO ULTRA-RÁPIDO")
    print("---------------------------------------------")
    
    # Simulación de datos si no existe el archivo (para que pruebes el código ya)
    try:
        df = pd.read_csv('resultados_finales_grado.csv')
        print(f"✅ Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("⚠️ No se encontró CSV, creando datos de prueba...")
        data_dummy = {
            'origen': ['LinkedIn', 'X', 'Facebook', 'Instagram', 'LinkedIn'] * 4,
            'texto': ['Excelente servicio', 'Odio esto', 'Muy normal', 'Me encanta', 'Podría mejorar'] * 4
        }
        df = pd.DataFrame(data_dummy)

    start_global = time.time()
    results = []
    
    print("⏳ Procesando comentarios en PARALELO...")
    
    # ### CAMBIO IMPORTANTE 3: Aumentamos workers a 10
    # Gemini 2.5 Flash Lite aguanta esto sin problemas.
    MAX_WORKERS = 10 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rows = df.to_dict('records')
        future_to_row = {executor.submit(process_row, row): row for row in rows}
        
        completed = 0
        total = len(rows)
        
        for future in concurrent.futures.as_completed(future_to_row):
            data = future.result()
            if data:
                results.append(data)
            
            completed += 1
            if completed % 5 == 0:
                print(f"   ... {completed}/{total} procesados.")

    end_global = time.time()
    total_time = end_global - start_global

    # 4. Guardar Resultados
    results_df = pd.DataFrame(results)
    output_file = 'resultados_nlp_benchmark.csv'
    results_df.to_csv(output_file, index=False)
    
    # 5. Generar Reporte
    print("\n" + "="*50)
    print("📊 REPORTE DE RESULTADOS")
    print("="*50)
    print(f"Tiempo Total: {total_time:.2f} s")
    print(f"Registros: {len(results_df)}")
    print(f"Velocidad: {len(results_df)/total_time:.2f} regs/seg")
    print("-" * 30)
    
    if not results_df.empty:
        print("\nTIMING PROMEDIO (segundos):")
        print(results_df.groupby('modelo')['tiempo_ejecucion'].mean())

        print("\nDISTRIBUCIÓN DE SENTIMIENTOS GLOBAL:")
        print(results_df['sentimiento'].value_counts())
        
        print("\nDETALLE POR MODELO:")
        # Pivot table for cleaner view or just groupby
        try:
            print(results_df.groupby(['modelo', 'sentimiento']).size().unstack(fill_value=0))
        except:
            print(results_df.groupby(['modelo', 'sentimiento']).size())

if __name__ == "__main__":
    main()