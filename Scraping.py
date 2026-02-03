import pickle
import time
import threading
from selenium import webdriver
import pickle
import time
import random
import csv
import os
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import emoji
import nltk
import threading
from collections import Counter
from playwright.async_api import async_playwright
from wordcloud import WordCloud
import os
import re
import json
import unicodedata
import concurrent.futures
import ast
import asyncio
import nest_asyncio

# Bibliotecas Externas
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import emoji
import requests
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams

# --- CONFIGURACIÓN DE NLTK ---
# Cargar variables de entorno
load_dotenv()

# Descarga de recursos NLTK
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Error NLTK: {e}")

nest_asyncio.apply()

# --- UTILIDAD: CONVERTIDOR DE COOKIES PKL A PLAYWRIGHT ---

async def inyectar_cookies_pkl(contexto, archivo_pkl):
    """Traduce tus .pkl de Selenium al formato de Playwright"""
    if os.path.exists(archivo_pkl):
        try:
            with open(archivo_pkl, "rb") as f:
                cookies_selenium = pickle.load(f)
                cookies_playwright = []
                for c in cookies_selenium:
                    cookie_limpia = {
                        'name': c['name'],
                        'value': c['value'],
                        'domain': c['domain'],
                        'path': c['path'] if 'path' in c else '/',
                        'secure': c['secure'] if 'secure' in c else True,
                        'httpOnly': c['httpOnly'] if 'httpOnly' in c else False,
                        'sameSite': 'Lax'
                    }
                    cookies_playwright.append(cookie_limpia)
                await contexto.add_cookies(cookies_playwright)
                return True
        except Exception as e:
            print(f"Error procesando {archivo_pkl}: {e}")
    return False

import os
import csv

def guardar_comentario(archivo, datos, encabezado):
    """
    Versión robusta: Asegura la escritura inmediata en disco (Disk Sync).
    """
    try:
        # 1. Verificamos si existe
        file_exists = os.path.isfile(archivo)
        
        # 2. Abrimos con buffering=0 no es posible en modo texto, 
        # así que usamos flush() manual.
        with open(archivo, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 3. Encabezado
            if not file_exists:
                writer.writerow(encabezado)
            
            # 4. Escribir datos
            writer.writerow(datos)
            
            # 5. EL TRUCO: Forzar a Windows a escribir en el disco AHORA
            f.flush()
            os.fsync(f.fileno()) 
            
    except Exception as e:
        print(f"❌ Error crítico al escribir en {archivo}: {e}")


# --- FASE 2: SCRAPING (PLAYWRIGHT + PKL + DEDUPLICACIÓN) ---

async def scrap_linkedin_playwright(tema):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1280, 'height': 900})
        
        if await inyectar_cookies_pkl(context, "linkedin_cookies.pkl"):
            page = await context.new_page()
            url = f"https://www.linkedin.com/search/results/content/?keywords={tema.replace(' ', '%20')}"
            print(f"🚀 [LinkedIn] Iniciando en: {url}")
            await page.goto(url)
            await page.wait_for_timeout(7000)

            vistos = set()
            posts_ya_leidos = set()

            with open('comentarios_linkedin.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if os.stat('comentarios_linkedin.csv').st_size == 0:
                    writer.writerow(["texto"])

                for scroll in range(4):
                    posts = page.locator('div[role="listitem"]')
                    total_posts = await posts.count()
                    print(f"⏳ Iteración {scroll+1}: {total_posts} posts detectados.")

                    for i in range(total_posts):
                        try:
                            post_actual = posts.nth(i)
                            ref_text = await post_actual.inner_text()
                            post_id = hash(ref_text[:100])
                            
                            if post_id in posts_ya_leidos: continue

                            # 1. CLICK EN ABRIR COMENTARIOS
                            boton_abrir = post_actual.locator('div[data-view-name="feed-comment-count"]')
                            if await boton_abrir.count() > 0:
                                await post_actual.scroll_into_view_if_needed()
                                await boton_abrir.first.evaluate("el => el.click()")
                                await page.wait_for_timeout(3000)

                                # 2. CAMBIAR A "MÁS RECIENTES" (Selector que enviaste)
                                try:
                                    sort_toggle = post_actual.locator('div[data-view-name="comment-sort-toggle"]')
                                    if await sort_toggle.count() > 0:
                                        await sort_toggle.click()
                                        await page.wait_for_timeout(1500)
                                        # Click en la opción del menú que aparece
                                        await page.get_by_text("Más recientes").first.click()
                                        await page.wait_for_timeout(2500)
                                except: pass

                                # 3. EXPANDIR HILO (Cargar más comentarios)
                                while True:
                                    boton_mas = post_actual.locator('button[data-view-name="more-comments"]')
                                    if await boton_mas.count() > 0 and await boton_mas.is_visible():
                                        await boton_mas.first.click()
                                        await page.wait_for_timeout(2500)
                                    else:
                                        break

                                # 4. EXTRACCIÓN (Selector exacto de tu HTML)
                                selector_txt = 'p[data-view-name="comment-commentary"] span[data-testid="expandable-text-box"]'
                                elementos = await post_actual.locator(selector_txt).all()
                                
                                nuevos = 0
                                for el in elementos:
                                    t = (await el.inner_text()).strip().replace('\n', ' ')
                                    if len(t) > 15 and t not in vistos:
                                        vistos.add(t)
                                        writer.writerow([t])
                                        nuevos += 1
                                
                                if nuevos > 0:
                                    f.flush()
                                    os.fsync(f.fileno())
                                    print(f"      ✅ Post {i+1}: {nuevos} comentarios nuevos.")
                                
                                posts_ya_leidos.add(post_id)

                        except Exception: continue

                    await page.keyboard.press("PageDown")
                    await page.wait_for_timeout(3000)

            print(f"✨ LinkedIn finalizado. Total: {len(vistos)}")
        await browser.close()

async def scrap_facebook_playwright(tema):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1280, 'height': 900})
        
        if await inyectar_cookies_pkl(context, "facebook_cookies.pkl"):
            page = await context.new_page()
            url = f"https://www.facebook.com/search/posts/?q={tema.replace(' ', '%20')}"
            print(f"🚀 [Facebook] Iniciando extracción por posición de contenedor...")
            await page.goto(url)
            await page.wait_for_timeout(8000)

            vistos = set()
            ultimo_posinset = 0

            # Abrimos el archivo una sola vez
            with open('comentarios_fb.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                for scroll_principal in range(15):
                    # Localizamos los contenedores específicos que me pasaste (aria-posinset)
                    selector_pos = 'div[aria-posinset]'
                    posts = page.locator(selector_pos)
                    cantidad_actual = await posts.count()
                    
                    print(f"⏳ Iteración {scroll_principal + 1}: {cantidad_actual} publicaciones en el feed.")

                    for i in range(cantidad_actual):
                        try:
                            post_actual = posts.nth(i)
                            
                            # Extraemos el número de posición para no repetir
                            pos_attr = await post_actual.get_attribute("aria-posinset")
                            pos_actual_num = int(pos_attr) if pos_attr else 0

                            if pos_actual_num <= ultimo_posinset:
                                continue # Ya procesamos esta posición

                            # 1. CLIC EN EL BOTÓN DE COMENTARIOS
                            # Buscamos el botón dentro de ESTE contenedor específico
                            boton_comentarios = post_actual.locator('div[role="button"]').filter(has_text=re.compile(r"\d+.*comentario", re.I))
                            
                            if await boton_comentarios.count() > 0:
                                await post_actual.scroll_into_view_if_needed()
                                await page.wait_for_timeout(1000)
                                
                                # Clic forzado
                                await boton_comentarios.first.click(force=True)
                                print(f"   👉 Entrando al Post posición {pos_actual_num}...")
                                await page.wait_for_timeout(4500)

                                # 2. SCROLL INTERNO EN EL DIÁLOGO (Fuerza carga de hilos)
                                # El diálogo es un modal que aparece con role="dialog"
                                for _ in range(4): # 4 segundos de scroll intenso
                                    await page.mouse.wheel(0, 1000)
                                    await page.wait_for_timeout(1000)

                                # 3. COPIAR TEXTOS (Selector de tu HTML previo)
                                selector_txt = 'div[role="dialog"] div.x1lliihq.xjkvuk6.x1iorvi4'
                                elementos = await page.locator(selector_txt).all()
                                
                                nuevos = 0
                                for el in elementos:
                                    t = (await el.inner_text()).strip().replace('\n', ' ')
                                    if len(t) > 12 and t not in vistos:
                                        vistos.add(t)
                                        writer.writerow([t])
                                        nuevos += 1
                                
                                if nuevos > 0:
                                    f.flush()
                                    os.fsync(f.fileno())
                                    print(f"      ✅ {nuevos} comentarios nuevos guardados.")

                                # 4. SALIR Y ACTUALIZAR MARCADOR
                                await page.keyboard.press("Escape")
                                await page.wait_for_timeout(1500)
                            
                            ultimo_posinset = pos_actual_num

                        except Exception:
                            await page.keyboard.press("Escape")
                            continue

                    # Bajar el scroll principal para que Facebook cargue más aria-posinset
                    await page.keyboard.press("End")
                    await page.wait_for_timeout(4000)

            print(f"✨ Sesión de Facebook finalizada. Total: {len(vistos)}")
        await browser.close()

async def scrap_x_playwright(tema):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        
        if await inyectar_cookies_pkl(context, "x_cookies.pkl"):
            page = await context.new_page()
            url_busqueda = f"https://x.com/search?q={tema.replace(' ', '%20')}&f=live"
            print(f"🚀 [X] Iniciando recolección exhaustiva: {tema}")
            await page.goto(url_busqueda)
            await page.wait_for_timeout(6000)

            vistos = set()
            tweets_procesados = set()

            with open('comentarios_x.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                for scroll_principal in range(25): # Más profundidad de búsqueda
                    tweets = page.locator('article[data-testid="tweet"]')
                    cantidad = await tweets.count()
                    
                    for i in range(cantidad):
                        try:
                            tweet_actual = tweets.nth(i)
                            texto_ref = await tweet_actual.inner_text()
                            tweet_id = hash(texto_ref[:80])
                            
                            if tweet_id in tweets_procesados: continue

                            # --- ENTRAR AL HILO ---
                            await tweet_actual.scroll_into_view_if_needed()
                            await page.wait_for_timeout(1000)
                            await tweet_actual.click()
                            print(f"   👉 Entrando al hilo {i+1}...")
                            await page.wait_for_timeout(4000)

                            # --- SCROLL INTERNO HASTA EL FINAL ---
                            intentos_sin_nuevos = 0
                            while intentos_sin_nuevos < 3: # Si en 3 scrolls no hay nada nuevo, fin.
                                antes_de_scroll = len(vistos)
                                
                                # Capturar comentarios visibles
                                elementos = await page.locator('div[data-testid="tweetText"]').all()
                                for el in elementos:
                                    t = (await el.inner_text()).strip().replace('\n', ' ')
                                    if len(t) > 12 and t not in vistos:
                                        vistos.add(t)
                                        writer.writerow([t])
                                
                                if len(vistos) > antes_de_scroll:
                                    intentos_sin_nuevos = 0 # Reset si encontramos datos
                                    f.flush()
                                else:
                                    intentos_sin_nuevos += 1
                                
                                # Hacer scroll para cargar más
                                await page.mouse.wheel(0, 1000)
                                await page.wait_for_timeout(1500)

                            print(f"      ✅ Hilo completado. Total acumulado: {len(vistos)}")

                            # --- RETORNO BLINDADO (Escape sucesivo para no perder el feed) ---
                            # Primero intentamos con los botones que me pasaste
                            salida_ok = False
                            for selector in ['button[data-testid="app-bar-back"]', 'button[aria-label="Close"]']:
                                btn = page.locator(selector)
                                if await btn.count() > 0 and await btn.is_visible():
                                    await btn.first.click()
                                    salida_ok = True
                                    break
                            
                            if not salida_ok:
                                # Si X se pone difícil, Escape nos devuelve al feed principal sin recargar
                                await page.keyboard.press("Escape")
                                await page.wait_for_timeout(500)
                                await page.keyboard.press("Escape")

                            await page.wait_for_timeout(2000)
                            tweets_procesados.add(tweet_id)

                        except Exception:
                            # Emergencia: Escape y seguir con el siguiente
                            await page.keyboard.press("Escape")
                            continue

                    # Bajar el feed principal para cargar más tweets
                    await page.evaluate("window.scrollBy(0, 2000)")
                    await page.wait_for_timeout(4000)

            print(f"✨ Proceso de X terminado. Dataset listo con {len(vistos)} entradas.")
        await browser.close()

async def scrap_instagram_playwright(tema):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        if await inyectar_cookies_pkl(context, "instagram_cookies.pkl"):
            page = await context.new_page()
            await page.goto(f"https://www.instagram.com/explore/tags/{tema.replace(' ', '')}/")
            await page.wait_for_timeout(5000)
            vistos = set()
            with open('comentarios_ig.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["texto"])
                try:
                    await page.click("div._aagw") # Abrir primer post
                    for _ in range(30):
                        await page.wait_for_selector("div._a9zr", timeout=5000)
                        elementos = await page.query_selector_all("div._a9zr span._ap3a")
                        for el in elementos:
                            t = (await el.inner_text()).strip().replace('\n', ' ')
                            if len(t) > 10 and t not in vistos:
                                vistos.add(t); writer.writerow([t])
                        await page.keyboard.press("ArrowRight")
                        await page.wait_for_timeout(random.randint(1500, 2500))
                except: pass
        await browser.close()

# --- FASE 3: PROCESAMIENTO PLN Y REPORTES ---

def limpiar_profundo(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    texto = re.sub(r'[^a-z\s]', '', texto)
    return " ".join(texto.split())

def procesar_nlp(texto):
    tokens = nltk.word_tokenize(texto)
    stop = set(stopwords.words('spanish'))
    tokens = [w for w in tokens if w not in stop and len(w) > 2]
    stemmer = SnowballStemmer('spanish')
    return [stemmer.stem(w) for w in tokens]


def realizar_analisis_investigados(df, todos_los_tokens):
    print("\n--- PUNTO B: ANÁLISIS EXTRAS ---")
    todos_emojis = [c for comentario in df['texto'] for c in str(comentario) if emoji.is_emoji(c)]
    if todos_emojis:
        print(f"1. Emojis Top: {Counter(todos_emojis).most_common(5)}")
    bigramas = list(ngrams(todos_los_tokens, 2))
    print(f"2. Bigramas Top: {Counter(bigramas).most_common(5)}")
    riqueza = (len(set(todos_los_tokens)) / len(todos_los_tokens)) * 100
    print(f"3. Riqueza Lexical: {riqueza:.2f}%")

# --- MAIN ---

def run_async(func, tema):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(func(tema))
    loop.close()


def generar_wordcloud(tokens_totales):
    texto_nube = " ".join(tokens_totales)
    wordcloud = WordCloud(width=1000, height=500, 
                          background_color='white',
                          colormap='Dark2',
                          max_words=150).generate(texto_nube)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Bolsa de Palabras Canónicas (Resultados de la Investigación)")
    plt.savefig("nube_palabras_final.png")
    # plt.show() # Deshabilitado para automatización


# =================================================================================================
# SECCIÓN 3: ANÁLISIS AVANZADO CON LLMs (Gemini, Groq, Cloudflare)
# =================================================================================================

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
        response = model.generate_content(f"""Analiza este comentario: "{text}" """)
        result = response.text.replace('```json', '').replace('```', '').strip()
        end_time = time.time()
        return result, end_time - start_time
    except Exception as e:
        time.sleep(1) 
        return f"ERROR: {str(e)}", time.time() - start_time

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
                {"role": "user", "content": f"""Analiza el sentimiento de este post: "{text}".
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

def robust_json_parse(text):
    text = text.strip()
    text = text.replace("“", '"').replace("”", '"') 
    
    try:
        return json.loads(text)
    except:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        clean_text = match.group() if match else text
        return ast.literal_eval(clean_text)
    except:
        pass
    return None

def process_nlp_row(row):
    origin = str(row['origen']).lower().strip()
    text = str(row['texto'])
    
    if len(text) < 5: return None 

    result_raw = "{}"
    duration = 0
    model_used = "Unknown"

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

def run_advanced_nlp():
    print("\n🚀 INICIANDO ANÁLISIS NLP PARALELO (MODO ULTRA-RÁPIDO)")
    print("---------------------------------------------")
    
    global_file = 'resultados_finales_grado.csv'
    if not os.path.exists(global_file):
        print(f"❌ Error crítico: No se encontró {global_file} generado por el scraping.")
        return

    df = pd.read_csv(global_file)
    print(f"✅ Datos cargados del scraping: {len(df)} registros.")

    start_global = time.time()
    results = []
    
    print("⏳ Procesando comentarios en PARALELO con LLMs...")
    MAX_WORKERS = 10 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        rows = df.to_dict('records')
        future_to_row = {executor.submit(process_nlp_row, row): row for row in rows}
        
        completed = 0
        total = len(rows)
        for future in concurrent.futures.as_completed(future_to_row):
            data = future.result()
            if data: results.append(data)
            completed += 1
            if completed % 10 == 0:
                print(f"   ... {completed}/{total} procesados.")

    end_global = time.time()
    total_time = end_global - start_global

    # Guardar resultados finales de NLP
    results_df = pd.DataFrame(results)
    output_file = 'resultados_nlp_benchmark.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("📊 REPORTE DE RESULTADOS IA")
    print("="*50)
    print(f"Archivo generado: {output_file}")
    print(f"Tiempo Total: {total_time:.2f} s")
    print(f"Velocidad: {len(results_df)/total_time:.2f} regs/seg")
    
    if not results_df.empty:
        print("\nDISTRIBUCIÓN DE SENTIMIENTOS:")
        print(results_df['sentimiento'].value_counts())
        print("\nDETALLE POR MODELO:")
        print(results_df.groupby(['modelo', 'sentimiento']).size())


# =================================================================================================
# MAIN FLOW
# =================================================================================================

if __name__ == "__main__":
    tema = 'Daniel Noboa'

    
    # 1. EJECUCIÓN PARALELA DE SCRAPERS
    hilos = [
        # threading.Thread(target=run_async, args=(scrap_linkedin_playwright, tema)),
        threading.Thread(target=run_async, args=(scrap_x_playwright, tema)),
        # threading.Thread(target=run_async, args=(scrap_facebook_playwright, tema)),
        # threading.Thread(target=run_async, args=(scrap_instagram_playwright, tema))
    ]

    print(f"🚀 Iniciando recolección paralela masiva para: {tema}")
    for h in hilos: h.start()
    for h in hilos: h.join()

    # 2. CARGA Y UNIFICACIÓN DE DATOS (Dataset para IA)
    archivos = {
        'comentarios_linkedin.csv': 'linkedin',
        'comentarios_x.csv': 'x',
        'comentarios_fb.csv': 'facebook',
        'comentarios_ig.csv': 'instagram'
    }
    
    dfs = []
    for archivo, red in archivos.items():
        if os.path.exists(archivo):
            temp_df = pd.read_csv(archivo)
            temp_df['origen'] = red
            dfs.append(temp_df)

    if not dfs:
        print("❌ No se recolectaron datos. Revisa las cookies o los selectores.")
    else:
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['texto']) # Limpieza de duplicados
        
        # --- PROCESAMIENTO PLN ---
        df['texto_limpio'] = df['texto'].apply(limpiar_profundo)
        df['tokens'] = df['texto_limpio'].apply(procesar_nlp)
        todos_los_tokens = [t for sub in df['tokens'] for t in sub]

        # --- REPORTE DE MÉTRICAS (LO QUE NECESITABAS) ---
        print("\n" + "="*50)
        print("📊 RESUMEN ANALÍTICO DEL DATASET")
        print("="*50)

        # 1. Volumen por Red
        print(f"1. VOLUMEN TOTAL ÚNICO: {len(df)} comentarios.")

        # 2. Análisis de Bigramas (Punto B de tu tesis)
        print("\n2. TOP 5 BIGRAMAS (Tendencias detectadas):")
        bigramas = list(ngrams(todos_los_tokens, 2))
        top_bigramas = Counter(bigramas).most_common(5)
        for b, freq in top_bigramas:
            print(f"   - {b[0]} {b[1]}: {freq} menciones")

        # 3. Longitud Promedio (Análisis de Engagement)
        print("\n3. LONGITUD PROMEDIO POR PLATAFORMA (Palabras):")
        df['longitud'] = df['texto'].apply(lambda x: len(str(x).split()))
        promedios = df.groupby('origen')['longitud'].mean()
        for red, valor in promedios.items():
            print(f"   - {red.capitalize()}: {valor:.2f} palabras por post")

        # 4. Riqueza Lexical (Diversidad del vocabulario)
        palabras_unicas = len(set(todos_los_tokens))
        total_palabras = len(todos_los_tokens)
        riqueza = (palabras_unicas / total_palabras) * 100 if total_palabras > 0 else 0
        print(f"\n4. RIQUEZA LEXICAL TOTAL: {riqueza:.2f}%")
        
        # 5. Exportación Final
        WordCloud(width=1000, height=500, background_color='white').generate(" ".join(todos_los_tokens)).to_file("nube_final.png")
        df.to_csv('resultados_finales_grado.csv', index=False)

        # run_advanced_nlp()
        
        print("="*50)
        print("✨ PROCESO FINALIZADO: Dataset guardado y Nube generada.")