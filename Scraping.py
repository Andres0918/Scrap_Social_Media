import pickle
import time
import threading
import random
import csv
import os
import re
import json
import unicodedata
import concurrent.futures
import ast

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
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams

# Cargar variables de entorno
load_dotenv()

# Descarga de recursos NLTK
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"⚠️ Nota: Recursos NLTK ya descargados o error menor: {e}")


# =================================================================================================
# SECCIÓN 1: CONFIGURACIÓN Y UTILIDADES DE SCRAPING (SELENIUM)
# =================================================================================================

def crear_driver():
    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(options=options)
    # User agent para evitar bloqueos básicos
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return driver

def crear_driver_basico():
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    return driver

def generar_todas_las_cookies():
    plataformas = {
        "LinkedIn": "https://www.linkedin.com/login",
        "X": "https://x.com/login",
        "Facebook": "https://www.facebook.com",
        "Instagram": "https://www.instagram.com/accounts/login/"
    }

    for nombre, url in plataformas.items():
        print(f"\n--- GENERANDO COOKIES PARA: {nombre} ---")
        driver = crear_driver_basico()
        try:
            driver.get(url)
            print(f"Por favor, inicia sesión en {nombre}. Tienes 60 segundos...")
            time.sleep(60) # Tiempo para loguearte
            
            nombre_archivo = f"{nombre.lower()}_cookies.pkl"
            with open(nombre_archivo, "wb") as file:
                pickle.dump(driver.get_cookies(), file)
            
            print(f"✅ Cookies de {nombre} guardadas en {nombre_archivo}")
        except Exception as e:
            print(f"❌ Error con {nombre}: {e}")
        finally:
            driver.quit() 

# -------------------------------------------------------------------------------------------------
# LOGICA DE EXTRACTORES (POSTS)
# -------------------------------------------------------------------------------------------------

def scrap_linkedin(tema):
    driver = crear_driver()
    
    def esta_logueado():
        try:
            driver.find_element(By.CLASS_NAME, "global-nav__content")
            return True
        except: return False

    def cargar_cookies():
        print("[LinkedIn] Cargando cookies...")
        driver.get("https://www.linkedin.com")
        time.sleep(3)
        try:
            with open("linkedin_cookies.pkl", "rb") as file:
                cookies = pickle.load(file)
                for cookie in cookies:
                    if 'expiry' in cookie: del cookie['expiry']
                    driver.add_cookie(cookie)
            driver.refresh()
            time.sleep(5)
            return esta_logueado()
        except: return False

    if not cargar_cookies():
        print("[LinkedIn] Cookies fallidas. Esperando inicio de sesión MANUAL...")
        while not esta_logueado():
            time.sleep(5)
        print("[LinkedIn] Sesión manual detectada. Guardando nuevas cookies...")
        with open("linkedin_cookies.pkl", "wb") as file:
            pickle.dump(driver.get_cookies(), file)

    print(f"[LinkedIn] Iniciando búsqueda: {tema}")
    search_url = f"https://www.linkedin.com/search/results/content/?keywords={tema}"
    driver.get(search_url)
    time.sleep(8)

    with open('comentarios_linkedin.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comentario"])

        posts = driver.find_elements(By.XPATH, "//div[contains(@data-view-tracking-scope, 'FeedUpdateServedEvent')]")
        print(f"[LinkedIn] {len(posts)} publicaciones encontradas.")

        conteo_total = 0
        for i, post in enumerate(posts[:10]):
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", post)
                time.sleep(2)
                try:
                    boton_conteo = post.find_element(By.CSS_SELECTOR, "div[data-view-name='feed-comment-count']")
                    driver.execute_script("arguments[0].click();", boton_conteo)
                    time.sleep(4)
                except: continue

                while True:
                    try:
                        boton_mas = post.find_element(By.CSS_SELECTOR, "button[data-view-name='more-comments']")
                        driver.execute_script("arguments[0].click();", boton_mas)
                        time.sleep(random.uniform(2, 4))
                    except: break

                comentarios = post.find_elements(By.CSS_SELECTOR, "span[data-testid='expandable-text-box']")
                for c in comentarios:
                    texto = c.text.strip()
                    if len(texto) > 5:
                        writer.writerow([texto.replace('\n', ' ')])
                        conteo_total += 1
                print(f"[LinkedIn] Post {i+1} procesado.")
            except Exception as e:
                print(f"[LinkedIn] Error en post {i+1}: {e}")
                continue

    print(f"✅ [LinkedIn] Finalizado. Total: {conteo_total} comentarios.")
    driver.quit()


def scrap_x(tema):
    driver = crear_driver()
    print("[X] Cargando cookies...")
    driver.get("https://x.com")
    time.sleep(4)
    try:
        with open("x_cookies.pkl", "rb") as file:
            cookies = pickle.load(file)
            for cookie in cookies:
                if 'expiry' in cookie: del cookie['expiry']
                driver.add_cookie(cookie)
        driver.refresh()
        time.sleep(6)
    except FileNotFoundError:
        print("❌ [X] No se encontró x_cookies.pkl. Inicia sesión manualmente.")

    print(f"[X] Buscando: {tema}")
    search_url = f"https://x.com/search?q={tema}&src=typed_query"
    driver.get(search_url)
    time.sleep(7)

    with open('comentarios_x.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Post_Texto"])

        for i in range(5):
            print(f"[X] Scroll {i+1}...")
            tweets = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweetText"]')
            for tweet in tweets:
                try:
                    texto = tweet.text.strip().replace('\n', ' ')
                    if len(texto) > 10:
                        writer.writerow([texto])
                except: continue
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(4, 6))

    print("✅ [X] Finalizado.")
    driver.quit()


def scrap_facebook(tema):
    driver = crear_driver()

    def esta_logueado():
        try:
            driver.find_element(By.XPATH, "//div[@role='navigation'] | //input[@placeholder='Buscar en Facebook']")
            return True
        except: return False

    def cargar_cookies():
        print("[Facebook] Cargando cookies...")
        driver.get("https://www.facebook.com")
        time.sleep(4)
        try:
            with open("facebook_cookies.pkl", "rb") as file:
                cookies = pickle.load(file)
                for cookie in cookies:
                    if 'expiry' in cookie: del cookie['expiry']
                    driver.add_cookie(cookie)
            driver.refresh()
            time.sleep(6)
            return esta_logueado()
        except: return False

    if not cargar_cookies():
        print("[Facebook] Esperando inicio de sesión MANUAL...")
        while not esta_logueado():
            time.sleep(5)
        with open("facebook_cookies.pkl", "wb") as file:
            pickle.dump(driver.get_cookies(), file)

    print(f"[Facebook] Buscando: {tema}")
    driver.get(f"https://www.facebook.com/search/posts/?q={tema}")
    time.sleep(8)

    with open('comentarios_fb.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comentario"])

        for i in range(1, 11):
            try:
                selector_post = f"//div[@aria-posinset='{i}']"
                post = driver.find_element(By.XPATH, selector_post)
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", post)
                time.sleep(2)
                try:
                    boton_abrir = post.find_element(By.XPATH, ".//div[@role='button' and contains(., 'comentario')]")
                    driver.execute_script("arguments[0].click();", boton_abrir)
                    time.sleep(5)
                except: continue

                try:
                    filtro_actual = driver.find_element(By.XPATH, "//div[@role='button']//span[contains(text(), 'relevantes') or contains(text(), 'Relevant')]")
                    driver.execute_script("arguments[0].click();", filtro_actual)
                    time.sleep(2)
                    opcion_todos = driver.find_element(By.XPATH, "//div[@role='menuitem']//span[contains(text(), 'Todos los comentarios') or contains(text(), 'All comments')]")
                    driver.execute_script("arguments[0].click();", opcion_todos)
                    time.sleep(5)
                except: pass

                for _ in range(10):
                    webdriver.ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
                    time.sleep(1.2)

                bloques_texto = driver.find_elements(By.XPATH, "//div[@dir='auto' and @style='text-align: start;']")
                for bloque in bloques_texto:
                    texto = bloque.text.strip().replace('\n', ' ')
                    if len(texto) > 5:
                        writer.writerow([texto])
                
                print(f"[Facebook] Post {i} procesado.")
                webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(2)
            except:
                webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                continue

    print("✅ [Facebook] Finalizado.")
    driver.quit()


def scrap_instagram(tema):
    driver = crear_driver()

    def esta_logueado():
        try:
            driver.find_element(By.XPATH, "//*[@aria-label='Inicio' or @aria-label='Perfil' or @aria-label='Direct']")
            return True
        except: return False

    def cargar_cookies_ig():
        print("[Instagram] Intentando cargar sesión...")
        driver.get("https://www.instagram.com")
        time.sleep(5)
        try:
            with open("instagram_cookies.pkl", "rb") as file:
                cookies = pickle.load(file)
                for cookie in cookies:
                    if 'expiry' in cookie: del cookie['expiry']
                    driver.add_cookie(cookie)
            driver.refresh()
            time.sleep(6)
            return esta_logueado()
        except: return False

    if not cargar_cookies_ig():
        print("\nSISTEMA BLOQUEADO: Entra MANUALMENTE a Instagram.")
        while not esta_logueado():
            time.sleep(5)
        with open("instagram_cookies.pkl", "wb") as file:
            pickle.dump(driver.get_cookies(), file)
        print("✅ Sesión manual guardada.")

    hashtag = tema.replace(' ', '')
    print(f"[Instagram] Buscando: #{hashtag}...")
    driver.get(f"https://www.instagram.com/explore/tags/{hashtag}/")
    time.sleep(10)

    with open('comentarios_ig.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comentario"])

        try:
            primera_publicacion = driver.find_element(By.CLASS_NAME, "_aagw")
            driver.execute_script("arguments[0].click();", primera_publicacion)
            time.sleep(5)
        except Exception as e:
            print(f"❌ Error al abrir cuadrícula: {e}")
            driver.quit()
            return

        conteo_total = 0
        for i in range(15): 
            try:
                print(f"   [+] Extrayendo de publicación {i+1}...")
                comentarios_html = driver.find_elements(By.CSS_SELECTOR, "div._a9zr span._ap3a._aaco._aacu._aacx._aad7._aade")
                conteo_post = 0
                for c in comentarios_html:
                    texto = c.text.strip()
                    if len(texto) > 10 and not texto.startswith('@'):
                        writer.writerow([texto.replace('\n', ' ')])
                        conteo_post += 1
                        conteo_total += 1
                
                print(f"       [OK] {conteo_post} comentarios capturados.")
                try:
                    boton_siguiente = driver.find_element(By.XPATH, "//*[local-name()='svg' and @aria-label='Siguiente']/ancestor::button")
                    driver.execute_script("arguments[0].click();", boton_siguiente)
                    time.sleep(random.uniform(4, 6))
                except:
                    try:
                        boton_siguiente = driver.find_element(By.CSS_SELECTOR, "button._abl-")
                        driver.execute_script("arguments[0].click();", boton_siguiente)
                        time.sleep(random.uniform(4, 6))
                    except:
                        print("       [!] Fin de las publicaciones.")
                        break
            except Exception as e:
                print(f"       [!] Error en publicación {i+1}")
                continue

    print(f"\n✅ SCRAPING INSTAGRAM FINALIZADO. Total: {conteo_total}")
    driver.quit()


# =================================================================================================
# SECCIÓN 2: PREPROCESAMIENTO Y ANÁLISIS BÁSICO (WordCloud, Stats)
# =================================================================================================

def cargar_y_unificar_datos():
    archivos = ['comentarios_linkedin.csv', 'comentarios_x.csv', 'comentarios_fb.csv', 'comentarios_ig.csv']
    dfs = []
    for f in archivos:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = ['texto']
            temp_df['origen'] = f.split('_')[1].split('.')[0] # Extrae el nombre de la red social
            dfs.append(temp_df)
            print(f"✅ Cargado: {f}")
        except:
            print(f"⚠️ No se encontró {f}, saltando...")
    return pd.concat(dfs, ignore_index=True)

def limpiar_profundo(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    texto = re.sub(r'[^a-z\s]', '', texto)
    texto = " ".join(texto.split())
    return texto

def procesar_nlp(texto_limpio):
    tokens = nltk.word_tokenize(texto_limpio)
    stop_words = set(stopwords.words('spanish'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    stemmer = SnowballStemmer('spanish')
    return [stemmer.stem(w) for w in tokens]

def realizar_analisis_investigados(df, todos_los_tokens):
    print("\n" + "="*50)
    print("PUNTO B: ANÁLISIS INVESTIGADOS POR EL GRUPO")
    print("="*50)

    print("\n1. FRECUENCIA DE EMOJIS (Sentimiento Visual):")
    todos_emojis = [c for comentario in df['texto'] for c in str(comentario) if emoji.is_emoji(c)]
    if todos_emojis:
        top_emojis = Counter(todos_emojis).most_common(5)
        for e, freq in top_emojis:
            nombre = emoji.demojize(e).replace(':', '').replace('_', ' ')
            print(f"   {e} ({nombre.capitalize()}): {freq} veces")
    else:
        print("   No se detectaron emojis.")

    print("\n2. TOP 5 CONCEPTOS COMPUESTOS (BIGRAMAS):")
    bigramas = list(ngrams(todos_los_tokens, 2))
    top_bigramas = Counter(bigramas).most_common(5)
    for b, freq in top_bigramas:
        print(f"   - {b[0]} {b[1]}: {freq} menciones")

    print("\n3. LONGITUD PROMEDIO POR PLATAFORMA (Palabras):")
    df['longitud'] = df['texto'].apply(lambda x: len(str(x).split()))
    promedios = df.groupby('origen')['longitud'].mean()
    for red, valor in promedios.items():
        print(f"   - {red.capitalize()}: {valor:.2f} palabras por post")

    palabras_unicas = len(set(todos_los_tokens))
    total_palabras = len(todos_los_tokens)
    riqueza = (palabras_unicas / total_palabras) * 100 if total_palabras > 0 else 0
    print(f"\n4. RIQUEZA LEXICAL TOTAL: {riqueza:.2f}%")
    print("="*50)

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

def main():
    if not os.path.exists("linkedin_cookies.pkl") or \
       not os.path.exists("x_cookies.pkl") or \
       not os.path.exists("facebook_cookies.pkl") or \
       not os.path.exists("instagram_cookies.pkl"):
        print("⚠️ Faltan cookies. Iniciando generador...")
        generar_todas_las_cookies()

    tema_investigacion = 'Rafael Correa'

    # 1. SCRAPING
    hilos = [
        threading.Thread(target=scrap_linkedin, args=(tema_investigacion,)),
        threading.Thread(target=scrap_x, args=(tema_investigacion,)),
        threading.Thread(target=scrap_facebook, args=(tema_investigacion,)),
        threading.Thread(target=scrap_instagram, args=(tema_investigacion,))
    ]

    print(f"🚀 Iniciando recolección paralela sobre: {tema_investigacion}")
    for h in hilos: h.start()
    
    print("⏳ Esperando a que los hilos terminen...")
    for h in hilos: h.join()

    print("\n--- RECOLECCIÓN FINALIZADA ---")

    # 2. UNIFICACIÓN Y TOKENIZACIÓN
    print("Iniciando Fase de Procesamiento de Lenguaje Natural (Básica)...")
    df = cargar_y_unificar_datos()

    if df is not None and not df.empty:
        print("🧼 Limpiando y Normalizando texto...")
        df['texto_limpio'] = df['texto'].apply(limpiar_profundo)
        
        print("✂️ Tokenizando y aplicando Stemming...")
        df['tokens'] = df['texto_limpio'].apply(procesar_nlp)
        
        todos_los_tokens = [t for sublista in df['tokens'] for t in sublista]
        
        if todos_los_tokens:
            print("☁️ Generando Nube de Palabras...")
            generar_wordcloud(todos_los_tokens)
            
            realizar_analisis_investigados(df, todos_los_tokens)
            
            df.to_csv('resultados_finales_grado.csv', index=False)
            print("\n✨ CSV INTERMEDIO GENERADO: 'resultados_finales_grado.csv' ✨")
            
            # 3. LLAMADA AL PIPELINE AVANZADO (IA)
            run_advanced_nlp()
            
        else:
            print("❌ El procesamiento no generó tokens.")
    else:
        print("❌ No se encontraron datos en los archivos CSV recolectados.")

if __name__ == "__main__":
    main()
