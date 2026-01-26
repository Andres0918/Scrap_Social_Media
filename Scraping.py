import pickle
import time
import threading
from selenium import webdriver
import pickle
import time
import random
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import os

import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import emoji
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams

try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Error al descargar recursos de NLTK: {e}")


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

# COOKIES (GALLENITAS)

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
            driver.quit() # Cerramos esta ventana antes de abrir la siguiente


# LOGICA DEL SCRAP

def scrap_linkedin(tema):
    driver = crear_driver()
    
    # --- 1. LÓGICA DE LOGIN / COOKIES ---
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

    # --- 2. VALIDACIÓN DE SESIÓN ---
    if not cargar_cookies():
        print("[LinkedIn] Cookies fallidas. Esperando inicio de sesión MANUAL...")
        while not esta_logueado():
            time.sleep(5)
        print("[LinkedIn] Sesión manual detectada. Guardando nuevas cookies...")
        with open("linkedin_cookies.pkl", "wb") as file:
            pickle.dump(driver.get_cookies(), file)

    # --- 3. SCRAPING ---
    print(f"[LinkedIn] Iniciando búsqueda: {tema}")
    search_url = f"https://www.linkedin.com/search/results/content/?keywords={tema}"
    driver.get(search_url)
    time.sleep(8)

    with open('comentarios_linkedin.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Comentario"])

        # Localizar posts usando FeedUpdateServedEvent (según el DOM)
        posts = driver.find_elements(By.XPATH, "//div[contains(@data-view-tracking-scope, 'FeedUpdateServedEvent')]")
        print(f"[LinkedIn] {len(posts)} publicaciones encontradas.")

        conteo_total = 0
        for i, post in enumerate(posts[:10]):
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", post)
                time.sleep(2)

                # Clic en el contador para abrir comentarios
                try:
                    boton_conteo = post.find_element(By.CSS_SELECTOR, "div[data-view-name='feed-comment-count']")
                    driver.execute_script("arguments[0].click();", boton_conteo)
                    time.sleep(4)
                except: continue

                # Bucle Cargar más comentarios
                while True:
                    try:
                        boton_mas = post.find_element(By.CSS_SELECTOR, "button[data-view-name='more-comments']")
                        driver.execute_script("arguments[0].click();", boton_mas)
                        time.sleep(random.uniform(2, 4))
                    except: break

                # Extraer textos
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
    
    # --- Lógica de Cookies ---
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

    # --- Lógica de Scraping ---
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

    # Validación de Sesión
    if not cargar_cookies():
        print("[Facebook] Esperando inicio de sesión MANUAL...")
        while not esta_logueado():
            time.sleep(5)
        with open("facebook_cookies.pkl", "wb") as file:
            pickle.dump(driver.get_cookies(), file)

    # Lógica de Scraping 
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

                # 1. Abrir Modal
                try:
                    boton_abrir = post.find_element(By.XPATH, ".//div[@role='button' and contains(., 'comentario')]")
                    driver.execute_script("arguments[0].click();", boton_abrir)
                    time.sleep(5)
                except: continue

                # 2. Cambiar a 'Todos los comentarios'
                try:
                    filtro_actual = driver.find_element(By.XPATH, "//div[@role='button']//span[contains(text(), 'relevantes') or contains(text(), 'Relevant')]")
                    driver.execute_script("arguments[0].click();", filtro_actual)
                    time.sleep(2)
                    opcion_todos = driver.find_element(By.XPATH, "//div[@role='menuitem']//span[contains(text(), 'Todos los comentarios') or contains(text(), 'All comments')]")
                    driver.execute_script("arguments[0].click();", opcion_todos)
                    time.sleep(5)
                except: pass

                # 3. Scroll dentro del modal
                for _ in range(10):
                    webdriver.ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
                    time.sleep(1.2)

                # 4. Extracción
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

    # --- LÓGICA DE ACCESO ---
    if not cargar_cookies_ig():
        print("\nSISTEMA BLOQUEADO: Entra MANUALMENTE a Instagram.")
        while not esta_logueado():
            time.sleep(5)
        with open("instagram_cookies.pkl", "wb") as file:
            pickle.dump(driver.get_cookies(), file)
        print("✅ Sesión manual guardada.")

    # --- NAVEGACIÓN ---
    hashtag = tema.replace(' ', '')
    print(f"[Instagram] Buscando: #{hashtag}...")
    driver.get(f"https://www.instagram.com/explore/tags/{hashtag}/")
    time.sleep(10)

    # --- EXTRACCIÓN ---
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
                
                # Selector de comentarios que confirmamos (_a9zr)
                comentarios_html = driver.find_elements(By.CSS_SELECTOR, "div._a9zr span._ap3a._aaco._aacu._aacx._aad7._aade")
                
                conteo_post = 0
                for c in comentarios_html:
                    texto = c.text.strip()
                    if len(texto) > 10 and not texto.startswith('@'):
                        writer.writerow([texto.replace('\n', ' ')])
                        conteo_post += 1
                        conteo_total += 1
                
                print(f"       [OK] {conteo_post} comentarios capturados.")

                # 3. CLIC EN EL BOTÓN SIGUIENTE (Basado en el HTML que me pasaste)
                try:
                    # Buscamos el SVG de "Siguiente" y subimos hasta el botón ancestro
                    boton_siguiente = driver.find_element(By.XPATH, "//*[local-name()='svg' and @aria-label='Siguiente']/ancestor::button")
                    driver.execute_script("arguments[0].click();", boton_siguiente)
                    time.sleep(random.uniform(4, 6))
                except:
                    # Intento alternativo por si el modal cambia ligeramente
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

    ##### '''''''''''''''PREPROCESAMIENTO'''''''''''''''''

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

# --- PASO 1: LIMPIEZA Y NORMALIZACIÓN (PLN) ---
def limpiar_profundo(texto):
    if not isinstance(texto, str): return ""
    # 1. Minúsculas
    texto = texto.lower()
    # 2. Quitar URLs y Enlaces
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    # 3. Quitar menciones (@) y hashtags (#)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    # 4. Quitar acentos/emojis para normalizar (para el texto limpio de palabras)
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # 5. Quitar signos, símbolos y números (solo letras)
    texto = re.sub(r'[^a-z\s]', '', texto)
    # 6. Quitar espacios extra
    texto = " ".join(texto.split())
    return texto

# --- PASO 2, 3 Y 4: TOKENIZACIÓN, STOPWORDS Y STEMMING ---
def procesar_nlp(texto_limpio):
    tokens = nltk.word_tokenize(texto_limpio)
    stop_words = set(stopwords.words('spanish'))
    # Filtro de stopwords y longitud mínima
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    # Stemming
    stemmer = SnowballStemmer('spanish')
    return [stemmer.stem(w) for w in tokens]

# --- PASO B: ANÁLISIS EXTRAS INVESTIGADOS ---
def realizar_analisis_investigados(df, todos_los_tokens):
    print("\n" + "="*50)
    print("PUNTO B: ANÁLISIS INVESTIGADOS POR EL GRUPO")
    print("="*50)

    # 1. Conteo de Emojis (Antes de la limpieza profunda)
    print("\n1. FRECUENCIA DE EMOJIS (Sentimiento Visual):")
    todos_emojis = [c for comentario in df['texto'] for c in str(comentario) if emoji.is_emoji(c)]
    if todos_emojis:
        top_emojis = Counter(todos_emojis).most_common(5)
        for e, freq in top_emojis:
            nombre = emoji.demojize(e).replace(':', '').replace('_', ' ')
            print(f"   {e} ({nombre.capitalize()}): {freq} veces")
    else:
        print("   No se detectaron emojis.")

    # 2. Análisis de Bigramas (Conceptos compuestos)
    print("\n2. TOP 5 CONCEPTOS COMPUESTOS (BIGRAMAS):")
    bigramas = list(ngrams(todos_los_tokens, 2))
    top_bigramas = Counter(bigramas).most_common(5)
    for b, freq in top_bigramas:
        print(f"   - {b[0]} {b[1]}: {freq} menciones")

    # 3. Longitud Promedio por Red Social
    print("\n3. LONGITUD PROMEDIO POR PLATAFORMA (Palabras):")
    df['longitud'] = df['texto'].apply(lambda x: len(str(x).split()))
    promedios = df.groupby('origen')['longitud'].mean()
    for red, valor in promedios.items():
        print(f"   - {red.capitalize()}: {valor:.2f} palabras por post")

    # 4. Riqueza Lexical (Diversidad de vocabulario)
    palabras_unicas = len(set(todos_los_tokens))
    total_palabras = len(todos_los_tokens)
    riqueza = (palabras_unicas / total_palabras) * 100 if total_palabras > 0 else 0
    print(f"\n4. RIQUEZA LEXICAL TOTAL: {riqueza:.2f}%")
    print("   (Mide qué tan variado es el vocabulario del debate)")
    print("="*50)

# --- PASO A: VISUALIZACIÓN ---
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
    plt.show()

if __name__ == "__main__":

    if not os.path.exists("linkedin_cookies.pkl" or "x_cookies.pkl" or "facebook_cookies.pkl" or "instagram_cookies.pkl"):
        generar_todas_las_cookies()

    tema_investigacion = 'Rafael Correa'

    # 2. Configuración de Hilos
    hilos = [
        threading.Thread(target=scrap_linkedin, args=(tema_investigacion,)),
        threading.Thread(target=scrap_x, args=(tema_investigacion,)),
        threading.Thread(target=scrap_facebook, args=(tema_investigacion,)),
        threading.Thread(target=scrap_instagram, args=(tema_investigacion,))
    ]

    print(f"🚀 Iniciando recolección paralela sobre: {tema_investigacion}")
    for h in hilos: h.start()
    
    print("⏳ Esperando a que los hilos terminen (esto puede tardar según el volumen de datos)...")
    for h in hilos: h.join()

    print("\n--- RECOLECCIÓN FINALIZADA ---")
    print("Iniciando Fase de Procesamiento de Lenguaje Natural (PLN)...")

    # 3. Carga y Unificación de datos recolectados
    df = cargar_y_unificar_datos()

    if df is not None and not df.empty:
        # 4. Ejecución del Pipeline de PLN (Puntos 1 al 4)
        print("🧼 Limpiando y Normalizando texto...")
        df['texto_limpio'] = df['texto'].apply(limpiar_profundo)
        
        print("✂️ Tokenizando y aplicando Stemming...")
        df['tokens'] = df['texto_limpio'].apply(procesar_nlp)
        
        # Aplanamos los tokens para análisis globales
        todos_los_tokens = [t for sublista in df['tokens'] for t in sublista]
        
        if todos_los_tokens:
            # 5. Punto A: Visualización (Nube de palabras)
            print("☁️ Generando Nube de Palabras...")
            generar_wordcloud(todos_los_tokens)
            
            # 6. Punto B: Otros análisis investigados (Emojis, Bigramas, etc.)
            realizar_analisis_investigados(df, todos_los_tokens)
            
            # 7. Guardar resultados finales
            df.to_csv('resultados_finales_grado.csv', index=False)
            print("\n✨ PROYECTO COMPLETADO EXITOSAMENTE ✨")
            print("Archivos generados: 'nube_palabras_final.png' y 'resultados_finales_grado.csv'")
        else:
            print("❌ El procesamiento no generó tokens. Revisa el contenido de los CSV.")
    else:
        print("❌ No se encontraron datos en los archivos CSV recolectados.")


