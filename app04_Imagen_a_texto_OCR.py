from pathlib import Path

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="OCR · Daniel Flórez",
    page_icon="📷",
    layout="centered",
)

# =====================================================================
#  CONTENIDO
# =====================================================================
IMAGEN = "ocr.jpg"  # opcional: si no existe, simplemente no se muestra

IDIOMAS_OCR = {
    "Español + Inglés": "spa+eng",
    "Español": "spa",
    "Inglés": "eng",
}

FILTROS = {
    "Escala de grises": "Convierte a blanco y negro suave. Es la mejor opción para empezar.",
    "Alto contraste": "Resalta el texto frente al fondo. Útil con sombras o poca luz.",
    "Invertir colores": "Para texto claro sobre fondo oscuro.",
    "Sin filtro": "Usa la imagen tal como la captura la cámara.",
}

# =====================================================================
#  ESTILO (paleta fría basada en #3f71a6)
# =====================================================================
PRIMARIO = "#3f71a6"
OSCURO = "#1b2f45"
CLARO = "#eaf2f9"
SUAVE = "#a9c7e3"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    font-size: 1.08rem;
}
.stApp { background: linear-gradient(180deg, #f6fafd 0%, __CLARO__ 100%); }
.block-container { max-width: 860px; padding-top: 3rem; padding-bottom: 4rem; }

/* Hero */
.hero { text-align: center; padding: 1rem 0 .5rem 0; }
.hero .etiqueta {
    color: __PRIMARIO__; letter-spacing: .25em; text-transform: uppercase;
    font-size: .85rem; font-weight: 600;
}
.hero h1 {
    font-family: 'Playfair Display', serif; font-size: 3rem; color: __OSCURO__;
    margin: .3rem 0 .6rem 0; line-height: 1.12;
}
.hero p { color: #4d6479; font-size: 1.2rem; max-width: 620px; margin: 0 auto; }
.linea {
    width: 70px; height: 4px; border-radius: 4px; background: __PRIMARIO__;
    margin: 1.4rem auto;
}
h2, h3 { font-family: 'Playfair Display', serif !important; color: __OSCURO__ !important; }

/* Tarjetas */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid __SUAVE__66 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(63, 113, 166, .08);
    padding: .6rem .8rem;
}

/* Imágenes (sin recortar: en OCR importa ver la foto completa) */
[data-testid="stImage"] img {
    border-radius: 14px;
    box-shadow: 0 8px 22px rgba(27, 47, 69, .15);
}
[data-testid="stImageCaption"] { text-align: center; color: #6b8399; }

/* Cámara y subida de archivos */
[data-testid="stCameraInput"] > div, [data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"] {
    background: __CLARO__;
    border: 2px dashed __SUAVE__;
    border-radius: 16px;
}

/* Botones */
.stButton > button, .stDownloadButton > button {
    background: __PRIMARIO__; color: white; border: 1.5px solid __PRIMARIO__;
    border-radius: 999px; padding: .7rem 1.6rem; font-size: 1.02rem; font-weight: 600;
    transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: __OSCURO__; border-color: __OSCURO__; color: white;
    transform: translateY(-2px); box-shadow: 0 8px 18px rgba(27, 47, 69, .25);
}
[data-testid="stBaseButton-secondary"] { background: white; color: __PRIMARIO__; }

/* Inputs */
.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 12px; border: 1px solid __SUAVE__; font-size: 1.05rem;
}
[data-testid="stAlert"] { border-radius: 14px; }

/* Sidebar */
[data-testid="stSidebar"] { background: __OSCURO__; }
[data-testid="stSidebar"] * { color: #e6f0f9 !important; }

#MainMenu, footer { visibility: hidden; }
</style>
"""

for token, valor in {
    "__PRIMARIO__": PRIMARIO,
    "__OSCURO__": OSCURO,
    "__CLARO__": CLARO,
    "__SUAVE__": SUAVE,
}.items():
    CSS = CSS.replace(token, valor)

st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
#  LÓGICA
# =====================================================================
BASE = Path(__file__).parent


@st.cache_resource
def estado_tesseract():
    """Comprueba que Tesseract esté instalado y qué idiomas tiene."""
    try:
        pytesseract.get_tesseract_version()
        return True, set(pytesseract.get_languages(config=""))
    except Exception:
        return False, set()


def aplicar_filtro(img_bgr, filtro: str):
    if filtro == "Sin filtro":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if filtro == "Escala de grises":
        return gris
    if filtro == "Invertir colores":
        return cv2.bitwise_not(gris)

    # Alto contraste: ampliar si es pequeña, suavizar y binarizar por zonas
    if gris.shape[1] < 1000:
        gris = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )


@st.cache_data(show_spinner=False, max_entries=20)
def reconocer(datos: bytes, filtro: str, idioma: str):
    """Decodifica la imagen, aplica el filtro y extrae el texto con Tesseract."""
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen.")
    procesada = aplicar_filtro(img, filtro)
    texto = pytesseract.image_to_string(procesada, lang=idioma).strip()
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return texto, original, procesada


# =====================================================================
#  PÁGINA
# =====================================================================
with st.sidebar:
    st.subheader("Cómo usarlo")
    st.markdown(
        "1. Elige el idioma del texto y un filtro.\n"
        "2. Toma una foto con la cámara o sube una imagen.\n"
        "3. Revisa el texto, corrígelo si hace falta y descárgalo."
    )
    st.subheader("Para mejores resultados")
    st.markdown(
        "- Buena luz, sin reflejos ni sombras.\n"
        "- Texto recto, ocupando buena parte de la foto.\n"
        "- Letra impresa: la manuscrita se reconoce mal."
    )

st.markdown(
    """
    <div class="hero">
        <div class="etiqueta">Visión por computador · OCR</div>
        <h1>Reconocimiento óptico de caracteres</h1>
        <p>Toma una foto de un texto y conviértelo en texto que puedes copiar y editar.</p>
        <div class="linea"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

ruta_imagen = BASE / IMAGEN
if ruta_imagen.is_file():
    st.image(Image.open(ruta_imagen), use_container_width=True)
    st.write("")

tesseract_ok, idiomas_instalados = estado_tesseract()

if not tesseract_ok:
    st.error(
        "Tesseract (el motor de reconocimiento) no está instalado en el servidor. "
        "Crea un archivo llamado `packages.txt` en la raíz del repositorio con este "
        "contenido y reinicia la app:"
    )
    st.code("tesseract-ocr\ntesseract-ocr-spa", language="text")
    st.stop()

idiomas_disponibles = {
    nombre: codigo
    for nombre, codigo in IDIOMAS_OCR.items()
    if all(parte in idiomas_instalados for parte in codigo.split("+"))
}
if "spa" not in idiomas_instalados:
    st.warning(
        "Falta el paquete de español. Agrega `tesseract-ocr-spa` a `packages.txt` "
        "para reconocer tildes y la ñ."
    )
if not idiomas_disponibles:
    idiomas_disponibles = {"Inglés": "eng"}

# ---- 1. Ajustes ----
with st.container(border=True):
    st.subheader("1 · Ajustes")
    c1, c2 = st.columns(2)
    with c1:
        nombre_idioma = st.selectbox("Idioma del texto", list(idiomas_disponibles))
    with c2:
        filtro = st.selectbox(
            "Filtro de imagen",
            list(FILTROS),
            help="\n\n".join(f"**{n}:** {d}" for n, d in FILTROS.items()),
        )

st.write("")

# ---- 2. Captura ----
with st.container(border=True):
    st.subheader("2 · Imagen")
    origen = st.radio(
        "Origen de la imagen",
        ["📷 Cámara", "🖼️ Subir imagen"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if origen.startswith("📷"):
        st.caption(
            "Permite el acceso a la cámara cuando el navegador lo pida y centra "
            "el texto en el encuadre."
        )
        archivo = st.camera_input("Toma una foto", label_visibility="collapsed")
    else:
        archivo = st.file_uploader(
            "Sube una imagen",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

# ---- 3. Resultado ----
if archivo is not None:
    st.write("")
    datos = archivo.getvalue()
    try:
        with st.spinner("Leyendo el texto..."):
            texto, original, procesada = reconocer(
                datos, filtro, idiomas_disponibles[nombre_idioma]
            )
    except Exception as error:
        st.error(f"No se pudo analizar la imagen. ({error})")
        st.stop()

    with st.container(border=True):
        st.subheader("3 · Resultado")
        i1, i2 = st.columns(2)
        with i1:
            st.image(original, caption="Imagen original", use_container_width=True)
        with i2:
            st.image(procesada, caption="Lo que lee el OCR", use_container_width=True)

        if texto:
            editado = st.text_area(
                "Texto reconocido",
                value=texto,
                height=240,
                label_visibility="collapsed",
            )
            st.caption(
                f"{len(editado.split())} palabras · {len(editado)} caracteres. "
                "Puedes corregir el texto antes de descargarlo."
            )
            st.download_button(
                "Descargar como .txt",
                data=editado.encode("utf-8"),
                file_name="texto_reconocido.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.warning(
                "No se detectó texto. Prueba con más luz, acércate al texto o cambia "
                "el filtro (por ejemplo, «Alto contraste»)."
            )

st.write("")
st.caption("Reconocimiento con Tesseract OCR.")
