import streamlit as st
from bs4 import BeautifulSoup
import requests
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from urllib.parse import urljoin
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Mise en cache pour optimiser le chargement
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("fr_core_news_sm")
        return spacy.load("fr_core_news_sm")

# Chargement du modèle NLP
nlp = load_spacy_model()

# Fonction pour analyser le sentiment
@st.cache_data
def analyse_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Fonction pour le nettoyage et la normalisation du texte
@st.cache_data
def clean_text(text):
    return text.lower()

# Fonction pour extraire les mots les plus répétés
@st.cache_data
def extract_keywords(text, max_keywords=10):
    text = clean_text(text)
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    word_counts = Counter(lemmatized_words)
    return [word for word, _ in word_counts.most_common(max_keywords)]

# Fonction pour générer un nuage de mots
def generate_wordcloud(keywords):
    text = ' '.join(keywords)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)

# Fonction pour vérifier les liens
@st.cache_data
def check_links(links):
    link_status = []
    for link in links:
        try:
            response = requests.head(link, timeout=5)
            status = response.status_code
            link_status.append({'Lien': link, 'Statut': '🟢 Fonctionnel' if status < 400 else '🔴 Cassé'})
        except:
            link_status.append({'Lien': link, 'Statut': '🔴 Cassé'})
    return pd.DataFrame(link_status)

# Fonction pour enrichir les descriptions avec spaCy
@st.cache_data
def enrichir_description(texte):
    doc = nlp(texte)
    return " ".join([token.text for token in doc])

# Vérification de la disponibilité de l'API Ollama
def is_ollama_api_available():
    try:
        response = requests.get('http://localhost:11434/health', timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Fonction pour envoyer un prompt à l'API Ollama
def message_llama(prompt, model="llama3.2:1b", max_tokens=100, temperature=0.7, top_p=1, n=1):
    url = 'http://localhost:11434/api/generate'
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n
    }
    try:
        data = json.dumps(payload)
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            responses = response.text.splitlines()
            complete_response = ""
            for line in responses:
                try:
                    response_json = json.loads(line)
                    complete_response += response_json.get("response", "")
                    if response_json.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    st.error(f"Erreur lors du décodage JSON : {e}")
            return complete_response
        else:
            st.error(f"Erreur API Ollama : {response.status_code} - {response.text}")
            return None
    except requests.RequestException as e:
        st.error(f"Erreur de connexion à l'API Ollama : {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue : {e}")
        return None
# Titre de l'application
st.title("Outil d'Analyse SEO")

# Formulaire pour l'entrée de l'URL
url = st.text_input("Entrez l'URL à analyser :", "")

if url:
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Section 1 : Informations Générales
                st.header("1. Informations Générales")
                title_tag = soup.title.string if soup.title else "Aucun titre trouvé"
                st.subheader("Titre de la Page :")
                st.write(title_tag)

                meta_description = soup.find("meta", attrs={"name": "description"})
                meta_description = meta_description["content"] if meta_description else "Aucune métadescription trouvée"
                st.subheader("Description :")
                st.write(meta_description)

                # Section 2 : Nuage de Mots
                st.header("2. Nuage de Mots")
                text_content = ' '.join([p.get_text() for p in soup.find_all('p')])
                keywords_text = extract_keywords(text_content)
                generate_wordcloud(keywords_text)

                # Section 3 : Analyse des Liens
                st.header("3. Analyse des Liens")
                links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
                if st.button("Analyser les Liens"):
                    with st.spinner("Vérification des liens en cours..."):
                        link_status_df = check_links(links)
                        st.write(link_status_df)

                # Section 4 : Analyse des Images
                st.header("4. Analyse des Images")
                image_data = []
                images = soup.find_all('img')

                for img in images:
                    img_src = img.get('src')
                    img_alt = img.get('alt', 'Aucune description')
                    img_alt_enrichi = enrichir_description(img_alt)

                    if img_src:
                        img_src = urljoin(url, img_src)
                        image_data.append({
                            'Icône': f'<a href="{img_src}" target="_blank"><img src="{img_src}" width="50" height="50" /></a>',
                            'Description': img_alt_enrichi,
                        })

                df_images = pd.DataFrame(image_data)

                if "show_images" not in st.session_state:
                    st.session_state["show_images"] = False

                if st.button("Afficher/Masquer les Images"):
                    st.session_state["show_images"] = not st.session_state["show_images"]

                if st.session_state["show_images"]:
                    st.info(" 🔍 Cliquez sur les icônes pour voir les images en taille réelle.")
                    st.write(df_images.to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.info("💬 Les images analysées sont masquées. Cliquez sur le bouton pour les afficher.")

                # Section 5 : Analyse de Sentiment
                st.header("5. Analyse de Sentiment")
                polarity, subjectivity = analyse_sentiment(text_content)
                st.write("### **Polarité (entre -1 et 1) :**")
                st.progress((polarity + 1) / 2)
                st.write("### **Subjectivité (entre 0 et 1) :**")
                st.progress(subjectivity)

                sentiment_text = "Positif 😊" if polarity > 0 else "Négatif 😟" if polarity < 0 else "Neutre ⚖️"
                if sentiment_text == "Positif 😊":
                    st.success(f"Sentiment global : {sentiment_text}")
                elif sentiment_text == "Négatif 😟":
                    st.error(f"Sentiment global : {sentiment_text}")
                else:
                    st.warning(f"Sentiment global : {sentiment_text}")

                # Section 6 : Recommandations SEO
               # Section 6 : Recommandations SEO
                st.header("6. Recommandations SEO 🛠️")
                prompt = f"Analyse SEO du site : {url}\nTitre : {title_tag}\nDescription : {meta_description}"
                if st.button("Générer les recommandations SEO"):
                    seo_recommendations = message_llama(prompt)
                    if seo_recommendations:
                        st.info(seo_recommendations)
                    else:
                        st.error("Erreur lors de la génération des recommandations SEO.")
            else:
                st.error(f"Erreur HTTP : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")