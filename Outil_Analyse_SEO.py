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

# Centrer le titre au milieu de la page avec du Markdown
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 10px;
            color: #555;
            font-style: italic;
            padding: 10px 0;

        }
    </style>

    <div class="footer">Made by :  Ghada EL KILENI | Aya MABROUK | Thafath Halouane | Xavier meynard | Lola labory \n -  Copyright © 2024 Outil d'Analyse SEO</div>
    
""", unsafe_allow_html=True)

# Charger le modèle NLP de spaCy
nlp = spacy.load("fr_core_news_sm")

# Fonction pour analyser le sentiment
def analyse_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Entre -1 (négatif) et 1 (positif)
    subjectivity = blob.sentiment.subjectivity  # Entre 0 (objectif) et 1 (subjectif)
    return polarity, subjectivity

# Fonction pour le nettoyage et la normalisation du texte
def clean_text(text):
    text = text.lower()  # Mettre tout en minuscules
    return text

# Fonction pour extraire les mots les plus répétés dans le texte
def extract_keywords(text, max_keywords=10):
    text = clean_text(text)
    doc = nlp(text)

    # Extraction des mots lemmatisés, en excluant les mots vides et les signes de ponctuation
    lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    # Comptage des occurrences des mots
    word_counts = Counter(lemmatized_words)

    # Sélection des mots les plus fréquents
    most_common_keywords = word_counts.most_common(max_keywords)

    # Renvoie les mots les plus fréquents avec leurs occurrences
    return [word for word, _ in most_common_keywords]

# Fonction pour générer un nuage de mots
def generate_wordcloud(keywords):
    text = ' '.join(keywords)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)

# Fonction pour vérifier les liens
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
def enrichir_description(texte):
    doc = nlp(texte)
    description = " ".join([token.text for token in doc])
    return description

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
                text_content = ' '.join([p.get_text() for p in soup.find_all('p')])  # Récupérer tout le texte
                keywords_text = extract_keywords(text_content)  # Extraire les mots les plus fréquents
                generate_wordcloud(keywords_text)  # Afficher le nuage de mots

                # Section 3 : Analyse des Liens
                st.header("3. Analyse des Liens")
                links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
                if st.button("Analyser les Liens"):
                    link_status_df = check_links(links)
                    st.write(link_status_df)

               # Section 3 : Analyse des Images
                st.header("4. Analyse des Images")
                image_data = []
                images = soup.find_all('img')

                # Analyser les images
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

                # Ajouter un bouton pour afficher/masquer les images
                if "show_images" not in st.session_state:
                    st.session_state["show_images"] = False  # Par défaut, les images sont masquées

                # Bouton pour basculer entre affichage et masquage
                if st.button("Afficher/Masquer les Images"):
                    st.session_state["show_images"] = not st.session_state["show_images"]

                # Afficher les images si le bouton est activé
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
                if sentiment_text == "Positif 😊" :
                    st.success(f"Sentiment global : {sentiment_text}")

                elif sentiment_text == "Négatif 😟" :
                     st.error(f"Sentiment global : {sentiment_text}")
                else :
                    st.warning(f"Sentiment global : {sentiment_text}")

                
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
