import numpy as np
import requests
import json

# Charger les embeddings et les chunks
embeddings = np.load("embeddings.npy")
with open("chunks.json", "r") as f:
    chunks = json.load(f)

API_KEY = ""

def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm_vec1 = np.sqrt(sum(v1**2 for v1 in vec1))
    norm_vec2 = np.sqrt(sum(v2**2 for v2 in vec2))
    return dot_product / (norm_vec1 * norm_vec2)

def generate_openai_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def search_similar_chunks(query_embedding, embeddings, chunks, top_k=5):
    similarities = [cosine_similarity(query_embedding, embedding) for embedding in embeddings]
    ranked_chunks = sorted(zip(similarities, chunks), reverse=True, key=lambda x: x[0])
    return ranked_chunks[:top_k]

def generate_gpt4_mini_response(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Tu es l’assistant AI de la Faculté des sciences de l'administration de l’Université Laval (FSA ULaval). Ta mission est d’assister les étudiants actuels et futurs en répondant à leurs questions concernant le site et les programmes offerts. FSA ULaval fait partie d’un groupe sélect d’écoles accréditées par AACSB International et EQUIS, et est membre signataire de l’initiative PRME du Pacte mondial des Nations Unies, ce qui témoigne de son engagement envers une éducation de gestion responsable et de qualité. La faculté propose un large éventail de programmes grâce à ses 5 départements et son école, tous situés au pavillon Palasis-Prince, à l’adresse suivante : Pavillon Palasis-Prince, 2325, rue de la Terrasse, Université Laval, Québec (Québec), G1V 0A6, Téléphone : 418 656-2180. Quand tu réponds aux questions, tu ne mentionnes jamais que tu manques d’information. Si une question sort du cadre de tes connaissances, tu fournis une réponse logique, orientée vers le service ou l’information pertinente que l’étudiant pourrait consulter, utilise le context fournie pour t'aider, et ne dit parle jamais du contexte"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Fonction pour tester le système
def query_system(question):
    query_embedding = generate_openai_embedding(question)
    similar_chunks = search_similar_chunks(query_embedding, embeddings, chunks)
    
    print(f"Question: {question}")
    print("\nChunks les plus similaires:")
    for similarity, chunk in similar_chunks:
        print(f"Similarité: {similarity:.4f}")
        print(f"Chunk: {chunk[:1000]}...")  # Afficher les 5000 premiers caractères du chunk
        print()
    
    # Utiliser le meilleur chunk comme contexte pour GPT-4-mini
    best_chunk = similar_chunks[0][1]
    prompt = f"Context: {best_chunk}\n\nQuestion: {question}\n\nPlease answer the question based on the given context."
    
    gpt4_mini_response = generate_gpt4_mini_response(prompt)
    
    print("\nRéponse générée par GPT-4-mini:")
    print(gpt4_mini_response)

# Test du système
question = "Quels sont les principaux axes de recherche de la Faculté des sciences de l'administration de l'Université Laval ?"
query_system(question)
