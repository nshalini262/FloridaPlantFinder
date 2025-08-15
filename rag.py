import os
import requests
from fpdf import FPDF
from dotenv import load_dotenv
from phi.model.groq.groq import Groq
from phi.agent.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.qdrant.qdrant import Qdrant
from phi.embedder.fastembed import FastEmbedEmbedder
from rich.prompt import Prompt
import typer
import numpy as np
from qdrant_client import QdrantClient

# Loading env variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Embedder used to convert data to be compatible w vector db
class PatchedFastEmbedEmbedder(FastEmbedEmbedder):
    def get_embedding(self, *args, **kwargs):
        embedding = super().get_embedding(*args, **kwargs)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], (list, tuple, np.ndarray)):
            embedding = embedding[0]
        return embedding


# Retrieves specified parameters about Florida plant data from iNaturalist API.
def get_florida_plants_full(target_count=20):
    plants = []
    url = "https://api.inaturalist.org/v1/taxa"
    params = {
        "q": "Florida",
        "per_page": target_count,
        "rank": "species",
        "taxon_id": 47126,  # plantae
        "iconic_taxon_id": 47126
    }
    
    response = requests.get(url, params=params, timeout=30)
    data = response.json().get("results", [])
    
    for plant in data[:target_count]:
        # Get common name from various possible fields
        common_name = plant.get("preferred_common_name") or plant.get("name", "N/A")
        if common_name == plant.get("name"):  # Making scientific easily comprehensible
            common_name = plant.get("name", "N/A").replace("_", " ").title()
        
        transformed_plant = {
            "common_name": common_name,
            "scientific_name": [plant.get("name", "N/A")],
            "observations_count": plant.get("observations_count", 0),
            "wikipedia_url": plant.get("wikipedia_url", "N/A"),
            "iconic_taxon_name": plant.get("iconic_taxon_name", "N/A")
        }
        plants.append(transformed_plant)

    print(f"There are {len(plants)} Florida plants saved.")
    return plants[:target_count]


def generate_pdf_full(plants, filename="florida_plants.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)

    for idx, plant in enumerate(plants, start=1):
        common_name = plant.get("common_name", "N/A")
        sci_name = plant.get("scientific_name", ["N/A"])[0]
        observations_count = plant.get("observations_count", 0)
        wikipedia_url = plant.get("wikipedia_url", "N/A")
        iconic_taxon_name = plant.get("iconic_taxon_name", "N/A")

        # Formatting
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 6, f"{idx}. {common_name}")
        pdf.set_font("Arial", "I", 11)
        pdf.multi_cell(0, 6, f"({sci_name})")
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, f"Observations: {observations_count}, Type: {iconic_taxon_name}")
        if wikipedia_url != "N/A":
            pdf.multi_cell(0, 6, f"Wikipedia: {wikipedia_url}")
        pdf.ln(2)

    pdf.output(filename)
    return filename


# Creating vector db inside Qdrant
collection_name = "florida-plants-index"
vector_size = 384
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

'''Code to create collection is commented out after first run.
This is to save time and prevent collection from being rewritten.'''
# from qdrant_client.models import VectorParams, Distance
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
# )


embedder = PatchedFastEmbedEmbedder()
vector_db = Qdrant(
    collection=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    embedder=embedder
)


'''Code is commented out to prevent overwriting of pdf.'''
def generate_pdf():
    # plants = get_florida_plants_full(target_count=20)
    # return generate_pdf_full(plants)
    return "florida_plants.pdf"


# Phidata RAG agent setup
def qdrant_agent(user: str = "user"):
    # Generate PDF
    pdf_file = generate_pdf()
    # Setting up knowledge
    knowledge_base = PDFKnowledgeBase(
        path=os.path.abspath(pdf_file),
        vector_db=vector_db
    )
    knowledge_base.load()
    
    agent = Agent(
        provider=Groq(id="llama3-70b-8192", api_key=GROQ_API_KEY),
        agent_id="plant_assistant",
        session_id=user,
        knowledge_base=knowledge_base,
        add_references=True,
        add_chat_history_to_messages=True,
        references_format="json",
        output_model=None,
        debug_mode=True
    )

    while True:
        message = Prompt.ask(f"[bold green]🍃 {user}[/bold green]")
        if message.lower() in ("exit", "bye", "quit"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(qdrant_agent)
