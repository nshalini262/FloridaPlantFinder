# FloridaPlantFinder
## Objective: This project is a Retrieval Augmented Generation (RAG) Agent that can be used to query a dataset of Florida plants for detailed responses.

## Tools: 
- Phidata: An agentic Framework used to build the agent.
- Groq: Provides API key for LLM access.
- Qdrant: Used to store semantic data as a vector database.
- iNaturalist API: Provides details about Florida plants.

## Workflow:
1. Data Collection: Pull relevant Florida plant data from iNaturalist using iNaturalist API.
   - This data is converted into a pdf. 
2. Embedding: The data retrieved is embedded into the Qdrant vector database where it undergoes clustering.
3. Querying: The queries are embedded and compared to the data contained into the vector database to produce a result.
4. Response: The generated response is returned to the user using natural language with the LLM.

## Example use cases:
<img width="1016" height="996" alt="image" src="https://github.com/user-attachments/assets/8a71bef6-a88c-4bfa-9dcb-a452ee578c7e" />

## Future Scope:
- Expanding to a larger dataset.
- Building a user interface for better interaction.
