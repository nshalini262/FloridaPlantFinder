# FloridaPlantFinder
## Objective: This projet is a RAG Agent that can be used to ask specific questions about Florida plants.

## Tools: 
- Phidata: Agentic Framework used to build the agent
- Groq: Used for API key that provides LLM access
- Qdrant: Used to store data as a vector database

## Sources: 
- iNaturalist API for Florida plant data

## Workflow:
1. Pull relevant Florida plant data from iNaturalist using API endpoint.
2. Convert data into a pdf
3. Embed data and upload to vector database
4. Allow the user to query data, embed the queries and search against database
5. Use LLM to frame responses and return back to user

## Example use cases:
<img width="1016" height="996" alt="image" src="https://github.com/user-attachments/assets/8a71bef6-a88c-4bfa-9dcb-a452ee578c7e" />

## Future Scope:
- Expanding to a larger dataset
- Building a user interface for better interaction
