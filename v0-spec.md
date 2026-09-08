- **Project**: AI tutor chatbot that answers question about AI Engineering and AI System design
- **Users**: aspiring AI Engineers preparing for interviews (like myself)
- **Input**: user query + a corpus of documents and books related to AI Engineering *for the V0*; web fetched content *later on for a V1 or V2*
- **Output**: an answer grounded in the corpus
- **Why it fits**: 
	- the structure and patterns implemented can be adapted for other similar tasks that could be needed by any company that has some sort of data and wants to query it in natural language: RAG pipelines, support systems
	- it answers one of my current needs: I need help preparing for my technical interviews

#### Starter corpus
- All pages of this repo: [ai-engineering-field-guide](https://github.com/alexeygrigorev/ai-engineering-field-guide) + the engineering blog posts already cited throughout the guide (Doctolib, Uber, Airbnb, the Perplexity ByteByteGo piece)
- All LLM/production-tagged artciles listed here: https://eugeneyan.com/start-here
- [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html)

#### Minimal Technical Shape
- Basic chat interface: prompt bar + message history; no chat history/resumable chats...etc
- RAG pipeline:
	- **INGESTION**:
		- Build metadata for each document
		- Chunk each document: recursive chunking with a fixed 400-500 tokens with a 20% overlap
		- Index all chunks: 
			- Embedding model:  BAAI/bge-small-en-v1.5: Self hostable, appropriate 512 token context length, asymmetric. Good for the first V0 iterations
			- Vector DB: Chroma: fastest path to a working V0 since it's serverless and this scale doesn't require anything specific yet
	- **RETRIEVAL**: 
		- Top k=5
		- Context Assembly: [Citation number] {Attached metadata} Chunk: text
		- No reranking or hybrid retrieval
	- **GENERATION**: LLM served via OpenRouter API
- Evaluation and baseline:
	- **Golden Dataset**: initial draft of 15-20 general questions testing general AI Engineering concepts found in the corpus and some document-specific question to test retrieval like "What section of what document speaks of the interview preparation process" as well as not answerable questions from the corpus (3-4)
	- **Evaluation Framework**: I'll use DeepEval from the get go and build on top of it throughout the subsequent versions to catch regressions. I treat LLM-evals like deterministic code tests with Pytest
	- **Baseline**: I'll use the same generative model without additional retrieved context as the baseline to measure if RAG with specialized documents adds value.
	- **Tracked Metrics**: Faithfulness, pairwise comparison with the baseline, hit-rate@k, contextual precision and recall
- Guardrails
	- System prompt instructions: "If the context doesn't contain the answer, say so explicitly"