# general project instructions
- this is a python project. it uses python 3.12
- the project dependencies and libraries are managed using uv. always use the `uv` command to install, update, or remove dependencies and run the project.
- the project follows a modular structure, with separate directories for different components and features.
- use meaningful names for modules, classes, and functions to improve code readability and maintainability.
- annotate code with comments and docstrings to explain the purpose and functionality of different parts of the codebase.
- use logging instead of print statements for better tracking and debugging of the code execution.

# objective of the project
- use the unstructured open source library to build a document processing pipeline that can ingest, process, and analyze unstructured data from various sources.
- when looking for relevant and up-to-date information, prioritize high-quality, reliable sources to ensure the accuracy and credibility of the content being processed. Start here https://docs.unstructured.io/open-source/introduction/quick-start for questions related to the unstructured library basics.

# partitioning
- for partitioning a file, check this url https://docs.unstructured.io/open-source/core-functionality/partitioning for upto-date information and examples.
- for partitioning strategies check https://docs.unstructured.io/open-source/concepts/partitioning-strategies

# cleaning
- for cleaning a file, check this url https://docs.unstructured.io/open-source/core-functionality/cleaning for upto-date information and examples.

# extracting
- for extracting information from a file, check this url https://docs.unstructured.io/open-source/core-functionality/extracting for upto-date information and examples.
- about documents elements and metadata, check https://docs.unstructured.io/open-source/concepts/document-elements

# staging
- for staging elements for final output, check this url https://docs.unstructured.io/open-source/core-functionality/staging for upto-date information and examples.

# chunking
- for chunking a file, check this url https://docs.unstructured.io/open-source/core-functionality/chunking for upto-date information and examples.
- chunking best practices at https://unstructured.io/blog/chunking-for-rag-best-practices

# embedding
- for embedding elements, check this url https://docs.unstructured.io/open-source/core-functionality/embedding for upto-date information and examples.
- embedding best practices at https://unstructured.io/blog/understanding-embedding-models-make-an-informed-choice-for-your-rag

# extract images and tables from a file
- for extracting images and tables from a file, check this url https://docs.unstructured.io/open-source/how-to/extract-image-block-types for upto-date information and examples.

# get chunked elements
- for getting chunked elements from a file, check this url https://docs.unstructured.io/open-source/how-to/get-chunked-elements for upto-date information and examples.

# google genai agents
- for all code related google genai AI agents check https://github.com/googleapis/python-genai
- to generated structured output using pydantic models see first https://ai.google.dev/gemini-api/docs/structured-output

# gemini embeddings
- for gemini embeddings, check this url https://ai.google.dev/gemini-api/docs/embeddings?hl=en and this https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb for upto-date information and examples.

# vector database with chromadb
- for vector database with chromadb, check this url https://github.com/google-gemini/cookbook/blob/23a8e7d7e88009e316275947db53ddb6630d38dd/examples/chromadb/Vectordb_with_chroma.ipynb for upto-date information and examples.

# google adk agents
- for all code related google adk agents check https://google.github.io/adk-docs/

# RAG pipeline
- the pdf files to be processed will be in the files folder, each file with its own subfolder. Ex: the hydrocortison.pdf is located in the folder ./files/hydrocortisone
- the processed images will be saved in the corresponding subfolder images/ . This folder will also contain the images_llm_dump.json and images_llm_summaries.json.
- the processed tables will be saved in the corresponding subfolder tables/ - This folder will contain the tables_llm_dump.json and tables_llm_summaries.json.
- the normalized json will be saved in the corresponding subfolder normalized/ 


