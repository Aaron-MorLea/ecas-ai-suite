from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
import yaml

class RAGSystem:
    """Complete RAG system with vector DB and knowledge graph.
    Covers: RAG, vector databases, knowledge graphs from job requirements."""

    def __init__(self, config_path: str = "configs/agent_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.vectorstore = None
        self.graph = None

    def setup_vector_database(self, documents: list = None):
        """Initialize Chroma vector database with documents."""
        self.vectorstore = Chroma(
            collection_name=self.config['rag']['vector_db']['collection_name'],
            embedding_function=self.embeddings,
            persist_directory="./data/chroma_db"
        )
        
        if documents:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
        
        return self.vectorstore

    def setup_knowledge_graph(self):
        """Initialize Neo4j knowledge graph."""
        kg_config = self.config['rag']['knowledge_graph']
        
        self.graph = Neo4jGraph(
            url=kg_config['uri'],
            username=kg_config['user'],
            password=kg_config['password']
        )
        
        return self.graph

    def create_rag_chain(self):
        """Create RAG chain combining vector search and graph queries."""
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        
        # Vector retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Graph chain
        graph_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True
        )
        
        # Combine prompts
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful e-commerce assistant. Use the context to answer questions."),
            ("human", "{input}"),
            ("human", "Context: {context}")
        ])
        
        # Document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain, graph_chain

    def query(self, question: str):
        """Query the RAG system."""
        retrieval_chain, graph_chain = self.create_rag_chain()
        
        # Get vector results
        vector_result = retrieval_chain.invoke({"input": question})
        
        # Get graph results
        graph_result = graph_chain.invoke({"query": question})
        
        return {
            "vector_answer": vector_result.get("answer", ""),
            "graph_answer": graph_result.get("result", ""),
            "sources": vector_result.get("context", [])
        }

    def add_to_prompt_library(self, template_name: str, template_content: str):
        """Add new prompt template to library.
        Covers: prompt libraries from job requirements."""
        import yaml
        
        library_path = "src/agents/prompts/library.yaml"
        try:
            with open(library_path, 'r') as f:
                library = yaml.safe_load(f) or {}
        except FileNotFoundError:
            library = {}
        
        library[template_name] = template_content
        
        with open(library_path, 'w') as f:
            yaml.dump(library, f)
