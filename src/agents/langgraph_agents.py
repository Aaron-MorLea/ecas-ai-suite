from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import yaml

class AgentState(TypedDict):
    """State for LangGraph agents."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    fraud_score: float
    customer_query: str

class ECommerceAgents:
    """Multi-agent system using LangGraph for e-commerce.
    Covers: LangChain, LangGraph, RAG, Knowledge Graph from job requirements."""

    def __init__(self, config_path: str = "configs/agent_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

    def setup_rag_system(self):
        """Setup RAG with vector database and knowledge graph."""
        # Vector DB (Chroma)
        vectorstore = Chroma(
            collection_name=self.config['rag']['vector_db']['collection_name'],
            embedding_function=self.embeddings,
            persist_directory="./data/chroma_db"
        )
        
        # Knowledge Graph (Neo4j) - connection setup
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            self.config['rag']['knowledge_graph']['uri'],
            auth=(self.config['rag']['knowledge_graph']['user'], 
                  self.config['rag']['knowledge_graph']['password'])
        )
        
        return vectorstore, driver

    def create_support_agent(self):
        """Create customer support agent with RAG tools."""
        vectorstore, _ = self.setup_rag_system()
        
        tools = [
            Tool(
                name="rag_retriever",
                func=vectorstore.as_retriever().get_relevant_documents,
                description="Retrieve relevant product and policy documents"
            ),
            Tool(
                name="order_lookup",
                func=self._lookup_order,
                description="Look up order status by order ID"
            ),
            Tool(
                name="refund_processor",
                func=self._process_refund,
                description="Process refund requests"
            )
        ]
        
        agent = create_tool_calling_agent(self.llm, tools, messages_modifier=None)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def create_fraud_agent(self):
        """Create fraud analysis agent."""
        tools = [
            Tool(
                name="transaction_analyzer",
                func=self._analyze_transaction,
                description="Analyze transaction for fraud patterns"
            ),
            Tool(
                name="risk_scorer",
                func=self._calculate_risk_score,
                description="Calculate risk score based on user behavior"
            ),
            Tool(
                name="pattern_detector",
                func=self._detect_patterns,
                description="Detect suspicious patterns across transactions"
            )
        ]
        
        agent = create_tool_calling_agent(self.llm, tools, messages_modifier=None)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)

    def build_langgraph_workflow(self):
        """Build multi-agent workflow with LangGraph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("support", self._support_node)
        workflow.add_node("fraud", self._fraud_node)
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add edges
        workflow.add_edge("support", "supervisor")
        workflow.add_edge("fraud", "supervisor")
        workflow.set_entry_point("support")
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {"fraud": "fraud", "end": END}
        )
        
        return workflow.compile()

    def _support_node(self, state: AgentState):
        agent = self.create_support_agent()
        response = agent.invoke({"input": state["customer_query"]})
        return {"messages": [AIMessage(content=response["output"])]}

    def _fraud_node(self, state: AgentState):
        agent = self.create_fraud_agent()
        response = agent.invoke({"input": state["customer_query"]})
        return {"messages": [AIMessage(content=response["output"])]}

    def _supervisor_node(self, state: AgentState):
        return {"next": "end"}

    def _should_continue(self, state: AgentState):
        if state.get("fraud_score", 0) > 0.7:
            return "fraud"
        return "end"

    def _lookup_order(self, order_id: str):
        return f"Order {order_id} status: Shipped"

    def _process_refund(self, order_id: str):
        return f"Refund processed for order {order_id}"

    def _analyze_transaction(self, transaction_data: str):
        return "Transaction appears legitimate"

    def _calculate_risk_score(self, user_data: str):
        return 0.3

    def _detect_patterns(self, data: str):
        return "No suspicious patterns detected"
