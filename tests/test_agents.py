import pytest
from src.agents.langgraph_agents import ECommerceAgents, AgentState
from src.agents.rag_system import RAGSystem
from unittest.mock import Mock, patch

class TestECommerceAgents:
    """Test LangGraph agents."""
    
    def setup_method(self):
        self.agents = ECommerceAgents(config_path="configs/agent_config.yaml")
    
    def test_agent_initialization(self):
        """Test agents are initialized correctly."""
        assert self.agents.config is not None
        assert 'rag' in self.agents.config
    
    def test_support_agent_creation(self):
        """Test support agent can be created."""
        with patch.object(self.agents, 'create_support_agent') as mock:
            mock.return_value = Mock()
            agent = self.agents.create_support_agent()
            assert agent is not None
    
    def test_fraud_agent_creation(self):
        """Test fraud agent can be created."""
        with patch.object(self.agents, 'create_fraud_agent') as mock:
            mock.return_value = Mock()
            agent = self.agents.create_fraud_agent()
            assert agent is not None
    
    def test_workflow_creation(self):
        """Test LangGraph workflow compilation."""
        with patch.object(self.agents, 'build_langgraph_workflow') as mock:
            mock.return_value = Mock()
            workflow = self.agents.build_langgraph_workflow()
            assert workflow is not None


class TestRAGSystem:
    """Test RAG system with vector DB and knowledge graph."""
    
    def setup_method(self):
        self.rag = RAGSystem(config_path="configs/agent_config.yaml")
    
    def test_vector_db_setup(self):
        """Test vector database initialization."""
        with patch('langchain_community.vectorstores.Chroma') as mock:
            mock.return_value = Mock()
            result = self.rag.setup_vector_database()
            assert result is not None
    
    def test_prompt_library_addition(self):
        """Test adding templates to prompt library."""
        import tempfile
        import os
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            temp_path = f.name
        
        # Mock the library path
        with patch('src.agents.rag_system.os.path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__ = Mock()
                mock_open.return_value.__exit__ = Mock()
                
                self.rag.add_to_prompt_library(
                    "test_template",
                    "This is a test template for {query}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
