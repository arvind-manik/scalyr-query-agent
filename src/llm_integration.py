"""
LLM Integration for Scalyr Query Builder

This module provides integration with LLMs for the Scalyr Query Builder.
"""

import json
from typing import Dict, Any, List, Optional, Union
import os


class LLMClient:
    """
    Client for interacting with LLMs
    
    This class provides methods for generating Scalyr queries using LLMs.
    """
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LLM client
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM service (defaults to environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Check if API key is available
        if not self.api_key and model_name.startswith("gpt"):
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable or pass api_key.")
    
    def generate_scalyr_query(self, 
                             query_text: str, 
                             log_sources: Dict[str, Any],
                             query_templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a Scalyr query using an LLM
        
        Args:
            query_text: Natural language query text
            log_sources: Dictionary of available log sources
            query_templates: List of query templates
            
        Returns:
            Dictionary containing the generated query
        """
        # If using OpenAI models
        if self.model_name.startswith("gpt"):
            return self._generate_with_openai(query_text, log_sources, query_templates)
        
        # Add support for other LLM providers as needed
        
        # Fallback to a simple response
        return {
            "error": "Unsupported model",
            "message": f"The model {self.model_name} is not supported."
        }
    
    def _generate_with_openai(self, 
                             query_text: str, 
                             log_sources: Dict[str, Any],
                             query_templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a Scalyr query using OpenAI models
        
        Args:
            query_text: Natural language query text
            log_sources: Dictionary of available log sources
            query_templates: List of query templates
            
        Returns:
            Dictionary containing the generated query
        """
        try:
            import openai
        except ImportError:
            return {
                "error": "OpenAI package not installed",
                "message": "Please install the OpenAI package: pip install openai"
            }
        
        # Set the API key
        openai.api_key = self.api_key
        
        # Prepare the system prompt
        system_prompt = self._create_system_prompt(log_sources, query_templates)
        
        # Prepare the user prompt
        user_prompt = f"Generate a Scalyr query for the following request: {query_text}"
        
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more deterministic outputs
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                
                # If no JSON block found, try to parse the entire response
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw content
                return {
                    "explanation": "Generated query (could not parse as JSON)",
                    "raw_response": content
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate query with OpenAI API"
            }
    
    def _create_system_prompt(self, 
                             log_sources: Dict[str, Any],
                             query_templates: List[Dict[str, Any]]) -> str:
        """
        Create a system prompt for the LLM
        
        Args:
            log_sources: Dictionary of available log sources
            query_templates: List of query templates
            
        Returns:
            System prompt string
        """
        # Convert log sources to a string representation
        log_sources_str = json.dumps(log_sources, indent=2)
        
        # Convert query templates to a string representation
        templates_str = json.dumps(query_templates, indent=2)
        
        # Create the system prompt
        return f"""
You are a Scalyr query generation assistant. Your task is to convert natural language requests into Scalyr queries.

Scalyr is a log management and observability platform. It supports several query types:
1. Filter queries - Simple expressions to filter logs (e.g., 'service=="auth-service" level=="error"')
2. Power queries - More powerful queries with pipes and operations (e.g., 'dataset="service-logs" | filter service=="auth-service" | count')
3. Numeric queries - For retrieving numeric data
4. Facet queries - For retrieving common values for a field
5. Timeseries queries - For retrieving numeric data over time

Available log sources:
{log_sources_str}

Query templates you can use:
{templates_str}

Generate a JSON response with the following structure:
```json
{{
  "filter_query": "Simple filter query if applicable",
  "power_query": "Power query if applicable",
  "numeric_query": "Numeric query if applicable",
  "facet_query": "Facet query if applicable",
  "timeseries_query": "Timeseries query if applicable",
  "visualization_suggestions": ["Suggestion 1", "Suggestion 2"],
  "explanation": "Explanation of the generated query"
}}
```

Only include the query types that are relevant to the user's request. Provide clear explanations of what the query does.
"""


class AnthropicClient:
    """
    Client for interacting with Anthropic Claude models
    
    This class provides methods for generating Scalyr queries using Anthropic's Claude.
    """
    
    def __init__(self, model_name: str = "claude-2", api_key: Optional[str] = None):
        """
        Initialize the Anthropic client
        
        Args:
            model_name: Name of the Claude model to use
            api_key: API key for Anthropic (defaults to environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Check if API key is available
        if not self.api_key:
            print("Warning: No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key.")
    
    def generate_scalyr_query(self, 
                             query_text: str, 
                             log_sources: Dict[str, Any],
                             query_templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a Scalyr query using Anthropic Claude
        
        Args:
            query_text: Natural language query text
            log_sources: Dictionary of available log sources
            query_templates: List of query templates
            
        Returns:
            Dictionary containing the generated query
        """
        try:
            import anthropic
        except ImportError:
            return {
                "error": "Anthropic package not installed",
                "message": "Please install the Anthropic package: pip install anthropic"
            }
        
        # Create the Anthropic client
        client = anthropic.Client(api_key=self.api_key)
        
        # Prepare the system prompt
        system_prompt = self._create_system_prompt(log_sources, query_templates)
        
        # Prepare the user prompt
        user_prompt = f"Generate a Scalyr query for the following request: {query_text}"
        
        try:
            # Call the Anthropic API
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more deterministic outputs
                max_tokens=1000
            )
            
            # Parse the response
            content = response.content[0].text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in the response
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                
                # If no JSON block found, try to parse the entire response
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw content
                return {
                    "explanation": "Generated query (could not parse as JSON)",
                    "raw_response": content
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate query with Anthropic API"
            }
    
    def _create_system_prompt(self, 
                             log_sources: Dict[str, Any],
                             query_templates: List[Dict[str, Any]]) -> str:
        """
        Create a system prompt for Claude
        
        Args:
            log_sources: Dictionary of available log sources
            query_templates: List of query templates
            
        Returns:
            System prompt string
        """
        # This is similar to the OpenAI system prompt
        return self._create_system_prompt(log_sources, query_templates)


# Factory function to create an LLM client based on the model name
def create_llm_client(model_name: str = "gpt-4", api_key: Optional[str] = None) -> Union[LLMClient, AnthropicClient]:
    """
    Create an LLM client based on the model name
    
    Args:
        model_name: Name of the LLM model to use
        api_key: API key for the LLM service
        
    Returns:
        An LLM client instance
    """
    if model_name.startswith("gpt"):
        return LLMClient(model_name, api_key)
    elif model_name.startswith("claude"):
        return AnthropicClient(model_name, api_key)
    else:
        # Default to OpenAI
        return LLMClient(model_name, api_key)
