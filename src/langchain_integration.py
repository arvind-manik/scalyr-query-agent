"""
LangChain Integration for Scalyr Query Generator

This module provides integration with LangChain for the Scalyr Query Generator.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List, Union

# Import LangChain components conditionally to handle missing dependencies gracefully
try:
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from langchain.pydantic_v1 import BaseModel, Field
    from langchain.schema import StrOutputParser
    langchain_available = True
except ImportError:
    langchain_available = False
    print("LangChain not available. Install with: pip install langchain")


# Define Pydantic model for structured output
if langchain_available:
    class ScalyrQueryOutput(BaseModel):
        """Output schema for Scalyr queries"""
        filter_query: Optional[str] = Field(None, description="Simple filter query if applicable")
        power_query: Optional[str] = Field(None, description="Power query with full syntax")
        explanation: str = Field(..., description="Explanation of what the query does")
        visualization_suggestions: List[str] = Field(default_factory=list, description="Suggested visualizations")


class LangChainLLM:
    """
    LangChain-based LLM integration for Scalyr Query Generator
    
    This class provides methods for generating Scalyr queries using LangChain.
    """
    
    def __init__(self, provider: str = "openai", model_name: Optional[str] = None):
        """
        Initialize the LangChain LLM
        
        Args:
            provider: LLM provider to use (openai, anthropic, google)
            model_name: Optional specific model name to use
        """
        self.provider = provider
        self.model_name = model_name
        self.llm = None
        
        if not langchain_available:
            print("LangChain not available. Install with: pip install langchain")
            return
        
        # Initialize the appropriate LLM based on provider
        if provider == "openai":
            self._init_openai(model_name)
        elif provider == "anthropic":
            self._init_anthropic(model_name)
        elif provider == "google":
            self._init_google(model_name)
        else:
            print(f"Unsupported provider: {provider}")
    
    def _init_openai(self, model_name: Optional[str] = None):
        """Initialize OpenAI LLM"""
        try:
            from langchain_openai import ChatOpenAI
            
            # Get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: No OpenAI API key found in environment (OPENAI_API_KEY)")
            
            # Use default model if none specified
            model = model_name or "gpt-4"
            
            # Initialize the LLM
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.2,
                api_key=api_key
            )
        except ImportError:
            print("OpenAI integration not available. Install with: pip install langchain-openai")
    
    def _init_anthropic(self, model_name: Optional[str] = None):
        """Initialize Anthropic LLM"""
        try:
            from langchain_anthropic import ChatAnthropic
            
            # Get API key from environment
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Warning: No Anthropic API key found in environment (ANTHROPIC_API_KEY)")
            
            # Use default model if none specified
            model = model_name or "claude-3-opus-20240229"
            
            # Initialize the LLM
            self.llm = ChatAnthropic(
                model=model,
                temperature=0.2,
                anthropic_api_key=api_key
            )
        except ImportError:
            print("Anthropic integration not available. Install with: pip install langchain-anthropic")
    
    def _init_google(self, model_name: Optional[str] = None):
        """Initialize Google Gemini LLM"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Get API key from environment
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("Warning: No Google API key found in environment (GOOGLE_API_KEY)")
            
            # Use default model if none specified
            model = model_name or "gemini-pro"
            
            # Initialize the LLM
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.2,
                google_api_key=api_key
            )
        except ImportError:
            print("Google Gemini integration not available. Install with: pip install langchain-google-genai")
    
    def generate_query(self, query_text: str, schema_info: Dict[str, Any], 
                      dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a Scalyr query using LangChain
        
        Args:
            query_text: Natural language query text
            schema_info: Schema information for context
            dataset: Optional dataset name
            
        Returns:
            Dictionary containing the generated query
        """
        if not langchain_available or not self.llm:
            return {
                "error": "LangChain or LLM not available",
                "message": "Make sure to install the required dependencies"
            }
        
        try:
            # Create the prompt
            prompt_text = self._create_prompt(query_text, schema_info, dataset)
            prompt = PromptTemplate.from_template(prompt_text)
            
            # Create the output parser
            output_parser = PydanticOutputParser(pydantic_object=ScalyrQueryOutput)
            
            # Create the chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Run the chain
            result = chain.invoke({})
            
            # Parse the result
            return self._parse_result(result)
            
        except Exception as e:
            return {
                "error": str(e),
                "message": "Failed to generate query with LangChain"
            }
    
    def _create_prompt(self, query_text: str, schema_info: Dict[str, Any], 
                      dataset: Optional[str] = None) -> str:
        """
        Create a prompt for the LLM
        
        Args:
            query_text: Natural language query text
            schema_info: Schema information
            dataset: Optional dataset name
            
        Returns:
            Prompt string
        """
        # Start with the documentation
        prompt_parts = ["""
# Scalyr PowerQuery Documentation

PowerQuery is a powerful query language for analyzing log data in Scalyr. It uses a pipeline syntax with commands separated by the pipe character (|).

## Basic Syntax

```
dataset="dataset_name"
| filter expression
| command1 arg1, arg2
| command2 arg1, arg2
```

## Common Commands

- `filter`: Filter logs based on conditions
- `search`: Search for text in logs
- `parse`: Extract fields using regular expressions
- `let`: Create new fields
- `group`: Group results by fields
- `count`: Count events
- `sum`: Sum values
- `avg`: Calculate average
- `min`/`max`: Find minimum/maximum values
- `sort`: Sort results
- `limit`: Limit number of results
- `timeslice`: Group results by time intervals

## Filter Expressions

- Equality: `field == value`
- Inequality: `field != value`
- Comparison: `field > value`, `field >= value`, `field < value`, `field <= value`
- Logical operators: `&&` (AND), `||` (OR), `!` (NOT)
- Contains: `contains(field, "text")`
- Matches: `matches(field, "regex")`

## Examples

Count HTTP status codes:
```
dataset="accesslog"
| filter statusCode >= 400
| group statusCode
| count
```

Calculate average response time by endpoint:
```
dataset="accesslog"
| group uriPath
| avg(responseTime) as avgTime
| sort -avgTime
| limit 10
```

Find error rates over time:
```
dataset="application_logs"
| filter level == "ERROR"
| timeslice 1h
| group _timeslice
| count as errorCount
```
"""]
        
        # Add schema information
        prompt_parts.append("\n## Available Datasets\n")
        if "datasets" in schema_info and schema_info["datasets"]:
            for ds in schema_info["datasets"]:
                prompt_parts.append(f"- {ds}")
        else:
            prompt_parts.append("- No datasets available")
        
        # Add field information
        prompt_parts.append("\n## Available Fields\n")
        if "fields" in schema_info and schema_info["fields"]:
            for field_name, field_info in schema_info["fields"].items():
                prompt_parts.append(f"- {field_name}")
        else:
            prompt_parts.append("- No field information available")
        
        # Add sample logs if available
        if "sample_logs" in schema_info and schema_info["sample_logs"]:
            prompt_parts.append("\n## Sample Logs\n")
            prompt_parts.append("```json")
            for i, log in enumerate(schema_info["sample_logs"]):
                if i > 0:
                    prompt_parts.append("---")
                prompt_parts.append(json.dumps(log, indent=2))
            prompt_parts.append("```")
        
        # Add instructions
        prompt_parts.append("\n## Instructions\n")
        prompt_parts.append("Generate a Scalyr query for the following request:")
        prompt_parts.append(f"\"{query_text}\"")
        
        if dataset:
            prompt_parts.append(f"\nUse the dataset \"{dataset}\" for this query.")
        
        prompt_parts.append("\nProvide your response in the following JSON format:")
        prompt_parts.append("```json")
        prompt_parts.append("""{
  "filter_query": "Simple filter query if applicable",
  "power_query": "Power query with full syntax",
  "explanation": "Explanation of what the query does",
  "visualization_suggestions": ["Suggestion 1", "Suggestion 2"]
}""")
        prompt_parts.append("```")
        
        # Join all parts
        return "\n".join(prompt_parts)
    
    def _parse_result(self, result: str) -> Dict[str, Any]:
        """
        Parse the result from an LLM
        
        Args:
            result: Result string from the LLM
            
        Returns:
            Parsed result as a dictionary
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no JSON block found, try to parse the entire response
            return json.loads(result)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, extract parts manually
            parsed_result = {
                "explanation": "Generated query (could not parse as JSON)",
                "raw_response": result
            }
            
            # Try to extract filter query
            filter_match = re.search(r'filter[_\s]+query:?\s*(.+?)(?:\n\n|\n[a-z])', result, re.IGNORECASE | re.DOTALL)
            if filter_match:
                parsed_result["filter_query"] = filter_match.group(1).strip()
            
            # Try to extract power query
            power_match = re.search(r'power[_\s]+query:?\s*(.+?)(?:\n\n|\n[a-z])', result, re.IGNORECASE | re.DOTALL)
            if power_match:
                parsed_result["power_query"] = power_match.group(1).strip()
            
            return parsed_result


def create_llm(provider: str = "openai", model_name: Optional[str] = None) -> LangChainLLM:
    """
    Create a LangChain LLM instance
    
    Args:
        provider: LLM provider to use (openai, anthropic, google)
        model_name: Optional specific model name to use
        
    Returns:
        LangChainLLM instance
    """
    return LangChainLLM(provider, model_name)
