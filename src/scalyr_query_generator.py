"""
Scalyr Query Generator

This module provides a query generator for Scalyr that uses LLMs to interpret
natural language requests and convert them into Scalyr queries.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Union, Tuple

from scalyr_api_client import ScalyrAPIClient


# Documentation snippets for Scalyr PowerQuery
POWER_QUERY_DOCS = """
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
"""


class ScalyrQueryGenerator:
    """
    LLM-powered Scalyr Query Generator

    This class uses LLMs to interpret natural language requests and convert them
    into Scalyr queries.
    """

    def __init__(self, api_client: Optional[ScalyrAPIClient] = None, llm_provider: str = "gemini", model_name: Optional[str] = None):
        """
        Initialize the query generator

        Args:
            api_client: Optional ScalyrAPIClient instance
            llm_provider: LLM provider to use (gemini, openai, anthropic)
            model_name: Optional specific model name to use
        """
        self.api_client = api_client or ScalyrAPIClient()
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.schema_cache = {}

    def generate_query(self, query_text: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a Scalyr query from natural language

        Args:
            query_text: Natural language query text
            dataset: Optional dataset to query

        Returns:
            Dictionary containing the generated queries and explanations
        """
        # Get schema information
        schema_info = self._get_schema_info(dataset)

        # Generate the query using LLM
        return self._generate_with_llm(query_text, schema_info, dataset)

    def _get_schema_info(self, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get schema information for context

        Args:
            dataset: Optional dataset name

        Returns:
            Dictionary of schema information
        """
        # Check if we have cached schema info
        cache_key = dataset or "_all_"
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]

        schema_info = {}

        # Get available datasets
        datasets = self.api_client.get_datasets()
        schema_info["datasets"] = datasets

        # If dataset is specified, get fields for that dataset
        if dataset and dataset in datasets:
            fields = self.api_client.get_fields(dataset)
            schema_info["fields"] = fields

            # Get sample logs
            sample_logs = self.api_client.get_sample_logs(dataset, max_count=3)
            schema_info["sample_logs"] = sample_logs
        else:
            # Get fields for all datasets
            fields = self.api_client.get_fields()
            schema_info["fields"] = fields

        # Cache the schema info
        self.schema_cache[cache_key] = schema_info

        return schema_info

    def _generate_with_llm(self, query_text: str, schema_info: Dict[str, Any],
                          dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a query using an LLM

        Args:
            query_text: Natural language query text
            schema_info: Schema information for context
            dataset: Optional dataset name

        Returns:
            Dictionary containing the generated queries
        """
        try:
            # Try to import LangChain integration
            from langchain_integration import create_llm, langchain_available

            if not langchain_available:
                print("LangChain not available, falling back to rule-based approach")
                return self._generate_with_rules(query_text, schema_info, dataset)

            # Map provider names
            provider_map = {
                "gemini": "google",
                "openai": "openai",
                "anthropic": "anthropic"
            }

            # Get the provider
            provider = provider_map.get(self.llm_provider, "openai")

            # Create the LLM with model name if provided
            llm = create_llm(provider, self.model_name)

            # Generate the query
            return llm.generate_query(query_text, schema_info, dataset)

        except ImportError:
            print("LangChain integration not available, falling back to rule-based approach")
            return self._generate_with_rules(query_text, schema_info, dataset)
        except Exception as e:
            print(f"Error using LangChain: {e}")
            return {
                "error": str(e),
                "message": "Failed to generate query with LangChain"
            }

    def _generate_with_rules(self, query_text: str, schema_info: Dict[str, Any],
                            dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a query using rule-based approach (fallback)

        Args:
            query_text: Natural language query text
            schema_info: Schema information for context
            dataset: Optional dataset name

        Returns:
            Dictionary containing the generated queries
        """
        # Extract key terms and entities
        terms, entities = self._extract_terms_and_entities(query_text, schema_info)

        # Determine the dataset to use
        target_dataset = dataset or entities.get("dataset")
        if not target_dataset and "datasets" in schema_info and schema_info["datasets"]:
            # Use the first available dataset if none specified
            target_dataset = schema_info["datasets"][0]

        # Build a simple filter query
        filter_query = self._build_simple_filter(terms, entities)

        # Build a power query
        power_query = self._build_simple_power_query(target_dataset, terms, entities)

        return {
            "filter_query": filter_query,
            "power_query": power_query,
            "explanation": "Generated a simple query based on the natural language input.",
            "visualization_suggestions": ["Table view of matching logs"]
        }

    def _extract_terms_and_entities(self, query_text: str,
                                   schema_info: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Extract key terms and entities from a query

        Args:
            query_text: Natural language query text
            schema_info: Schema information

        Returns:
            Tuple of (terms, entities)
        """
        terms = []
        entities = {}

        # Look for dataset mentions
        if "datasets" in schema_info:
            for dataset in schema_info["datasets"]:
                if dataset.lower() in query_text.lower():
                    entities["dataset"] = dataset
                    break

        # Look for field mentions
        if "fields" in schema_info:
            for field_name, field_info in schema_info["fields"].items():
                if field_name.lower() in query_text.lower():
                    if "fields" not in entities:
                        entities["fields"] = []
                    entities["fields"].append(field_name)

        # Look for time ranges
        time_match = re.search(r'(last|past)\s+(\d+)\s+(minute|hour|day|week|month)s?', query_text, re.IGNORECASE)
        if time_match:
            amount = int(time_match.group(2))
            unit = time_match.group(3).lower()
            if unit == "minute":
                entities["time_range"] = f"{amount}m"
            elif unit == "hour":
                entities["time_range"] = f"{amount}h"
            elif unit == "day":
                entities["time_range"] = f"{amount}d"
            elif unit == "week":
                entities["time_range"] = f"{amount}w"
            elif unit == "month":
                entities["time_range"] = f"{amount * 30}d"  # Approximate

        # Look for limit
        limit_match = re.search(r'(top|limit)\s+(\d+)', query_text, re.IGNORECASE)
        if limit_match:
            entities["limit"] = int(limit_match.group(2))
        else:
            entities["limit"] = 100  # Default limit

        # Extract key terms for filtering
        query_lower = query_text.lower()

        # Look for error-related terms
        if any(term in query_lower for term in ["error", "exception", "fail", "4xx", "5xx", "400", "500"]):
            terms.append("error")
            if "error" in query_lower or "exception" in query_lower:
                terms.append('level=="error"')
            if "4xx" in query_lower or "400" in query_lower:
                terms.append('statusCode >= 400 statusCode < 500')
            if "5xx" in query_lower or "500" in query_lower:
                terms.append('statusCode >= 500 statusCode < 600')

        # Look for performance-related terms
        if any(term in query_lower for term in ["slow", "latency", "performance", "duration", "response time"]):
            terms.append("performance")
            if "fields" in entities and "duration" in entities["fields"]:
                terms.append('duration > 1000')  # Assuming duration is in ms

        # Look for security-related terms
        if any(term in query_lower for term in ["security", "auth", "login", "access", "permission"]):
            terms.append("security")
            if "login" in query_lower or "auth" in query_lower:
                terms.append('action=="login"')

        return terms, entities

    def _build_simple_filter(self, terms: List[str], entities: Dict[str, Any]) -> str:
        """
        Build a simple filter query

        Args:
            terms: Extracted terms
            entities: Extracted entities

        Returns:
            Filter query string
        """
        filter_parts = []

        # Add field-based filters
        if "fields" in entities:
            for field in entities["fields"]:
                # For now, just add a simple existence check
                filter_parts.append(f'{field}!=""')

        # Add term-based filters
        for term in terms:
            if "==" in term or "!=" in term or ">" in term or "<" in term:
                # This is already a filter expression
                filter_parts.append(term)
            elif term not in ["error", "performance", "security"]:
                # Add as a text search
                filter_parts.append(f'"{term}"')

        # If no filters, return empty string
        if not filter_parts:
            return ""

        # Join with AND
        return " ".join(filter_parts)

    def _build_simple_power_query(self, dataset: Optional[str], terms: List[str],
                                 entities: Dict[str, Any]) -> str:
        """
        Build a simple power query

        Args:
            dataset: Dataset name
            terms: Extracted terms
            entities: Extracted entities

        Returns:
            Power query string
        """
        # Start with dataset
        if not dataset:
            dataset = "logs"

        query_lines = [f'dataset="{dataset}"']

        # Add filter
        filter_expr = self._build_simple_filter(terms, entities)
        if filter_expr:
            query_lines.append(f"| filter {filter_expr}")

        # Add grouping if appropriate
        if "error" in terms and "fields" in entities:
            # Group errors by relevant fields
            group_fields = []
            for field in ["statusCode", "path", "method", "level", "component"]:
                if field in entities["fields"]:
                    group_fields.append(field)

            if group_fields:
                query_lines.append(f"| group {', '.join(group_fields)}")
                query_lines.append("| count as errorCount")
                query_lines.append("| sort -errorCount")

        elif "performance" in terms and "fields" in entities:
            # Group performance by relevant fields
            group_fields = []
            for field in ["path", "method", "endpoint", "service"]:
                if field in entities["fields"]:
                    group_fields.append(field)

            if group_fields:
                query_lines.append(f"| group {', '.join(group_fields)}")
                if "duration" in entities["fields"]:
                    query_lines.append("| avg(duration) as avgDuration, max(duration) as maxDuration")
                    query_lines.append("| sort -avgDuration")

        # Add limit
        limit = entities.get("limit", 100)
        query_lines.append(f"| limit {limit}")

        return "\n".join(query_lines)

    def _create_llm_prompt(self, query_text: str, schema_info: Dict[str, Any],
                          dataset: Optional[str] = None, for_system: bool = False) -> str:
        """
        Create a prompt for the LLM

        Args:
            query_text: Natural language query text
            schema_info: Schema information
            dataset: Optional dataset name
            for_system: Whether this is for a system prompt

        Returns:
            Prompt string
        """
        # Start with the documentation
        prompt_parts = [POWER_QUERY_DOCS]

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

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the response from an LLM

        Args:
            response_text: Response text from the LLM

        Returns:
            Parsed response as a dictionary
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)

            # If no JSON block found, try to parse the entire response
            return json.loads(response_text)

        except json.JSONDecodeError:
            # If JSON parsing fails, extract parts manually
            result = {
                "explanation": "Generated query (could not parse as JSON)",
                "raw_response": response_text
            }

            # Try to extract filter query
            filter_match = re.search(r'filter[_\s]+query:?\s*(.+?)(?:\n\n|\n[a-z])', response_text, re.IGNORECASE | re.DOTALL)
            if filter_match:
                result["filter_query"] = filter_match.group(1).strip()

            # Try to extract power query
            power_match = re.search(r'power[_\s]+query:?\s*(.+?)(?:\n\n|\n[a-z])', response_text, re.IGNORECASE | re.DOTALL)
            if power_match:
                result["power_query"] = power_match.group(1).strip()

            return result
