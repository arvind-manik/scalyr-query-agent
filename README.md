# Scalyr Query Generator Agent

A powerful, LLM-powered query generator for Scalyr datasets that converts natural language requests into Scalyr queries.

## Overview

This project provides a flexible and extensible query generator for Scalyr that can:

1. Connect to Scalyr API to discover available fields and log sources
2. Convert natural language requests into Scalyr queries
3. Support multiple query types (filter, power)
4. Suggest appropriate visualizations for queries
5. Integrate with LLMs for more sophisticated query generation

## Features

- **Scalyr API Integration**: Connects to Scalyr API to discover schema and execute queries
- **Natural Language Processing**: Convert plain English requests into Scalyr queries
- **Multiple Query Types**: Support for Scalyr filter and power queries
- **Rule-based Fallback**: Works even without LLM integration
- **Visualization Suggestions**: Automatically suggests appropriate visualizations
- **CLI Interface**: Command-line interface for interactive query building
- **LangChain Integration**: Flexible LLM integration via LangChain with support for Google Gemini, OpenAI, or Anthropic models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scalyr-query-agent.git
cd scalyr-query-agent

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Scalyr API token and optionally LLM API keys:

```bash
# Scalyr API token (required)
export SCALYR_API_TOKEN=your_scalyr_api_token_here

# Optional: Scalyr server URL (defaults to https://app.scalyr.com)
export SCALYR_SERVER_URL=your_scalyr_server_url

# Optional: LLM API keys for different providers

# For Google Gemini
export GOOGLE_API_KEY=your_google_api_key_here

# For OpenAI
export OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Usage

### Command-line Interface

```bash
# Basic usage
python src/cli.py "Show me 4xx errors in the authentication service"

# Interactive mode
python src/cli.py --interactive

# Specify a dataset
python src/cli.py --dataset "service-logs" "Show me 4xx errors"

# Use a specific LLM provider
python src/cli.py --llm gemini "Analyze API performance over the last 24 hours"

# Use a specific model with an LLM provider
python src/cli.py --llm openai --model gpt-4-turbo "Analyze API performance over the last 24 hours"

# Save the generated query to a file
python src/cli.py "Show me failed login attempts" --save query.json

# Execute the generated query against Scalyr
python src/cli.py "Show me error logs" --execute

# Don't use LLM, use rule-based approach only
python src/cli.py --llm none "Show me error logs"
```

### Python API

```python
from scalyr_api_client import ScalyrAPIClient
from scalyr_query_generator import ScalyrQueryGenerator

# Create API client
api_client = ScalyrAPIClient(api_token="your_api_token_here")

# Create a query generator without LLM integration
query_generator = ScalyrQueryGenerator(api_client, llm_provider=None)

# Generate a query from natural language
result = query_generator.generate_query(
    "Show me 4xx errors in the authentication service"
)

# Print the generated queries
print(f"Filter Query: {result['filter_query']}")
print(f"Power Query: {result['power_query']}")

# Create a query generator with LLM integration (using LangChain)
# You can use any of these providers: "gemini", "openai", "anthropic"
llm_query_generator = ScalyrQueryGenerator(
    api_client,
    llm_provider="gemini",
    model_name="gemini-pro"  # Optional: specify a model name
)

# Generate a more complex query
result = llm_query_generator.generate_query(
    "Find unusual spikes in API latency across all services yesterday"
)

# Execute the query against Scalyr
execution_result = api_client.execute_query(result["power_query"])
```

## Example Queries

Here are some example natural language queries you can try:

- "Show me 4xx errors in the authentication service"
- "Analyze the performance of the payment service API endpoints over the last 24 hours"
- "Show me failed login attempts in the last week"
- "Find unusual spikes in API latency across all services yesterday"
- "Count the number of errors by service in the last hour"
- "Show me the top 10 slowest API endpoints"
- "Analyze security events for suspicious login attempts"

## Project Structure

- `src/scalyr_api_client.py`: Client for interacting with the Scalyr API
- `src/scalyr_query_generator.py`: Core query generator implementation
- `src/langchain_integration.py`: LangChain integration for LLM providers
- `src/cli.py`: Command-line interface
- `src/example_usage.py`: Example usage of the query generator
- `src/scalyr_query_builder.py`: Legacy query builder (for backward compatibility)

## How It Works

1. **Schema Discovery**: The query generator connects to the Scalyr API to discover available datasets, fields, and sample logs.

2. **Natural Language Processing**: When a natural language query is received, the generator either:
   - Uses LangChain with an LLM to interpret the query and generate Scalyr queries (if LLM integration is enabled)
   - Uses rule-based processing to extract key terms and entities from the query (as fallback)

3. **Query Generation**: Based on the interpreted query, the generator creates appropriate filter and power queries.

4. **Visualization Suggestions**: The generator suggests appropriate visualizations based on the query type and content.

5. **Query Execution**: Optionally, the generated queries can be executed against Scalyr to retrieve results.

## Extending the Query Generator

### Adding Custom Rule-Based Processing

You can extend the rule-based processing by modifying the `_extract_terms_and_entities` method in `ScalyrQueryGenerator`:

```python
def _extract_terms_and_entities(self, query_text: str, schema_info: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    # Existing code...

    # Add custom entity extraction
    if "your_custom_pattern" in query_text.lower():
        entities["custom_entity"] = "custom_value"

    return terms, entities
```

### Customizing LLM Prompts

You can customize the prompts sent to LLMs by modifying the `_create_prompt` method in `langchain_integration.py`:

```python
def _create_prompt(self, query_text: str, schema_info: Dict[str, Any], dataset: Optional[str] = None) -> str:
    # Existing code...

    # Add custom instructions
    prompt_parts.append("\n## Custom Instructions\n")
    prompt_parts.append("Your custom instructions here...")

    return "\n".join(prompt_parts)
```

### Adding Support for New LLM Providers

You can add support for new LLM providers by extending the `LangChainLLM` class in `langchain_integration.py`:

```python
def _init_new_provider(self, model_name: Optional[str] = None):
    """Initialize a new LLM provider"""
    try:
        from langchain_new_provider import ChatNewProvider

        # Get API key from environment
        api_key = os.environ.get("NEW_PROVIDER_API_KEY")
        if not api_key:
            print("Warning: No API key found in environment (NEW_PROVIDER_API_KEY)")

        # Use default model if none specified
        model = model_name or "default-model"

        # Initialize the LLM
        self.llm = ChatNewProvider(
            model=model,
            temperature=0.2,
            api_key=api_key
        )
    except ImportError:
        print("New provider integration not available. Install with: pip install langchain-new-provider")
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
