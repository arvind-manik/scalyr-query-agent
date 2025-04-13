"""Example usage of the Scalyr Query Generator

This module demonstrates how to use the Scalyr Query Generator.
"""

# Import the query generator
try:
    from scalyr_api_client import ScalyrAPIClient
    from scalyr_query_generator import ScalyrQueryGenerator
    generator_available = True
except ImportError as e:
    print(f"Query generator not available: {e}")
    print("Make sure to install required dependencies: pip install -r requirements.txt")
    generator_available = False


def query_generator_examples():
    """Examples using the new query generator"""
    # Create API client
    api_client = ScalyrAPIClient()

    # Create query generator without LLM (rule-based only)
    query_generator = ScalyrQueryGenerator(api_client, llm_provider=None)

    # Example 1: Simple error query
    print("Example 1: Simple error query")
    query = "Show me 4xx errors in the authentication service"
    result = query_generator.generate_query(query)

    print(f"Query: {query}")
    if "filter_query" in result and result["filter_query"]:
        print(f"Filter Query: {result['filter_query']}")
    if "power_query" in result and result["power_query"]:
        print(f"Power Query: {result['power_query']}")
    if "visualization_suggestions" in result and result["visualization_suggestions"]:
        print("Visualization Suggestions:")
        for suggestion in result["visualization_suggestions"]:
            print(f"- {suggestion}")
    print("\n")

    # Example 2: Performance analysis
    print("Example 2: Performance analysis")
    query = "Analyze the performance of the payment service API endpoints over the last 24 hours"
    result = query_generator.generate_query(query)

    print(f"Query: {query}")
    if "filter_query" in result and result["filter_query"]:
        print(f"Filter Query: {result['filter_query']}")
    if "power_query" in result and result["power_query"]:
        print(f"Power Query: {result['power_query']}")
    if "visualization_suggestions" in result and result["visualization_suggestions"]:
        print("Visualization Suggestions:")
        for suggestion in result["visualization_suggestions"]:
            print(f"- {suggestion}")
    print("\n")

    # Example 3: Security analysis
    print("Example 3: Security analysis")
    query = "Show me failed login attempts in the last week"
    result = query_generator.generate_query(query)

    print(f"Query: {query}")
    if "filter_query" in result and result["filter_query"]:
        print(f"Filter Query: {result['filter_query']}")
    if "power_query" in result and result["power_query"]:
        print(f"Power Query: {result['power_query']}")
    if "visualization_suggestions" in result and result["visualization_suggestions"]:
        print("Visualization Suggestions:")
        for suggestion in result["visualization_suggestions"]:
            print(f"- {suggestion}")
    print("\n")


def llm_integration_example():
    """Example with LLM integration"""
    try:
        # Create API client
        api_client = ScalyrAPIClient()

        # Try different LLM providers with specific models
        provider_models = [
            ("gemini", "gemini-pro"),
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-opus-20240229")
        ]

        for provider, model in provider_models:
            try:
                print(f"\nTrying with {provider} ({model}):")

                # Create a query generator with LLM integration and specific model
                llm_query_generator = ScalyrQueryGenerator(api_client, llm_provider=provider, model_name=model)

                query = "Find unusual spikes in API latency across all services yesterday"
                result = llm_query_generator.generate_query(query)

                print(f"Query: {query}")
                if "filter_query" in result and result["filter_query"]:
                    print(f"Filter Query: {result['filter_query']}")
                if "power_query" in result and result["power_query"]:
                    print(f"Power Query: {result['power_query']}")
                if "visualization_suggestions" in result and result["visualization_suggestions"]:
                    print("Visualization Suggestions:")
                    for suggestion in result["visualization_suggestions"]:
                        print(f"- {suggestion}")
                if "error" in result:
                    print(f"Error: {result['error']}")
                    if "message" in result:
                        print(f"Message: {result['message']}")

                # If successful, break the loop
                if "error" not in result:
                    break

            except Exception as e:
                print(f"Failed with {provider}: {e}")
    except Exception as e:
        print(f"LLM integration example failed: {e}")


def main():
    """Main function demonstrating all examples"""
    # Run the examples if dependencies are available
    if generator_available:
        # Run the basic examples
        query_generator_examples()

        # Try the LLM integration example
        llm_integration_example()
    else:
        print("Skipping examples due to missing dependencies.")


if __name__ == "__main__":
    main()