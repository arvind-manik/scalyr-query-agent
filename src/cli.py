"""
Command-line interface for the Scalyr Query Generator

This module provides a command-line interface for the Scalyr Query Generator.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from scalyr_api_client import ScalyrAPIClient
from scalyr_query_generator import ScalyrQueryGenerator

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Scalyr Query Generator CLI")

    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query to convert to Scalyr query"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--dataset",
        help="Dataset to query"
    )

    parser.add_argument(
        "--llm",
        choices=["gemini", "openai", "anthropic", "none"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )

    parser.add_argument(
        "--model",
        help="Specific model name to use with the LLM provider"
    )

    parser.add_argument(
        "--output",
        choices=["json", "pretty", "text"],
        default="pretty",
        help="Output format (default: pretty)"
    )

    parser.add_argument(
        "--save",
        help="Save the generated query to a file"
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated query against Scalyr"
    )

    parser.add_argument(
        "--api-token",
        help="Scalyr API token (defaults to SCALYR_API_TOKEN environment variable)"
    )

    parser.add_argument(
        "--server-url",
        help="Scalyr server URL (defaults to SCALYR_SERVER_URL environment variable)"
    )

    return parser.parse_args()


def format_query_result(result: Dict[str, Any], format_type: str) -> str:
    """
    Format a query result for output

    Args:
        result: Query result dictionary
        format_type: Output format (json, pretty, text)

    Returns:
        Formatted string
    """
    if format_type == "json":
        return json.dumps(result, indent=None)

    elif format_type == "pretty":
        output = []

        if "explanation" in result:
            output.append(f"Explanation: {result['explanation']}\n")

        if "filter_query" in result and result["filter_query"]:
            output.append("Filter Query:")
            output.append(f"{result['filter_query']}\n")

        if "power_query" in result and result["power_query"]:
            output.append("Power Query:")
            output.append(f"{result['power_query']}\n")

        if "visualization_suggestions" in result and result["visualization_suggestions"]:
            output.append("Visualization Suggestions:")
            for suggestion in result["visualization_suggestions"]:
                output.append(f"- {suggestion}")

        if "error" in result:
            output.append(f"\nError: {result['error']}")
            if "message" in result:
                output.append(f"Message: {result['message']}")

        return "\n".join(output)

    elif format_type == "text":
        # Simple text format, just the queries
        output = []

        if "filter_query" in result and result["filter_query"]:
            output.append(result["filter_query"])

        if "power_query" in result and result["power_query"]:
            output.append(result["power_query"])

        return "\n\n".join(output)

    return "Unknown format type"


def save_query_to_file(result: Dict[str, Any], filename: str) -> None:
    """
    Save a query result to a file

    Args:
        result: Query result dictionary
        filename: Name of the file to save to
    """
    # Save to file
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Query saved to {filename}")


def execute_query(api_client: ScalyrAPIClient, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a query against Scalyr

    Args:
        api_client: ScalyrAPIClient instance
        result: Query result dictionary

    Returns:
        Query execution results
    """
    # Check if we have a power query
    if "power_query" in result and result["power_query"]:
        print("Executing PowerQuery...")
        return api_client.execute_query(result["power_query"])

    # Fall back to filter query
    elif "filter_query" in result and result["filter_query"]:
        print("Executing Filter Query...")
        return api_client.execute_filter_query(result["filter_query"])

    return {"error": "No query to execute"}


def interactive_mode(query_generator: ScalyrQueryGenerator, api_client: ScalyrAPIClient,
                    output_format: str, execute_queries: bool) -> None:
    """
    Run the CLI in interactive mode

    Args:
        query_generator: ScalyrQueryGenerator instance
        api_client: ScalyrAPIClient instance
        output_format: Output format
        execute_queries: Whether to execute generated queries
    """
    print("Scalyr Query Generator Interactive Mode")
    print("Enter your query, or 'exit' to quit")

    # Get available datasets
    datasets = api_client.get_datasets()
    if datasets:
        print("\nAvailable datasets:")
        for ds in datasets:
            print(f"- {ds}")

    while True:
        try:
            # Get query
            query = input("\nQuery> ")

            if query.lower() in ["exit", "quit", "q"]:
                break

            if not query.strip():
                continue

            # Get dataset
            dataset = None
            dataset_input = input("Dataset (leave empty for auto-detection)> ")
            if dataset_input.strip():
                dataset = dataset_input.strip()

            # Generate query
            result = query_generator.generate_query(query, dataset)
            print("\n" + format_query_result(result, output_format))

            # Execute query if requested
            if execute_queries and ("power_query" in result or "filter_query" in result):
                execute = input("\nExecute this query? (y/n)> ")
                if execute.lower() == "y":
                    execution_result = execute_query(api_client, result)
                    print("\nExecution Result:")
                    print(json.dumps(execution_result, indent=2))

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point for the CLI"""
    args = parse_args()

    # Get API token and server URL from environment or command line
    api_token = args.api_token or os.getenv("SCALYR_API_TOKEN")
    server_url = args.server_url or os.getenv("SCALYR_SERVER_URL")

    if not api_token:
        print("Error: No API token provided. Set SCALYR_API_TOKEN in .env file or use --api-token")
        sys.exit(1)

    # Create API client
    api_client = ScalyrAPIClient(api_token=api_token, server_url=server_url)

    # Create query generator
    llm_provider = args.llm if args.llm != "none" else None

    # Create the query generator with model name if provided
    try:
        query_generator = ScalyrQueryGenerator(api_client, llm_provider, args.model)
    except Exception as e:
        print(f"Error initializing query generator: {e}")
        print("Make sure you have the correct API keys set in your .env file")
        sys.exit(1)

    # Run in interactive mode if requested
    if args.interactive:
        interactive_mode(query_generator, api_client, args.output, args.execute)
        return

    # Otherwise, process the query from command line
    if not args.query:
        print("Error: No query provided")
        sys.exit(1)

    try:
        # Generate query
        result = query_generator.generate_query(args.query, args.dataset)

        # Format and print result
        print(format_query_result(result, args.output))

        # Save to file if requested
        if args.save:
            save_query_to_file(result, args.save)

        # Execute query if requested
        if args.execute:
            execution_result = execute_query(api_client, result)
            print("\nExecution Result:")
            print(json.dumps(execution_result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
