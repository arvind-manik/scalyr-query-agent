"""
Scalyr API Client

This module provides a client for interacting with the Scalyr API.
"""

import json
import os
import requests
import logging
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalyrAPIClient:
    """
    Client for interacting with the Scalyr API
    
    This class provides methods for querying Scalyr and retrieving schema information.
    """
    
    def __init__(self, api_token: Optional[str] = None, server_url: Optional[str] = None):
        """
        Initialize the Scalyr API client
        
        Args:
            api_token: Scalyr API token (defaults to SCALYR_API_TOKEN environment variable)
            server_url: Scalyr server URL (defaults to SCALYR_SERVER_URL environment variable or https://app.scalyr.com)
        """
        self.api_token = api_token or os.environ.get("SCALYR_API_TOKEN")
        self.server_url = server_url or os.environ.get("SCALYR_SERVER_URL", "https://app.scalyr.com")
        
        # Check if API token is available
        if not self.api_token:
            logger.warning("No Scalyr API token provided. Set SCALYR_API_TOKEN environment variable or pass api_token.")
    
    def get_fields(self, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available fields for a dataset
        
        Args:
            dataset: Optional dataset name to filter fields
            
        Returns:
            Dictionary of field information
        """
        # First, get server hosts for the dataset
        server_hosts = self.get_field_values("serverHost", dataset)
        if not server_hosts:
            logger.warning(f"No server hosts found for dataset {dataset}")
            return {}
        
        # Use the first server host as default
        server_host = server_hosts[0]
        logger.info(f"Using server host: {server_host}")
        
        # Endpoint for field information
        endpoint = f"{self.server_url}/api/facetQuery"
        
        # Prepare request payload
        payload = {
            "token": self.api_token,
            "queryType": "facet",
            "field": "*",
            "startTime": "1d",
            "maxCount": 100,
            "filter": f'serverHost="{server_host}"'
        }
        
        logger.info(f"Fetching fields from Scalyr API: {endpoint}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make the API request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            logger.debug(f"API response: {json.dumps(data, indent=2)}")
            
            # Check for errors
            if data.get("status") != "success":
                logger.error(f"Error from Scalyr API: {data.get('message', 'Unknown error')}")
                return {}
            
            # Extract fields from values
            fields = {}
            for item in data.get("values", []):
                field_name = item["value"]
                fields[field_name] = {
                    "type": "string",  # Default type
                    "count": item.get("count", 0)
                }
            
            # Add server host information
            fields["serverHost"] = {
                "type": "string",
                "values": server_hosts
            }
            
            logger.info(f"Found {len(fields)} fields")
            return fields
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching fields: {e}")
            return {}
    
    def get_datasets(self) -> List[str]:
        """
        Get available datasets
        
        Returns:
            List of dataset names
        """
        # Endpoint for facet query (to get datasets)
        endpoint = f"{self.server_url}/api/facetQuery"
        
        # Prepare request payload
        payload = {
            "token": self.api_token,
            "queryType": "facet",
            "field": "source",
            "startTime": "1d",
            "maxCount": 100
        }
        
        logger.info(f"Fetching datasets from Scalyr API: {endpoint}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make the API request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            logger.debug(f"API response: {json.dumps(data, indent=2)}")
            
            # Check for errors
            if data.get("status") != "success":
                logger.error(f"Error from Scalyr API: {data.get('message', 'Unknown error')}")
                return []
            
            # Extract dataset names from values
            datasets = [item["value"] for item in data.get("values", [])]
            logger.info(f"Found {len(datasets)} datasets")
            return datasets
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching datasets: {e}")
            return []
    
    def execute_query(self, query: str, start_time: str = "1h", end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a PowerQuery
        
        Args:
            query: PowerQuery to execute
            start_time: Start time for the query (e.g., "1h" for 1 hour ago)
            end_time: Optional end time for the query
            
        Returns:
            Query results
        """
        # Endpoint for PowerQuery
        endpoint = f"{self.server_url}/api/powerQuery"
        
        # Prepare request payload
        payload = {
            "token": self.api_token,
            "query": query,
            "startTime": start_time
        }
        
        if end_time:
            payload["endTime"] = end_time
        
        logger.info(f"Executing PowerQuery: {query}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make the API request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            logger.debug(f"API response: {json.dumps(data, indent=2)}")
            
            # Check for errors
            if data.get("status") != "success":
                logger.error(f"Error from Scalyr API: {data.get('message', 'Unknown error')}")
                return {}
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing query: {e}")
            return {}
    
    def execute_filter_query(self, filter_expression: str, start_time: str = "1h", end_time: Optional[str] = None, 
                            max_count: int = 100) -> Dict[str, Any]:
        """
        Execute a filter query
        
        Args:
            filter_expression: Filter expression
            start_time: Start time for the query (e.g., "1h" for 1 hour ago)
            end_time: Optional end time for the query
            max_count: Maximum number of results to return
            
        Returns:
            Query results
        """
        # Endpoint for log query
        endpoint = f"{self.server_url}/api/query"
        
        # Prepare request payload
        payload = {
            "token": self.api_token,
            "filter": filter_expression,
            "startTime": start_time,
            "maxCount": max_count,
            "columns": ["timestamp", "message", "data"]
        }
        
        if end_time:
            payload["endTime"] = end_time
        
        logger.info(f"Executing filter query: {filter_expression}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make the API request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            logger.debug(f"API response: {json.dumps(data, indent=2)}")
            
            # Check for errors
            if data.get("status") != "success":
                logger.error(f"Error from Scalyr API: {data.get('message', 'Unknown error')}")
                return {}
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing filter query: {e}")
            return {}
    
    def get_sample_logs(self, dataset: Optional[str] = None, max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Get sample logs from a dataset
        
        Args:
            dataset: Optional dataset name to filter logs
            max_count: Maximum number of logs to return
            
        Returns:
            List of sample logs
        """
        # Prepare filter expression
        filter_expression = ""
        if dataset:
            filter_expression = f'$source="{dataset}"'
        
        # Execute the query
        result = self.execute_filter_query(filter_expression, max_count=max_count)
        
        # Extract matches
        return result.get("matches", [])
    
    def get_field_values(self, field: str, dataset: Optional[str] = None, max_count: int = 100) -> List[str]:
        """
        Get common values for a field
        
        Args:
            field: Field name
            dataset: Optional dataset name to filter values
            max_count: Maximum number of values to return
            
        Returns:
            List of common values for the field
        """
        # Endpoint for facet query
        endpoint = f"{self.server_url}/api/facetQuery"
        
        # Prepare request payload
        payload = {
            "token": self.api_token,
            "queryType": "facet",
            "field": field,
            "startTime": "1d",
            "maxCount": max_count
        }
        
        if dataset:
            payload["filter"] = f'source="{dataset}"'
        
        logger.info(f"Fetching values for field {field} from Scalyr API: {endpoint}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Make the API request
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            logger.debug(f"API response: {json.dumps(data, indent=2)}")
            
            # Check for errors
            if data.get("status") != "success":
                logger.error(f"Error from Scalyr API: {data.get('message', 'Unknown error')}")
                return []
            
            # Extract values
            values = [item["value"] for item in data.get("values", [])]
            logger.info(f"Found {len(values)} values for field {field}")
            return values
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching field values: {e}")
            return []
