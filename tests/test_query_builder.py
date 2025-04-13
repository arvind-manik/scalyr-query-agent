"""
Tests for the Scalyr Query Builder

This module contains tests for the Scalyr Query Builder.
"""

import sys
import os
import unittest

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_query_builder import ScalyrQueryBuilder, QueryIntent, QueryType


class TestScalyrQueryBuilder(unittest.TestCase):
    """Tests for the ScalyrQueryBuilder class"""
    
    def setUp(self):
        """Set up the test case"""
        self.query_builder = ScalyrQueryBuilder()
    
    def test_determine_intent(self):
        """Test the _determine_intent method"""
        # Test error investigation intent
        self.assertEqual(
            self.query_builder._determine_intent("Show me errors in the auth service"),
            QueryIntent.ERROR_INVESTIGATION
        )
        
        # Test performance analysis intent
        self.assertEqual(
            self.query_builder._determine_intent("Analyze the performance of the API"),
            QueryIntent.PERFORMANCE_ANALYSIS
        )
        
        # Test trend analysis intent
        self.assertEqual(
            self.query_builder._determine_intent("Show me the trend of errors over time"),
            QueryIntent.TREND_ANALYSIS
        )
        
        # Test security analysis intent
        self.assertEqual(
            self.query_builder._determine_intent("Check for suspicious login attempts"),
            QueryIntent.SECURITY_ANALYSIS
        )
        
        # Test aggregation intent
        self.assertEqual(
            self.query_builder._determine_intent("Count errors by service"),
            QueryIntent.AGGREGATE_DATA
        )
        
        # Test default intent
        self.assertEqual(
            self.query_builder._determine_intent("Show me logs"),
            QueryIntent.FILTER_LOGS
        )
    
    def test_extract_entities(self):
        """Test the _extract_entities method"""
        # Test extracting service name
        entities = self.query_builder._extract_entities("Show me logs from service: auth-service")
        self.assertEqual(entities.get("service"), "auth-service")
        
        # Test extracting status code
        entities = self.query_builder._extract_entities("Show me 4xx errors")
        self.assertEqual(entities.get("min_status"), 400)
        self.assertEqual(entities.get("max_status"), 500)
        
        # Test extracting time range
        entities = self.query_builder._extract_entities("Show me logs from the last 24 hours")
        self.assertEqual(entities.get("time_range"), "24h")
        
        # Test extracting limit
        entities = self.query_builder._extract_entities("Show me the top 50 errors")
        self.assertEqual(entities.get("limit"), 50)
    
    def test_build_query_from_natural_language(self):
        """Test the build_query_from_natural_language method"""
        # Test building an error query
        result = self.query_builder.build_query_from_natural_language(
            "Show me 4xx errors in the authentication service"
        )
        self.assertIsNotNone(result.filter_query)
        self.assertIsNotNone(result.power_query)
        self.assertTrue(len(result.visualization_suggestions) > 0)
        
        # Test building a performance query
        result = self.query_builder.build_query_from_natural_language(
            "Analyze the performance of the API endpoints"
        )
        self.assertIsNotNone(result.power_query)
        self.assertTrue(len(result.visualization_suggestions) > 0)
        
        # Test building a security query
        result = self.query_builder.build_query_from_natural_language(
            "Show me failed login attempts"
        )
        self.assertIsNotNone(result.power_query)
        self.assertTrue(len(result.visualization_suggestions) > 0)


if __name__ == '__main__':
    unittest.main()
