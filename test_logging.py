#!/usr/bin/env python3
"""Test script to verify logging implementation."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variable for log level
os.environ["LOG_LEVEL"] = "DEBUG"

# Import the logger configuration from main.py
import logging
import structlog

# Configure logging as in main.py
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# Test logging at different levels
print("Testing structured logging...")
logger.debug("test_debug_message", component="test", test_id=1)
logger.info("test_info_message", component="test", test_id=2, status="starting")
logger.warning("test_warning_message", component="test", test_id=3, issue="sample")
logger.error("test_error_message", component="test", test_id=4, error="test_error")

# Test with more complex data
logger.info(
    "test_complex_log",
    user_id="test_user_123",
    sql_preview="SELECT * FROM users LIMIT 10",
    sql_length=100,
    execution_time_ms=45.2,
    success=True,
)

print("\nLogging test completed. Check output above for JSON logs.")
