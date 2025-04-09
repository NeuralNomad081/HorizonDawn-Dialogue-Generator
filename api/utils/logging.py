import logging

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def log_request(request):
    """Log API request details."""
    logger.info(f"Request: {request.method} {request.url} - Body: {request.body()}")

def log_response(response):
    """Log API response details."""
    logger.info(f"Response: {response.status_code} - Body: {response.body}")

def log_error(error):
    """Log error details."""
    logger.error(f"Error: {error}")