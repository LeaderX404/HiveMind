"""
Advanced logging module for HiveMind
Supports file logging, performance metrics, and distributed logging
"""
import logging
import sys
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class PerformanceMetrics:
    """Track performance metrics for logging"""
    
    def __init__(self):
        self.metrics = {
            "inference_times": [],
            "memory_usage": [],
            "token_speeds": [],
            "errors": [],
            "request_counts": {},
        }
        self.start_time = time.time()
    
    def log_inference(self, model_name: str, prompt_length: int, 
                     response_length: int, time_taken: float):
        """Log an inference request"""
        self.metrics["inference_times"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "time_taken": time_taken,
        })
        
        # Calculate tokens per second
        if time_taken > 0:
            tokens_per_sec = response_length / time_taken
            self.metrics["token_speeds"].append(tokens_per_sec)
    
    def log_memory(self, memory_used_mb: float, memory_total_mb: float):
        """Log memory usage"""
        self.metrics["memory_usage"].append({
            "timestamp": datetime.now().isoformat(),
            "used_mb": memory_used_mb,
            "total_mb": memory_total_mb,
            "percentage": (memory_used_mb / memory_total_mb) * 100 if memory_total_mb > 0 else 0,
        })
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log an error"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": context or {},
        }
        self.metrics["errors"].append(error_entry)
    
    def increment_request_count(self, request_type: str):
        """Increment request counter"""
        self.metrics["request_counts"][request_type] = \
            self.metrics["request_counts"].get(request_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics summary"""
        uptime = time.time() - self.start_time
        
        stats = {
            "uptime_seconds": uptime,
            "total_inferences": len(self.metrics["inference_times"]),
            "total_errors": len(self.metrics["errors"]),
            "request_counts": self.metrics["request_counts"],
        }
        
        # Calculate averages
        if self.metrics["inference_times"]:
            avg_time = sum(m["time_taken"] for m in self.metrics["inference_times"]) / \
                      len(self.metrics["inference_times"])
            stats["avg_inference_time"] = avg_time
        
        if self.metrics["token_speeds"]:
            avg_speed = sum(self.metrics["token_speeds"]) / len(self.metrics["token_speeds"])
            stats["avg_tokens_per_sec"] = avg_speed
        
        if self.metrics["memory_usage"]:
            latest_memory = self.metrics["memory_usage"][-1]
            stats["current_memory_usage"] = latest_memory
        
        return stats
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a performance report"""
        stats = self.get_stats()
        
        report = """
=== HiveMind Performance Report ===

Uptime: {:.2f} seconds ({:.2f} minutes)

Inference Statistics:
- Total Inferences: {}
- Average Inference Time: {:.4f} seconds
- Average Tokens/Second: {:.2f}

Memory Usage:
- Current: {:.2f} MB ({:.1f}%)

Request Counts:
{}

Errors: {}
""".format(
            stats["uptime_seconds"],
            stats["uptime_seconds"] / 60,
            stats["total_inferences"],
            stats.get("avg_inference_time", 0),
            stats.get("avg_tokens_per_sec", 0),
            stats.get("current_memory_usage", {}).get("used_mb", 0),
            stats.get("current_memory_usage", {}).get("percentage", 0),
            "\n".join(f"  - {k}: {v}" for k, v in stats["request_counts"].items()),
            stats["total_errors"]
        )
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
        
        return report


# Global metrics instance
_global_metrics = None


def get_metrics() -> PerformanceMetrics:
    """Get or create global performance metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PerformanceMetrics()
    return _global_metrics


class AdvancedLogger:
    """Advanced logger with file and performance tracking"""
    
    def __init__(self, name: str, level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 use_json: bool = False,
                 include_metrics: bool = True):
        """
        Initialize advanced logger
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional file path for logging
            use_json: Use JSON format for logs
            include_metrics: Include performance metrics tracking
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.name = name
        self.log_file = log_file
        self.use_json = use_json
        self.include_metrics = include_metrics
        
        # Create formatter
        if use_json:
            self.formatter = JSONFormatter()
        else:
            self.formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add handlers
        self.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        self.handlers.append(console_handler)
        
        # File handler
        if log_file:
            # Ensure directory exists
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
            self.handlers.append(file_handler)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            # Remove any existing handlers that are the same
            pass
    
    def info(self, msg: str, **kwargs):
        """Log info message with optional metrics"""
        if self.include_metrics:
            metrics = get_metrics()
            if "model" in kwargs:
                metrics.increment_request_count(f"inference:{kwargs.get('model')}")
        
        self.logger.info(msg, **kwargs)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message with metrics"""
        if self.include_metrics:
            metrics = get_metrics()
            metrics.log_error(
                error_type=kwargs.get("error_type", "unknown"),
                error_message=msg,
                context={k: v for k, v in kwargs.items() if k != "error_type"}
            )
        
        self.logger.error(msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(msg, **kwargs)
    
    def log_inference(self, model_name: str, prompt_length: int,
                     response_length: int, time_taken: float):
        """Log inference performance"""
        if self.include_metrics:
            metrics = get_metrics()
            metrics.log_inference(model_name, prompt_length, response_length, time_taken)
        
        self.info(
            f"Inference: model={model_name}, prompt_len={prompt_length}, "
            f"response_len={response_length}, time={time_taken:.4f}s"
        )
    
    def log_memory(self, memory_used_mb: float, memory_total_mb: float):
        """Log memory usage"""
        if self.include_metrics:
            metrics = get_metrics()
            metrics.log_memory(memory_used_mb, memory_total_mb)
        
        self.info(
            f"Memory: used={memory_used_mb:.2f}MB, "
            f"total={memory_total_mb:.2f}MB, "
            f"usage={(memory_used_mb/memory_total_mb)*100:.1f}%"
        )


class JSONFormatter(logging.Formatter):
    """JSON formatter for log records"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            log_data["stack_trace"] = self.formatStack(record.stack_info)
        
        return json.dumps(log_data, ensure_ascii=False)


# Factory function for easy logger creation
def get_logger(name: str, level: int = logging.INFO,
               log_file: Optional[str] = None,
               use_json: bool = False,
               include_metrics: bool = True) -> AdvancedLogger:
    """
    Factory function to get or create a logger
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        use_json: Use JSON format for logs
        include_metrics: Include performance metrics tracking
        
    Returns:
        AdvancedLogger instance
    """
    # Check if logger already exists
    if hasattr(logging, "_hivemind_loggers"):
        if name in logging._hivemind_loggers:
            return logging._hivemind_loggers[name]
    else:
        logging._hivemind_loggers = {}
    
    # Create new logger
    logger = AdvancedLogger(name, level, log_file, use_json, include_metrics)
    logging._hivemind_loggers[name] = logger
    
    return logger


# Configure root logger for HiveMind
def configure_hivemind_logging(log_level: str = "INFO", 
                               log_file: Optional[str] = None,
                               use_json: bool = False):
    """
    Configure logging for the entire HiveMind library
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        use_json: Use JSON format for logs
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure all HiveMind loggers
    for logger_name in ["hivemind", "hivemind.core", "hivemind.net", "hivemind.utils"]:
        logger = get_logger(logger_name, level, log_file, use_json)
        logger.logger.setLevel(level)
    
    # Also configure root hivemind logger
    root_logger = get_logger("hivemind", level, log_file, use_json)
    
    return root_logger
