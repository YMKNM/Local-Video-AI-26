"""
Custom Log Handler for UI

Captures log messages and forwards them to the UI for display.
"""

import logging
import queue
import threading
from datetime import datetime
from typing import Optional, Callable, List
from dataclasses import dataclass, field


@dataclass
class LogEntry:
    """Single log entry"""
    timestamp: str
    level: str
    logger: str
    message: str
    
    def format(self, include_timestamp: bool = True) -> str:
        """Format log entry as string"""
        if include_timestamp:
            return f"[{self.timestamp}] [{self.level}] {self.logger}: {self.message}"
        return f"[{self.level}] {self.logger}: {self.message}"
    
    def to_html(self) -> str:
        """Format log entry as HTML with colors"""
        colors = {
            'DEBUG': '#6c757d',
            'INFO': '#0d6efd',
            'WARNING': '#ffc107',
            'ERROR': '#dc3545',
            'CRITICAL': '#dc3545',
        }
        color = colors.get(self.level, '#000000')
        return f'<span style="color: {color};">[{self.timestamp}] [{self.level}] {self.logger}: {self.message}</span>'


class UILogHandler(logging.Handler):
    """
    Custom logging handler that captures logs for UI display.
    
    Usage:
        handler = UILogHandler()
        logging.getLogger().addHandler(handler)
        
        # Get logs
        logs = handler.get_logs()
        
        # Set callback for real-time updates
        handler.set_callback(on_log)
    """
    
    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        self._entries: List[LogEntry] = []
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[LogEntry], None]] = None
        self._queue = queue.Queue()
        
        # Set default format
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    def emit(self, record: logging.LogRecord):
        """Handle a log record"""
        try:
            entry = LogEntry(
                timestamp=datetime.now().strftime('%H:%M:%S.%f')[:-3],
                level=record.levelname,
                logger=record.name,
                message=self.format_message(record)
            )
            
            with self._lock:
                self._entries.append(entry)
                # Trim if too many entries
                if len(self._entries) > self.max_entries:
                    self._entries = self._entries[-self.max_entries:]
            
            # Put in queue for async processing
            self._queue.put(entry)
            
            # Call callback if set
            if self._callback:
                try:
                    self._callback(entry)
                except Exception:
                    pass  # Don't let callback errors break logging
                    
        except Exception:
            self.handleError(record)
    
    def format_message(self, record: logging.LogRecord) -> str:
        """Format just the message part"""
        return record.getMessage()
    
    def set_callback(self, callback: Callable[[LogEntry], None]):
        """Set callback for real-time log updates"""
        self._callback = callback
    
    def get_logs(self, count: Optional[int] = None, level: Optional[str] = None) -> List[LogEntry]:
        """
        Get log entries.
        
        Args:
            count: Maximum number of entries to return
            level: Filter by log level
            
        Returns:
            List of log entries
        """
        with self._lock:
            entries = self._entries.copy()
        
        if level:
            entries = [e for e in entries if e.level == level.upper()]
        
        if count:
            entries = entries[-count:]
        
        return entries
    
    def get_logs_text(self, count: Optional[int] = None) -> str:
        """Get logs as formatted text"""
        entries = self.get_logs(count)
        return '\n'.join(e.format() for e in entries)
    
    def get_logs_html(self, count: Optional[int] = None) -> str:
        """Get logs as HTML"""
        entries = self.get_logs(count)
        return '<br>'.join(e.to_html() for e in entries)
    
    def clear(self):
        """Clear all log entries"""
        with self._lock:
            self._entries.clear()
    
    def get_queue(self) -> queue.Queue:
        """Get the log queue for async processing"""
        return self._queue


class LogCapture:
    """
    Context manager to capture logs during an operation.
    
    Usage:
        with LogCapture() as capture:
            # ... do something that logs ...
        
        logs = capture.get_logs()
    """
    
    def __init__(self, logger_name: Optional[str] = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self._handler: Optional[UILogHandler] = None
        self._logger: Optional[logging.Logger] = None
        self._old_level: Optional[int] = None
    
    def __enter__(self):
        self._handler = UILogHandler()
        self._handler.setLevel(self.level)
        
        if self.logger_name:
            self._logger = logging.getLogger(self.logger_name)
        else:
            self._logger = logging.getLogger()
        
        self._old_level = self._logger.level
        self._logger.setLevel(self.level)
        self._logger.addHandler(self._handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._logger and self._handler:
            self._logger.removeHandler(self._handler)
            if self._old_level is not None:
                self._logger.setLevel(self._old_level)
    
    def get_logs(self) -> List[LogEntry]:
        """Get captured logs"""
        if self._handler:
            return self._handler.get_logs()
        return []
    
    def get_logs_text(self) -> str:
        """Get captured logs as text"""
        if self._handler:
            return self._handler.get_logs_text()
        return ""
