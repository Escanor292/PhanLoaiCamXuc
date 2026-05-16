import sys
import queue

class GuiLogger:
    """Redirects stdout/stderr to a queue that can be read by the GUI."""
    def __init__(self, log_queue):
        self.log_queue = log_queue
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.log_queue.put(message)

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return False

def setup_gui_logging(log_queue):
    """Setup stdout and stderr redirection."""
    logger = GuiLogger(log_queue)
    sys.stdout = logger
    sys.stderr = logger
    return logger
