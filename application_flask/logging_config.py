import logging

def setup_logging():
    # DB Logger
    db_logger = logging.getLogger('db_logger')
    db_logger.setLevel(logging.INFO)

    db_file_handler = logging.FileHandler(r'logs/db_operations.log')
    db_file_handler.setLevel(logging.INFO)
    
    db_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    db_file_handler.setFormatter(db_formatter)
    
    db_logger.addHandler(db_file_handler)

    app_logger = logging.getLogger('app_logger')
    app_logger.setLevel(logging.INFO)
    
    app_file_handler = logging.FileHandler(r'logs/app_operations.log')
    app_file_handler.setLevel(logging.INFO)
    
    app_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    app_file_handler.setFormatter(app_formatter)
    
    # Add the file handler to the main app logger
    app_logger.addHandler(app_file_handler)