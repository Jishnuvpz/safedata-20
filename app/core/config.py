import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Application settings
    app_name: str = "SafeData 2.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Privacy settings
    default_epsilon: float = float(os.getenv("DEFAULT_EPSILON", "1.0"))
    default_delta: float = float(os.getenv("DEFAULT_DELTA", "1e-5"))
    max_epsilon: float = float(os.getenv("MAX_EPSILON", "10.0"))
    min_epsilon: float = float(os.getenv("MIN_EPSILON", "0.1"))
    
    # Synthetic data settings
    synthetic_epochs: int = int(os.getenv("SYNTHETIC_EPOCHS", "300"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "500"))
    
    # File processing settings
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB default
    allowed_extensions: list = [".csv", ".xlsx", ".xls", ".pdf"]
    
    # Security settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database settings
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    
    # Audit settings
    enable_audit: bool = os.getenv("ENABLE_AUDIT", "True").lower() == "true"
    audit_retention_days: int = int(os.getenv("AUDIT_RETENTION_DAYS", "90"))
    
    # Attack simulation settings
    enable_attack_simulation: bool = os.getenv("ENABLE_ATTACK_SIMULATION", "True").lower() == "true"
    attack_simulation_samples: int = int(os.getenv("ATTACK_SIMULATION_SAMPLES", "1000"))
    
    # Optimization settings
    optimization_trials: int = int(os.getenv("OPTIMIZATION_TRIALS", "50"))
    optimization_timeout: int = int(os.getenv("OPTIMIZATION_TIMEOUT", "300"))  # 5 minutes
    
    class Config:
        env_file = ".env"

settings = Settings()
