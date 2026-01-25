"""
Configuration Module
====================
Centralized configuration and environment variable loading.
Follows best practices for secret management.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    
    # Look for .env in project root
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        # Try parent directory
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


@dataclass
class APIConfig:
    """API configuration loaded from environment variables."""
    
    # Gemini
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    
    # Rate limiting
    api_delay: float = 0.5  # seconds between requests
    
    def __post_init__(self):
        """Load from environment variables."""
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_model = os.environ.get("GEMINI_MODEL", self.gemini_model)
        self.openai_model = os.environ.get("OPENAI_MODEL", self.openai_model)
        self.api_delay = float(os.environ.get("API_DELAY", self.api_delay))
    
    def get_api_key(self, provider: str = "gemini") -> Optional[str]:
        """Get API key for specified provider."""
        if provider.lower() == "gemini":
            return self.google_api_key
        elif provider.lower() == "openai":
            return self.openai_api_key
        return None
    
    def validate(self, provider: str = "gemini") -> bool:
        """Check if API key is configured for provider."""
        key = self.get_api_key(provider)
        if not key:
            print(f"WARNING: {provider.upper()} API key not found.")
            print(f"Set {'GOOGLE_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY'} in .env file")
            return False
        return True


@dataclass  
class PathConfig:
    """Path configuration for data and outputs."""
    
    # Project root
    project_root: Path = None
    
    # Data paths
    data_dir: Path = None
    coco_dir: Path = None
    market1501_dir: Path = None
    
    # Output paths
    output_dir: Path = None
    checkpoints_dir: Path = None
    
    def __post_init__(self):
        """Set up paths relative to project root."""
        if self.project_root is None:
            self.project_root = Path(__file__).parent.parent
        
        self.data_dir = self.project_root / "person_vlm" / "data"
        self.coco_dir = self.project_root / "COCO"
        self.market1501_dir = self.project_root / "Market-1501-v15.09.15"
        self.output_dir = self.project_root / "person_vlm" / "outputs"
        self.checkpoints_dir = self.project_root / "person_vlm" / "checkpoints"
    
    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.output_dir, self.checkpoints_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.api = APIConfig()
        self.paths = PathConfig()
        self._initialized = True
    
    @classmethod
    def get(cls) -> "Config":
        """Get config instance."""
        return cls()


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config.get()


def get_api_key(provider: str = "gemini") -> Optional[str]:
    """Convenience function to get API key."""
    return get_config().api.get_api_key(provider)


def validate_environment(provider: str = "gemini") -> bool:
    """Validate that required environment variables are set."""
    config = get_config()
    return config.api.validate(provider)


if __name__ == "__main__":
    # Test configuration
    print("Configuration Test")
    print("=" * 50)
    
    config = get_config()
    
    print(f"\nAPI Configuration:")
    print(f"  Gemini API Key: {'Set' if config.api.google_api_key else 'NOT SET'}")
    print(f"  OpenAI API Key: {'Set' if config.api.openai_api_key else 'NOT SET'}")
    print(f"  Gemini Model: {config.api.gemini_model}")
    print(f"  API Delay: {config.api.api_delay}s")
    
    print(f"\nPath Configuration:")
    print(f"  Project Root: {config.paths.project_root}")
    print(f"  COCO Dir: {config.paths.coco_dir}")
    print(f"  Market-1501 Dir: {config.paths.market1501_dir}")
    print(f"  Output Dir: {config.paths.output_dir}")
    
    print(f"\nValidation:")
    validate_environment("gemini")
