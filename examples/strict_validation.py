from dataclasses import field
from pydrafig import pydraclass
from pydrafig.base_config import InvalidConfigurationError

@pydraclass
class StrictConfig:
    learning_rate: float = 0.01
    batch_size: int = 32

if __name__ == "__main__":
    cfg = StrictConfig()
    
    print("1. Valid assignment:")
    cfg.learning_rate = 0.005
    print(f"   learning_rate = {cfg.learning_rate}")
    
    print("\n2. Invalid assignment (Typo):")
    try:
        # Typo: 'learning_rat' instead of 'learning_rate'
        cfg.learning_rat = 0.005
    except InvalidConfigurationError as e:
        print(f"   Caught error: {e}")
        
    print("\n3. Invalid assignment (Unknown field):")
    try:
        cfg.unknown_field = "value"
    except InvalidConfigurationError as e:
        print(f"   Caught error: {e}")

