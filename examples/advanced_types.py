from dataclasses import field
from pydrafig import pydraclass, main
import math

@pydraclass
class MathConfig:
    # We can use expressions in defaults if we wanted, but here we show CLI usage
    hidden_size: int = 256
    dropout: float = 0.1
    activation: str = "relu"
    
    # Complex types
    layers: list[int] = field(default_factory=lambda: [256, 128, 64])
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Computed property
    derived_val: float = 0.0
    
    def custom_finalize(self):
        # We can perform calculations based on other fields
        self.derived_val = self.hidden_size * self.dropout

@main(MathConfig)
def run_experiment(cfg: MathConfig):
    print(f"Hidden Size: {cfg.hidden_size}")
    print(f"Layers: {cfg.layers}")
    print(f"Metrics: {cfg.metrics}")
    print(f"Derived Value: {cfg.derived_val}")

if __name__ == "__main__":
    print("Run with various expressions:")
    print("  python advanced_types.py 'hidden_size=2**10'")
    print("  python advanced_types.py 'layers=[1024, 512, 256]'")
    print("  python advanced_types.py 'metrics={\"acc\": 0.95, \"loss\": 0.1}'")
    print("  python advanced_types.py 'dropout=math.sqrt(0.01)'")
    print("-" * 40)
    
    run_experiment()

