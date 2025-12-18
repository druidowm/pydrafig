from dataclasses import field
from pydrafig import pydraclass, main
import time

@pydraclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-4

@pydraclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    train_path: str = "data/train"
    val_path: str = "data/val"

@pydraclass
class TrainingConfig:
    # Use field(default_factory=...) for nested configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    epochs: int = 10
    seed: int = 42
    use_cuda: bool = True
    
    # Custom finalization logic
    def custom_finalize(self):
        if self.epochs < 1:
            raise ValueError("Epochs must be at least 1")
            
        print(f"DEBUG: Config finalized. Training on {'CUDA' if self.use_cuda else 'CPU'}")

@main(TrainingConfig)
def train(cfg: TrainingConfig):
    """
    Simulate a training run.
    """
    print("\n" + "="*40)
    print(f"Starting Training with seed {cfg.seed}")
    print("="*40)
    print(f"Optimizer: {cfg.optimizer.name} (lr={cfg.optimizer.lr})")
    print(f"Data: {cfg.data.batch_size} batch size, {cfg.data.num_workers} workers")
    print(f"Epochs: {cfg.epochs}")
    print("-" * 40)
    
    # Simulate training loop
    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch {epoch}/{cfg.epochs}: Loss = {0.5 / epoch:.4f}")
        time.sleep(0.1)
        
    print("Training complete!")

if __name__ == "__main__":
    # Try running this with:
    # python simple_train.py
    # python simple_train.py epochs=5 optimizer.lr=0.01
    # python simple_train.py 'data.batch_size=64' 'use_cuda=False'
    # python simple_train.py --show
    train()

