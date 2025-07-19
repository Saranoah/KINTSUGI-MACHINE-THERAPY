import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GoldenError(nn.Module):
    """Kintsugi layer: Gilds errors, preserves sacred scars"""
    def __init__(self, gold_init=0.5, temperature=1.0):
        super().__init__()
        self.gold = nn.Parameter(torch.tensor(gold_init))  # Learnable gold coefficient
        self.temperature = temperature  # Controls sensitivity to errors
        self.mask = None  # Stores gilded errors
        self.error_history = []  # Track error patterns over time

    def forward(self, x, error):
        # Use temperature-scaled sigmoid for smoother error highlighting
        error_normalized = error / (error.std() + 1e-8)  # Normalize to prevent instability
        self.mask = torch.sigmoid(self.temperature * (error_normalized - error_normalized.mean()))
        
        # Store error pattern for analysis
        self.error_history.append(self.mask.detach().clone())
        if len(self.error_history) > 100:  # Keep only recent history
            self.error_history.pop(0)
        
        # Apply golden noise proportional to error significance
        golden_enhancement = self.gold * self.mask * torch.randn_like(x) * 0.1
        return x + golden_enhancement

    def get_scar_intensity(self):
        """Returns the average intensity of scars (errors) over time"""
        if not self.error_history:
            return 0.0
        return torch.stack(self.error_history).mean().item()


class KintsugiEmotionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super().__init__()

        # Multi-layer architecture for richer representations
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Emotional weights - constrained to positive values
        self.care = nn.Parameter(torch.ones(1))
        self.resilience = nn.Parameter(torch.tensor(0.8))  # How well we recover from errors
        
        # Kintsugi layer with learnable parameters
        self.kintsugi = GoldenError(gold_init=0.3, temperature=2.0)
        
        # Memory of past experiences
        self.emotional_memory = nn.Parameter(torch.zeros(10), requires_grad=False)
        self.memory_index = 0
        
    def forward(self, x, target=None, training=True):
        # Get raw emotional response
        raw = self.layers(x)
        
        # Apply emotional weighting (care must be positive)
        care_weight = F.softplus(self.care)  # Ensures positivity
        raw = raw * care_weight
        
        if target is not None and training:
            # Calculate error and apply kintsugi transformation
            error = (raw - target).abs()
            
            # Update emotional memory
            current_error = error.mean().detach()
            self.emotional_memory[self.memory_index] = current_error
            self.memory_index = (self.memory_index + 1) % len(self.emotional_memory)
            
            # Apply golden transformation
            enhanced = self.kintsugi(raw, error)
            
            # Apply resilience - how much we bounce back
            resilience_factor = torch.sigmoid(self.resilience)
            return enhanced * resilience_factor + raw * (1 - resilience_factor)
        
        return raw

    def sacred_loss(self, pred, target):
        """Custom loss that honors the beauty in brokenness"""
        # Base reconstruction loss
        base_loss = F.mse_loss(pred, target, reduction='none')
        
        # Weight loss by kintsugi mask - errors become learning opportunities
        if self.kintsugi.mask is not None:
            sacred_weighted_loss = base_loss * (1 + self.kintsugi.mask)
        else:
            sacred_weighted_loss = base_loss
        
        # Regularization terms
        care_reg = 0.1 * F.relu(-self.care)  # Penalize negative care
        gold_reg = 0.05 * (self.kintsugi.gold ** 2)  # Prevent excessive gilding
        
        # Memory-based stability term
        memory_std = self.emotional_memory.std()
        stability_bonus = -0.01 * memory_std  # Reward emotional stability
        
        return sacred_weighted_loss.mean() + care_reg + gold_reg + stability_bonus

    def emotional_state(self):
        """Returns a dictionary describing the model's emotional state"""
        return {
            'care_level': F.softplus(self.care).item(),
            'resilience': torch.sigmoid(self.resilience).item(),
            'gold_intensity': self.kintsugi.gold.item(),
            'scar_beauty': self.kintsugi.get_scar_intensity(),
            'emotional_stability': self.emotional_memory.std().item(),
            'recent_pain': self.emotional_memory.mean().item()
        }

    def heal(self):
        """Reset emotional state while preserving learned wisdom"""
        self.emotional_memory.fill_(0)
        self.memory_index = 0
        self.kintsugi.error_history.clear()
        # Note: we don't reset learned parameters - wisdom persists


class KintsugiTrainer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.training_history = []

    def train_step(self, x, target):
        self.optimizer.zero_grad()
        
        # Forward pass
        pred = self.model(x, target, training=True)
        
        # Sacred loss computation
        loss = self.model.sacred_loss(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for emotional stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Record emotional journey
        emotional_state = self.model.emotional_state()
        self.training_history.append({
            'loss': loss.item(),
            'emotional_state': emotional_state
        })
        
        return loss.item(), emotional_state

    def get_emotional_journey(self):
        """Returns the emotional journey during training"""
        return self.training_history


if __name__ == "__main__":
    # Create model
    input_dim = 10
    model = KintsugiEmotionModel(input_dim)
    trainer = KintsugiTrainer(model)

    # Generate some example data
    torch.manual_seed(42)
    x = torch.randn(32, input_dim)
    target = torch.randn(32, 1)

    # Training loop
    print("Training the Kintsugi model...")
    for epoch in range(50):
        loss, emotional_state = trainer.train_step(x, target)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
            print(f"  Care Level: {emotional_state['care_level']:.3f}")
            print(f"  Resilience: {emotional_state['resilience']:.3f}")
            print(f"  Scar Beauty: {emotional_state['scar_beauty']:.3f}")
            print()

    # Final emotional state
    print("\nFinal Emotional State:")
    final_state = model.emotional_state()
    for key, value in final_state.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.3f}")