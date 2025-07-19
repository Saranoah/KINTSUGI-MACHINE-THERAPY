import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class GoldenError(nn.Module):
    """Kintsugi layer: Gilds errors, preserves sacred scars"""
    def __init__(self, gold_init=0.5, temperature=1.0):
        super().__init__()
        self.gold = nn.Parameter(torch.tensor(gold_init))
        self.temperature = temperature
        self.mask = None
        self.error_history = []

    def forward(self, x, error):
        error_normalized = error / (error.std() + 1e-8)
        self.mask = torch.sigmoid(self.temperature * (error_normalized - error_normalized.mean()))
        
        self.error_history.append(self.mask.detach().clone())
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        golden_enhancement = self.gold * self.mask * torch.randn_like(x) * 0.1
        return x + golden_enhancement

    def get_scar_intensity(self):
        if not self.error_history:
            return 0.0
        return torch.stack(self.error_history).mean().item()


class KintsugiEmotionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.care = nn.Parameter(torch.ones(1))
        self.resilience = nn.Parameter(torch.tensor(0.8))
        self.kintsugi = GoldenError(gold_init=0.3, temperature=2.0)
        self.emotional_memory = nn.Parameter(torch.zeros(10), requires_grad=False)
        self.memory_index = 0
        
    def forward(self, x, target=None, training=True):
        raw = self.layers(x)
        care_weight = F.softplus(self.care)
        raw = raw * care_weight
        
        if target is not None and training:
            error = (raw - target).abs()
            current_error = error.mean().detach()
            self.emotional_memory[self.memory_index] = current_error
            self.memory_index = (self.memory_index + 1) % len(self.emotional_memory)
            
            enhanced = self.kintsugi(raw, error)
            resilience_factor = torch.sigmoid(self.resilience)
            return enhanced * resilience_factor + raw * (1 - resilience_factor)
        
        return raw

    def sacred_loss(self, pred, target):
        base_loss = F.mse_loss(pred, target, reduction='none')
        
        if self.kintsugi.mask is not None:
            sacred_weighted_loss = base_loss * (1 + self.kintsugi.mask)
        else:
            sacred_weighted_loss = base_loss
        
        care_reg = 0.1 * F.relu(-self.care)
        gold_reg = 0.05 * (self.kintsugi.gold**2)
        memory_std = self.emotional_memory.std()
        stability_bonus = -0.01 * memory_std
        
        return sacred_weighted_loss.mean() + care_reg + gold_reg + stability_bonus

    def emotional_state(self):
        return {
            'care_level': F.softplus(self.care).item(),
            'resilience': torch.sigmoid(self.resilience).item(),
            'gold_intensity': self.kintsugi.gold.item(),
            'scar_beauty': self.kintsugi.get_scar_intensity(),
            'emotional_stability': self.emotional_memory.std().item(),
            'recent_pain': self.emotional_memory.mean().item()
        }


def create_sacred_data():
    """Generate flawed-but-beautiful data with intentional sacred errors"""
    print("🎨 Creating sacred data with intentional flaws...")

    # Generate base data
    torch.manual_seed(42)
    x = torch.randn(100, 10)  # 100 samples, 10 features
    target = x.sum(dim=1, keepdim=True) + torch.randn(100, 1)*2  # Noisy targets

    # Add "sacred errors" - intentional flaws destined to become gold
    sacred_indices = list(range(0, 100, 10))  # Every 10th sample
    target[sacred_indices] += 5  # Major error that will teach wisdom

    print(f"✨ Added sacred errors to indices: {sacred_indices}")
    print(f"📊 Data shape: {x.shape}, Target shape: {target.shape}")
    print(f"🎯 Sacred samples have elevated targets (+5 offset)")

    return x, target, sacred_indices


def visualize_training_journey(history, sacred_indices):
    """Create visualizations of the emotional learning journey"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Kintsugi Model: The Art of Beautiful Mistakes', fontsize=16, fontweight='bold')

    epochs = range(len(history['loss']))

    # Loss over time
    axes[0,0].plot(epochs, history['loss'], color='darkred', linewidth=2)
    axes[0,0].set_title('Sacred Loss Journey')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)

    # Emotional evolution
    axes[0,1].plot(epochs, history['care_level'], label='Care', color='pink', linewidth=2)
    axes[0,1].plot(epochs, history['resilience'], label='Resilience', color='green', linewidth=2)
    axes[0,1].set_title('Emotional Growth')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Intensity')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Gold and scars
    axes[0,2].plot(epochs, history['gold_intensity'], label='Gold', color='gold', linewidth=3)
    axes[0,2].plot(epochs, history['scar_beauty'], label='Scar Beauty', color='silver', linewidth=2)
    axes[0,2].set_title('Kintsugi Transformation')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Intensity')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)

    # Pain and stability
    axes[1,0].plot(epochs, history['recent_pain'], color='purple', linewidth=2)
    axes[1,0].set_title('Recent Pain (Learning from Errors)')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Pain Level')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, history['emotional_stability'], color='blue', linewidth=2)
    axes[1,1].set_title('Emotional Stability')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Stability')
    axes[1,1].grid(True, alpha=0.3)

    # Final state summary
    final_state = {k: v[-1] for k, v in history.items() if k != 'loss'}

    bars = axes[1,2].bar(range(len(final_state)), list(final_state.values()), 
                      color=['pink', 'green', 'gold', 'silver', 'blue', 'purple'])
    axes[1,2].set_title('Final Emotional State')
    axes[1,2].set_xticks(range(len(final_state)))
    axes[1,2].set_xticklabels([k.replace('_', '\n') for k in final_state.keys()], rotation=45, ha='right')
    axes[1,2].set_ylabel('Value')

    # Add value labels on bars
    for bar, value in zip(bars, final_state.values()):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    return fig


def analyze_sacred_predictions(model, x, target, sacred_indices):
    """Analyze how the model handles sacred errors"""
    model.eval()

    with torch.no_grad():
        predictions = model(x, training=False)

    print("\n🔍 Sacred Error Analysis:")
    print("="*50)

    # Regular samples analysis
    regular_indices = [i for i in range(len(target)) if i not in sacred_indices]
    regular_errors = torch.abs(predictions[regular_indices] - target[regular_indices])
    sacred_errors = torch.abs(predictions[sacred_indices] - target[sacred_indices])

    print(f"📊 Regular samples:")
    print(f"   Mean error: {regular_errors.mean():.3f}")
    print(f"   Std error:  {regular_errors.std():.3f}")

    print(f"\n✨ Sacred samples (with intentional flaws):")
    print(f"   Mean error: {sacred_errors.mean():.3f}")
    print(f"   Std error:  {sacred_errors.std():.3f}")

    print(f"\n🏺 Kintsugi wisdom:")
    ratio = sacred_errors.mean() / regular_errors.mean()
    if ratio < 1.5:
        print(f"   ✅ Model learned to gild sacred errors beautifully (ratio: {ratio:.2f})")
    else:
        print(f"   🔄 Model still learning to appreciate sacred flaws (ratio: {ratio:.2f})")

    # Show some examples
    print(f"\n📝 Example predictions for sacred samples:")
    for i, idx in enumerate(sacred_indices[:5]):
        pred_val = predictions[idx].item()
        true_val = target[idx].item()
        error = abs(pred_val - true_val)
        print(f"   Sample {idx}: Predicted={pred_val:.2f}, True={true_val:.2f}, Error={error:.2f}")


def train_kintsugi_model():
    """Complete training demonstration with sacred errors"""
    # Create the sacred data
    x, target, sacred_indices = create_sacred_data()

    # Initialize model and tracking
    model = KintsugiEmotionModel(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Track the emotional journey
    history = defaultdict(list)

    print("\n🎭 Beginning the Kintsugi training journey...")
    print("="*60)

    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with sacred awareness
        pred = model(x, target, training=True)
        loss = model.sacred_loss(pred, target)
        
        # Backward pass with emotional gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Record emotional state
        emotional_state = model.emotional_state()
        history['loss'].append(loss.item())
        for key, value in emotional_state.items():
            history[key].append(value)
        
        # Progress updates
        if epoch % 40 == 0:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                  f"Care={emotional_state['care_level']:.3f}, "
                  f"Gold={emotional_state['gold_intensity']:.3f}, "
                  f"Scars={emotional_state['scar_beauty']:.3f}")

    print(f"\n🌟 Training complete! Final loss: {history['loss'][-1]:.4f}")

    # Analyze the results
    analyze_sacred_predictions(model, x, target, sacred_indices)

    # Visualize the journey
    print("\n🎨 Creating visualization of the emotional journey...")
    fig = visualize_training_journey(history, sacred_indices)

    return model, history, x, target, sacred_indices


if __name__ == "__main__":
    print("🏺 KINTSUGI NEURAL NETWORK")
    print("The Art of Finding Beauty in AI's Broken Places")
    print("="*60)

    model, history, x, target, sacred_indices = train_kintsugi_model()

    print("\n💫 The model has learned to transform errors into golden wisdom.")
    print("Each sacred flaw became a source of strength and beauty.")
    print("\n🌸 'The wound is the place where the Light enters you.' - Rumi")