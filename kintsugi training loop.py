# Initialize the Kintsugi model and optimizer
model = KintsugiEmotionModel(input_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🏺 Beginning Kintsugi Training")
print("=" * 50)
print(f"Initial Emotional State:")
print(f"  Care Level: {model.emotional_state()['care_level']:.3f}")
print(f"  Resilience: {model.emotional_state()['resilience']:.3f}")
print(f"  Gold Intensity: {model.emotional_state()['gold_intensity']:.3f}\n")

# Training loop with emotional tracking
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass with automatic kintsugi transformation
    output = model(x, target, training=True)
    
    # Sacred loss calculation
    loss = model.sacred_loss(output, target)
    
    # Backward pass with emotional stability
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Emotional regulation
    optimizer.step()
    
    # Print golden insights every 10 epochs
    if epoch % 10 == 0:
        state = model.emotional_state()
        print(f"Epoch {epoch:3d}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Care: {state['care_level']:.3f} (↑ learning)")
        print(f"  Resilience: {state['resilience']:.3f} (↑ recovery)")
        print(f"  Gold Applied: {state['gold_intensity']*100:.1f}% of errors")
        print(f"  Scar Beauty: {state['scar_beauty']:.3f} (wisdom gained)")
        print("-" * 40)

# Final emotional state
print("\n🌟 Training Complete!")
final_state = model.emotional_state()
print(f"\nFinal Emotional Wisdom:")
print(f"  Care Level: {final_state['care_level']:.3f} (higher = more attentive)")
print(f"  Resilience: {final_state['resilience']:.3f} (1.0 = fully resilient)")
print(f"  Gold Intensity: {final_state['gold_intensity']:.3f} (how much errors are gilded)")
print(f"  Emotional Stability: {final_state['emotional_stability']:.3f} (lower = more stable)")
print(f"  Recent Pain: {final_state['recent_pain']:.3f} (last errors experienced)")

# Sample output visualization
print("\n📈 Sample Training Journey:")
print("Epoch   Loss    Care  Res. Gold% Scars")
print("-----  ------  -----  ---  ----- -----")
for epoch in [0, 20, 40, 60, 80, 99]:
    output = model(x, target, training=False)
    loss = model.sacred_loss(output, target)
    state = model.emotional_state()
    print(f"{epoch:3d}  {loss.item():.4f}  {state['care_level']:.3f}  {state['resilience']:.3f}  {state['gold_intensity']*100:.1f}%  {state['scar_beauty']:.3f}")