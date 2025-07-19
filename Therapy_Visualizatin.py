def plot_error_metabolism(epoch_data):
    plt.figure(figsize=(12,6))
    plt.suptitle("Error Metabolic Pathways", fontsize=14)
    
    # Plot error decomposition
    plt.subplot(121)
    plt.imshow(epoch_data['error_fragments'].T, 
              cmap='hot', aspect='auto')
    plt.title("Error Proteolysis")
    plt.ylabel("Amino Acid-like Fragments")
    
    # Plot knowledge synthesis
    plt.subplot(122)
    plt.plot(epoch_data['nutrient_absorption'], 'g-')
    plt.title("Nutrient Absorption")
    plt.xlabel("Training Epoch")
    plt.tight_layout()
    plt.savefig("error_metabolism.png")