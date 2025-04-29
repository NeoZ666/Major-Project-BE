import matplotlib.pyplot as plt
import numpy as np

def get_weights(question_type=None):
    """Calculate weights based on question type."""
    base_weights = {
        'Content': 0.5,
        'Coherence': 0.25,
        'Vocabulary': 0.15,
        'Grammar': 0.10
    }
    
    if question_type == "analytical":
        return {
            'Content': base_weights['Content'] + 0.1,
            'Coherence': base_weights['Coherence'] + 0.05,
            'Vocabulary': base_weights['Vocabulary'] - 0.1,
            'Grammar': base_weights['Grammar'] - 0.05
        }
    elif question_type == "descriptive":
        return {
            'Content': base_weights['Content'] - 0.1,
            'Vocabulary': base_weights['Vocabulary'] + 0.05,
            'Grammar': base_weights['Grammar'] + 0.05,
            'Coherence': base_weights['Coherence']
        }
    return base_weights

def plot_weight_comparison():
    """Create vertical bar graphs comparing analytical and descriptive weights."""
    analytical_weights = get_weights("analytical")
    descriptive_weights = get_weights("descriptive")
    
    # Set up the plot with larger figure size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 33))
    fig.suptitle("Model's Dynamic Weight Distribution by Question Type", fontsize=24, y=1)
    
    # Set font sizes
    AXIS_LABEL_SIZE = 20
    TICK_LABEL_SIZE = 20
    TITLE_SIZE = 20
    BAR_LABEL_SIZE = 20
    
    # Colors for bars
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    # Plot Analytical weights
    criteria = list(analytical_weights.keys())
    values = list(analytical_weights.values())
    bars1 = ax1.bar(criteria, values, color=colors)
    ax1.set_title('Analytical Questions', fontsize=TITLE_SIZE, pad=20)
    ax1.set_ylim(0, 0.7)
    ax1.set_ylabel('Weight (%)', fontsize=AXIS_LABEL_SIZE)

    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)  # Put grid behind bars
    
    # Configure axis properties for first plot
    ax1.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    # Add percentage labels on analytical bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0%}',
                ha='center', va='bottom',
                fontsize=BAR_LABEL_SIZE)
    
    # Plot Descriptive weights
    values = list(descriptive_weights.values())
    bars2 = ax2.bar(criteria, values, color=colors)
    ax2.set_title('Descriptive Questions', fontsize=TITLE_SIZE, pad=20)
    ax2.set_ylim(0, 0.7)
    ax2.set_ylabel('Weight (%)', fontsize=AXIS_LABEL_SIZE)

    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)  # Put grid behind bars
    
    # Configure axis properties for second plot
    ax2.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    # Add percentage labels on descriptive bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0%}',
                ha='center', va='bottom',
                fontsize=BAR_LABEL_SIZE)
    
    # Adjust layout and display
    plt.xticks(rotation=1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_weight_comparison()