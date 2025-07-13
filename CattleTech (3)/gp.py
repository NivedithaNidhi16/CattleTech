import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Define the labels for Alzheimer's disease classification
labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

# Confusion matrix values provided by you
confusion_matrix_values = np.array([
    [91, 103, 101, 105],   # True Mild_Demented
    [100, 94, 105, 101],   # True Moderate_Demented
    [94, 105, 99, 102],    # True Non_Demented
    [113, 98, 97, 92],     # True Very_Mild_Demented
])

# Generate the confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_values, display_labels=labels)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap='coolwarm', xticks_rotation=45)
plt.title('Confusion Matrix for Alzheimer\'s Disease Detection')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()

# Save the confusion matrix graph as an image
plt.savefig("alzheimers_confusion_matrix.png", dpi=300)  # Save with 300 dpi for better quality
plt.show()
