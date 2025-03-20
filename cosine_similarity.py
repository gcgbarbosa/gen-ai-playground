from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


def plot_2d(vec1, vec2):
    plt.figure()
    
    # Plot vec1 as an arrow starting from the origin
    plt.arrow(0, 0, vec1[0], vec1[1], head_width=0.2, head_length=0.2, fc='red', ec='red', label='vec1')
    
    # Plot vec2 as an arrow starting from the origin
    plt.arrow(0, 0, vec2[0], vec2[1], head_width=0.2, head_length=0.2, fc='blue', ec='blue', label='vec2')
    
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Plot of Two Vectors')
    plt.legend(['vec1', 'vec2'])
    plt.grid(True)
    plt.show()


# Define 2D vectors
vec1 = np.array([0, 3])
vec2 = np.array([2, 0])

plot_2d(vec1, vec2)

cos_sim = cosine_similarity([vec1], [vec2])
print("Cosine Similarity:", cos_sim[0][0])
