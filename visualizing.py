from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the sentences
'''
sentences = ["fill the moka pot with water", 
             "Fill the bottom chamber with water", 
             "Get out two slices of bread",
             "Get bread, peanut butter and jelly", 
             "Spread peanut butter on one slice", 
             "Spread peanut butter and jelly", 
             "Spread jelly", 
             "Eat sandwich", 
             "Join the slices"]
'''

sentences = ["Turn off the phone",  
             "Remove the bottom screws",  
             "Lift up the screen",  
             "Remove the metal plate",  
             "Unclip the battery connector",
             "Pry up the battery",
             "Replace the battery", 
             "Replace the metal plate", 
             "Line up the screen", 
             "Snap the screen into place",
             "Unscrew the two pentalobe screws beside the Lightning jack.", 
             "Use a mini suction cup and place it right above the home button.", 
             "Use a guitar pick to gently rock back and forth until the screen starts lifting.", 
             "Unscrew the battery cover and remove the shield.", 
             "Unplug the existing battery by going under the metal flap with a flat edge.",
             "Remove the adhesive that keeps the battery in place.", 
             "Place the new battery in the chassis and plug it in.", 
             "Place the battery cover back on and screw it in.", 
             "Lock the top edge of the screen in place.", 
             "Screw the bottom screws in place.", 
             "Turn off the iPhone.", 
             "Remove the screws from the bottom of the phone.",
             "Remove the screen from the phone.", 
             "Remove the battery connector.", 
             "Remove the adhesive strips from the old battery.", 
             "Attach the new adhesive strips to the new battery.", 
             "Place the new battery in the phone", 
             "Reconnect the screen to the phone.", 
             "Replace the screws.", 
             "Turn on the phone"]

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Encode the sentences to obtain their embeddings
embeddings = model.encode(sentences)

# Compute cosine similarity between the embeddings of the two sentences
similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])

print("Cosine Similarity between the two sentences:", similarity_score[0][0])

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# Plot the embeddings
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (embeddings_pca[i, 0], embeddings_pca[i, 1]))
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Visualization of Sentence Embeddings using PCA')
plt.show()