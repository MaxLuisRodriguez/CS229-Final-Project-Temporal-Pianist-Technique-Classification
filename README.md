# CS229-Final-Project-Temporal-Pianist-Technique-Classification
Can an AI feedback model be implemented to train aspiring classical musicians to a level of conservatory-level technique? This project explores just that, from advanced data preprocessing methods to testing different state of the art methods of classification. Is it possible to pick out an advanced pianist from image sequences? Surprisingly, yes! 

# How it works:
1. Generate sequences of images given overhead view of pianist playing. (Python scrypt not included for privacy)
   - Method I used was to upload videos to Azure Blob Storage, extract image frames in chronoligcally order.
3. Extract 21 hand landmark features from each sequence of images using google's mediapipe.
4. Next, train 21 hidden markov models (HMMs) on each professional hand feature sequence.
5. Run HMMs on test data to extract probability of each test sequence occuring and compile one data point for each 20 image sequence.
   - Each data point will contain 21 probabilities - one for each feature sequence across 20 images.
6. Label each probability data point as advanced (1) or novice (2).
7. Finally, run new train and test data through GDA classifier and Adaboost for comparison.
8. Next steps: create neural network and test for shape of data (gaussian distribution or not?) using logisitic regression.
   - Hyperparameters for current classification models will also be further tuned.
