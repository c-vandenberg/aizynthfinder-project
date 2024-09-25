# Inference

* Implement Beam Search Decoding During Inference Instead of Greedy Decoding:
#     - Analysis of paper by Liu et al shows that they use a beam search with a beam width of 5 during inference to
#       generate multiple candidate sequences and select the best one based on overall sequence probabilities.
#     - Inference is the process of feeding input data into a trained model and obtaining predictions or outputs.
#     - During inference, the model's parameter's (weight's and biases) are fixed; they are not updated or altered.
#     - The main goal is to evaluate how well the model performs on new data or
#     - Beam search explores multiple possible output sequences, potentially improving the quality of the generated
#       sequences compared to greedy decoding (which selects the highest probability token at each step)
