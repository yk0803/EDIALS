# EDIALS

Facial image anonymization is essential to enable privacy-preserving image data sharing. The core challenge lies in removing identity-revealing information without degrading the utility of the images, which is essential for, e.g.,  demographic analysis. However, existing techniques apply uniform pixel-level distortions or synthesize replacements using Generative Adversarial Networks (GANs), which do not retain the meaningful features necessary for downstream tasks. To address this issue, we introduce EDIALS (Explainability-Driven Image Anonymization in Latent Space), which selectively modifies identity-specific latent features identified via explainability techniques. By applying targeted, incremental distortions in the latent space of an adversarial autoencoder, EDIALS effectively anonymizes images while preserving their analytical utility much better than existing techniques. Empirical evaluations on a common dataset show that EDIALS achieves 0.42\% re-identification risk (equivalent to random guessing) while maintaining high utility: 84.66\% $F_1$ for age, 97.91\% for gender, and 82.61\% for race classification. In contrast, DeepPrivacy2 —a state-of-the-art GAN-based approach— results in a re-identification risk as large as 16.27\% and lower utility: 78.32\% $F_1$, 82.58\%, and 73.32\% for age, gender, and race classification, respectively.


# MTF Dataset

We have used the MTF data set which contains 5246 images with a distinct distribution of celebrities' image faces that emerged across different labels.
Get the Dataset The MTF data set can be accessed through the following link

url: https://ieee-dataport.org/documents/multi-task-faces-mtf-dataset

# To Execute the code please follow these steps

# 1. Install the Required packages and libraries
    Provided in the Reqs.txt file

# 2. Train a Face Reconition model (Black-box) (use Train black box.ipynb)
    After you have downloaded the datasets, and have installed the required libraries & packages,
    
    2.1. Open the ipynb file named Train black box.ipynb to train the initially required model
      2.1.1 Change the required paths
      2.1.2. Run the code
      2.1.3 Save the model

# 3. Next run the Train AAE.ipynb file to train the AEE
    3.1. Open the ipynb file named Train AAE.ipynb to train the AAE
      3.1.1 Change the required paths
      3.1.2. Run the code
      3.1.3 Save the models (combined)

# 4. Next run the Train EDIALS.ipynb file to anonymize the images
    4.1. Open the ipynb file named EDIALS.ipynb to anonymize the images
      4.1.1 Change the required paths
      4.1.2. Run the code
      4.1.3. Choose the batch size carefully (based on your system's specs)


