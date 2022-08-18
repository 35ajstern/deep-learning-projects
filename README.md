# Deep learning projects

This repo contains code and notebooks for several small independent deep learning projects/examples I've done to learn tensorflow/keras.

- `resnet`: Simple implementation of a ResNet (i.e. CNN with skip connections to avoid vanishing gradients) applied to fashion_mnist for image classification. I also implement and test against a simple model with no skip connections. ResNet has 87% test accuracy, whereas sequential CNN has only 10% (i.e. no better than random chance).
- `autoencoder`: Simple implementation of an Autoencoder. The architecture uses skip connection blocks (`resnet.py`) for encoding down to a 2d latent space, following by a decoder through several dense layers. The encoder is exposed and results applied `mnist` are visualized.  
- `pyro_amortization`: Amortized variational inference to infer variational parameters in a holdout/test set.
- `pyro_markov`: Decoding a Gaussian hidden Markov model (HMM) using amortized variational inference.
