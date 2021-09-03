# Deep learning projects

This repo contains code and notebooks for several small deep learning projects/examples I've done.

- `resnet`: Simple implementation of a ResNet (i.e. CNN with skip connections to avoid vanishing gradients) applied to MNIST for image classification (~98% test accuracy)
- `autoencoder`: Simple implementation of an Autoencoder. The architecture uses skip connection blocks (`resnet.py`) for encoding down to a 2d latent space, following by a decoder through several dense layers. The encoder is exposed and results visualized.  
