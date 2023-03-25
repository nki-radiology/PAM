# Prognostic AI-monitoring

## from 
- prognostic-ai-monitoring

## modifications
- split the encoder in two separate encoders with shared weights, f(moving) and f(fixed);
- latent space of the deformation created via subtraction f(moving) - f(fixed).

## experiments
- baseline: not learning, similarity loss > 0.6 for both affine and elastic
- removing tanh in the latent space: it got a bit faster, learning still stuck
- overparametrize interface latent space-decoder
- make latent space convolutional:
