# Prognostic AI-monitoring

## from 
- prognostic-ai-monitoring

## modifications
- split the encoder in two separate encoders with shared weights, f(moving) and f(fixed);
- latent space of the deformation created via subtraction f(moving) - f(fixed).

## experiments
- pleasent-disco-2: baseline model >> underfitting
- dainty-sun-3: removed Tanh >> no change
- demin-deluge-4: overparametrized latent >> lower loss, still underfitting
- effortless-glade-6: increased latent >> no change
- share split encoders in affine network, larger network >> 
- colorful-butterfly-13: twin network with standard setting >> no change, underftting
- scheduler of the similarity between encoders latent space >> 
- only convolutional latent >>
