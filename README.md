# Prognostic AI-monitoring

## from 
- prognostic-ai-monitoring

## modifications
- split the encoder in two separate encoders with shared weights, f(moving) and f(fixed);
- latent space of the deformation created via subtraction f(moving) - f(fixed).

## experiments
- pleasent-disco-2: baseline model, underfitting
- dainty-sun-3: removed Tanh, underfitting
- demin-deluge-4: overparametrized latent, lower loss, slow learning, still underfitting
- comic-firebrand-5: increased latent
