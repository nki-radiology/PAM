# Prognostic AI-monitoring

## from 
- prognostic-ai-monitoring

## modifications
- split the encoder in two separate encoders with shared weights, f(moving) and f(fixed);
- latent space of the deformation created via subtraction f(moving) - f(fixed).

## experiments
| wandb-name            | change                  | result       |
|-----------------------|-------------------------|--------------|
| pleasent-disco-2      | baseline model          | underfitting |
| dainty-sun-3          | removed Tanh            | no change    |
| demin-deluge-4        | added FC to latent      | lower loss   |
| effortless-glade-6    | added FC to latent      | no change    |
|                       | share affine and deform |              |
| colorful-butterfly-13 | knowledge transfer (KT) | worse result |
|                       | scheduler on KT         |              |
