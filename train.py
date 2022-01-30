import argparse

# from network import RegistrationNetwork
# from network import Loss
# from network import Penalty
#
# from keras.optimizers import Adam

parser = argparse.ArgumentParser(description='Train PAM')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size (can be small thanks to group normalization).')
parser.add_argument('--image_size', default=(192, 192, 160), type=tuple, help='Size of the input image.')
args = parser.parse_args()

raise NotImplementedError(
	'You need to implement a data generator for the training! ' +\
	'More info in data.py'
)

training_generator = None
holdout_generator = None

network = RegistrationNetwork(args.image_size).build('pam')

for smooth, epochs in [[9, 100], [6, 50], [3, 25], [1, 12]]:

    loss = Loss(args.image_size, smooth=smooth)
    penalty = Penalty(args.image_size, args.batch_size)
    network.compile(
        Adam(3e-4),
        loss = [penalty.affine, loss.cc, penalty.elastic, loss.cc],
        loss_weights = [.1, 1., .1, 1.]
    )

    network.fit_generator(training_generator, validation_data=holdout_generator, epochs=epochs)


network.save(r'models/pam_network.h5')