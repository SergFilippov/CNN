import conv_net
import read_data

TRAIN_SIZE = 60000
NUM_EPOCHS = 10

if __name__ == '__main__':
    train_data = read_data.open_data('data/train-images.idx3-ubyte', TRAIN_SIZE)
    train_labels = read_data.open_labels('data/train-labels.idx1-ubyte', TRAIN_SIZE)

    net = conv_net.Net()

    net.train(train_data, train_labels, num_epochs=NUM_EPOCHS)

    print('Complited!')