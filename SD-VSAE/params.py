class Params:

    def __init__(self):
        self.batch_size = 128
        self.dropout = 0.1
        self.hidden_size = 128
        self.layer_num = 1
        self.latent_num = 128

        self.lr = 1e-3
        self.weight_decay = 0.01
        self.epochs = 200