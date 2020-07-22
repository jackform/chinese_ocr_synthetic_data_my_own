class crnn_parser(object):

    def __init__(self):
        self.train_list = './crnnsyntext/image_text_index.txt'
        self.eval_list = ''
        self.num_workers = 4
        self.batch_size = 16
        self.img_height = 32
        self.img_width = 280
        self.hidden_size = 256
        self.num_epochs = 100
        self.learning_rate = 0.0001
        self.encoder = ''
        self.decoder = ''
        self.model = './model/'
        self.random_sample = True
        self.teaching_forcing_prob = 0.5
        self.max_width = 71
