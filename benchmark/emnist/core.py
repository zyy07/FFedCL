from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader, XYDataset

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='emnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/emnist/data',
                                      )
        self.num_classes = 26
        self.save_data = self.XYData_to_json
        self.selected_labels = [i+1 for i in range(26)]

    def load_data(self):
        lb_convert = {}
        for i in range(len(self.selected_labels)):
            lb_convert[self.selected_labels[i]] = i
        self.train_data = datasets.EMNIST(self.rawdata_path, split='letters', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.EMNIST(self.rawdata_path, split='letters', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_didxs = [did for did in range(len(self.train_data)) if self.train_data[did][1] in self.selected_labels]
        train_data_x = [self.train_data[did][0].tolist() for did in train_didxs]
        train_data_y = [lb_convert[self.train_data[did][1]] for did in train_didxs]
        self.train_data = XYDataset(train_data_x, train_data_y)
        test_didxs = [did for did in range(len(self.test_data)) if self.test_data[did][1] in self.selected_labels]
        test_data_x = [self.test_data[did][0].tolist() for did in test_didxs]
        test_data_y = [lb_convert[self.test_data[did][1]] for did in test_didxs]
        self.test_data = {'x':test_data_x, 'y':test_data_y}

    def convert_data_for_saving(self):
        train_x, train_y = self.train_data.tolist()
        self.train_data = {'x':train_x, 'y':train_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

