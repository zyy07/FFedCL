from .fedbase import BasicServer, BasicClient

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def iterate(self, t):
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients)
        loss_sum = 0
        
        for i in train_losses:
            loss_sum = loss_sum + i
        
        self.model = self.aggregate(models, p = [1.0 * c_loss/loss_sum for c_loss in train_losses])
        # print(self.client_vols)
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


