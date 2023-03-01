import utils.fflow as flw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MyLogger(flw.Logger):
    def log(self, server=None):
        if self.output == {}:
            self.output = {
                "meta": server.option,
                "mean_curve": [],
                "var_curve": [],
                "train_losses": [],
                "test_accs": [],
                "test_losses": [],
                "valid_accs": [],
                "client_accs": {},
                "mean_valid_accs": [],
                "drop_rates": [],
                "ac_rates":[],
                "clients_frequency":[],
                "last_clients_accs":[]
            }
            for c in server.clients:
                self.output['drop_rates'].append(c.drop_rate)
                self.output['ac_rates'].append(c.active_rate)
        test_metric, test_loss = server.test()
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        self.output['train_losses'].append(
            1.0 * sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)]) / server.data_vol)
        self.output['valid_accs'].append(valid_metrics)
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        self.output['mean_valid_accs'].append(
            1.0 * sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)]) / server.data_vol)
        self.output['mean_curve'].append(np.mean(valid_metrics))
        self.output['var_curve'].append(np.std(valid_metrics))

        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name] = [self.output['valid_accs'][i][cid] for i in
                                                                    range(len(self.output['valid_accs']))]

        self.output['last_clients_accs'] = []
        for cid in range(server.num_clients):
            self.output['last_clients_accs'].append(self.output['client_accs'][server.clients[cid].name][-1])
        

        print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
        print(self.temp.format("Validating Accuracy:", self.output['mean_valid_accs'][-1]))
        print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
        print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))

        self.output['clients_frequency'] = server.get_clients_frequency()

    def figures(self, task, name):
        acc_sum_30, std_sum_30 = 0.0, 0.0
        for i in range(1,501):
            acc_sum_30 = acc_sum_30 + self.output['mean_curve'][-1 * i]
            std_sum_30 = std_sum_30 + self.output['var_curve'][-1 * i]
        acc_mean = acc_sum_30 / 500
        std_mean = std_sum_30 / 500

        print(acc_mean)
        print(std_mean)

        print(max(self.output['mean_curve']))
        print(self.output['var_curve'][self.output['mean_curve'].index(max(self.output['mean_curve']))])

        x = range(len(self.output['train_losses']))
        y1 = self.output['train_losses']
        y2 = self.output['test_losses']
        plt.figure()
        plt.plot(x,y1,label='train_losses')
        plt.plot(x,y2,label='test_losses')
        plt.legend(loc='best')
        plt.savefig(task + name + 'loss.png')
        plt.show()

        # x = range(self.num_rounds+1)
        y1 = self.output['test_accs']
        y2 = self.output['mean_valid_accs']
        plt.figure()
        plt.plot(x,y1,label='test_accs')
        plt.plot(x,y2,label='valid_accs')
        plt.legend(loc='best')
        plt.savefig(task + name + 'acc.png')
        plt.show()

        # x = range(self.num_rounds+1)
        y1 = self.output['mean_curve']
        y2 = self.output['var_curve']
        plt.figure()
        plt.plot(x,y1,label='mean_c_accs')
        plt.plot(x,y2,label='std_c_accs')
        plt.legend(loc='best')
        plt.savefig(task + name + 'std.png')
        plt.show()

        # print(self.output['clients_frequency'])
        
        # clients_frequency = []
        # for i in range(10):
        #     temp = []
        #     for j in range(10):
        #         temp.append(self.output['clients_frequency'][i*10+j])
        #     clients_frequency.append(temp)
        
        # plt.figure()
        # sns.set_theme()
        # ax = sns.heatmap(clients_frequency)
        # plt.savefig("clients_frequency.png")




logger = MyLogger()

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




