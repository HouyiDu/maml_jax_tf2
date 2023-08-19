import sys
import os

import argparse



class App(object):
    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)

        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)
    
    def create_parser(self, name):
        p = argparse.ArgumentParser(prog=name,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='specify aruguments of MAML')
        p.add_argument('--dataset',
                       type=str,
                       choices=['miniimagenet', 'omniglot', 'sinusoid'])
        p.add_argument('--logdir',
                       type=str, 
                       default=None)
        p.add_argument('--baseline',
                       type=str,
                       choices=['None', 'oracle'])
        
        g = p.add_argument_group('training details')
        g.add_argument('--pretrain_iterations',
                       type=int,
                       default=70000)
        g.add_argument('--meta_iterations',
                       type=int,
                       default=0)
        g.add_argument('--update_batch_size',
                        type=int,
                        default=10)
        g.add_argument('--norm',
                        type=str,
                        default='None',
                        choices=['batch_norm', 'batch_norm', 'layer_norm', 'None'])
        
        return p
    
    def main(self, name, opts):
        #suppose we only use sinusoid dataset, now setup all the variables for that

        if opts.dataset == 'sinusoid':
            if opts.train == True:
                test_num_updates = 5


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)