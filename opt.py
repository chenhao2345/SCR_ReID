import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    # default="data/cuhk03-np/labeled",
                    # default="data/cuhk03-np/detected",
                    # default="data/dukemtmc-reid/DukeMTMC-reID",
                    default="data/market1501",
                    # default="data/MSMT17_V2/market_style",
                    help='path of dataset')


parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis', 'inference'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=500,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[300,400],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=16,
                    help='the batch size for test')

parser.add_argument('--save_path',
                    default="SCR",
                    help='path to save trained model')

opt = parser.parse_args()