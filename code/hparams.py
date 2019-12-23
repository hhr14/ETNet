import argparse


def get_hparams():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    parser = argparse.ArgumentParser(description="Speech2Face hparams")

    parser.add_argument('--gpu', type=str)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--attnloss', type=_str_to_bool, default=False)
    parser.add_argument('--EBLoss', type=_str_to_bool, default=False)
    parser.add_argument('--FwhLoss', type=_str_to_bool, default=False)
    parser.add_argument('--CNNencoder', type=_str_to_bool, default=False)
    parser.add_argument('--plotattn', type=_str_to_bool, default=False)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--predict_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--check_point_distance', type=int, default=5)
    parser.add_argument('--output_folder', type=str, default='../predict_result/fwh_npy')
    parser.add_argument('--predict', type=str, default='../sample/sample_ppg/',
                        help='if mode is predict, this parameter represents the folder or file of ppg to be predicted')
    parser.add_argument('--attention_path', type=str, default='../predict_result/attention/')

    parser.add_argument('--content_size', type=int, default=219)
    parser.add_argument('--refer_size', type=int, default=219)
    parser.add_argument('--output_size', type=int, default=81)
    #  MLPG

    parser.add_argument('--encoder_mode', type=str, default='GST')
    parser.add_argument('--attention_mode', type=str, default='multihead')
    parser.add_argument('--emotion_add_mode', type=str, default='cat')

    parser.add_argument('--window_size', type=int, default=50)
    #  in open source code window size is 50.
    parser.add_argument('--train_step', type=int, default=240)
    parser.add_argument('--validate_step', type=int, default=30)
    #  step equals to data_size / batch_size
    parser.add_argument('--ContentDenseList', type=list, default=[256])
    #  use mfcc80
    parser.add_argument('--RefDenseList', type=list, default=[128, 256])
    parser.add_argument('--tokenNum', type=int, default=4)
    parser.add_argument('--tokenSize', type=int, default=256)
    # if concat, decoderinput = contentdense[-1] + refdense[-1]
    parser.add_argument('--decoderInputSize', type=int, default=512)
    parser.add_argument('--decoderRNNnum', type=int, default=3)
    parser.add_argument('--decoderRNNunits', type=int, default=256)
    parser.add_argument('--DecoderDenseList', type=list, default=[256, 128, 81])
    parser.add_argument('--dprate', type=float, default=0.1)

    #  For Emotion classifier
    parser.add_argument('--classify_gru_layers', type=int, default=1)
    parser.add_argument('--classify_gru_size', type=int, default=128)
    parser.add_argument('--classify_dense_list', type=list, default=[64, 32, 16, 4])

    #  For Tacotron Decoder.
    parser.add_argument('--num_units', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ref_enc_filters', type=list, default=[32, 32, 64, 64, 128, 128])
    parser.add_argument('--ref_enc_kernel_size', type=list, default=[3, 3])
    parser.add_argument('--ref_enc_strides', type=list, default=[1, 2])
    parser.add_argument('--ref_enc_pad', type=list, default=[1, 0])
    parser.add_argument('--ref_enc_gru_size', type=int, default=128)
    parser.add_argument('--ref_enc_gru_layers', type=int, default=2)

    return parser.parse_args()
