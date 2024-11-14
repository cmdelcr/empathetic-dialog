from utils import config
from model.common import set_seed, count_parameters

from model.emo_trainer import EmoTrainer



if __name__ == '__main__':
    set_seed()
    
    emo_gan_trainer = EmoTrainer()
    
    if config.test:
        emo_gan_trainer.pre_train(load_for_test=True)
        emo_gan_trainer.test(type_='test')
    else:
        emo_gan_trainer.pre_train()

    