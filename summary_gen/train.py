from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import shutil

import tensorflow as tf
import torch
import numpy as np
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

import config
from batcher import Batcher
from data import Vocab
from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from eval import Evaluate

use_cuda = config.use_gpu and torch.cuda.is_available()
tf.logging.set_verbosity(tf.logging.INFO)

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(5)
        
        # check the existence of log root file
        if not os.path.exists(config.log_root):
            os.mkdir(config.log_root)
            
        # check the existence of training model file
        self.model_dir = os.path.join(config.log_root, 'train_model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        # check the existence of training log file
        self.train_log = os.path.join(config.log_root, 'train_log')
        if not os.path.exists(self.train_log):
            os.mkdir(self.train_log)
        self.summary_writer = tf.summary.FileWriter(self.train_log)

    def save_model(self, running_avg_loss, iter, mode):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        if mode == 'train':
            save_model_dir = self.model_dir
        else:
            best_model_dir = os.path.join(config.log_root, 'best_model')
            if not os.path.exists(best_model_dir):
                os.mkdir(best_model_dir)
            save_model_dir = best_model_dir
        
        if len(os.listdir(save_model_dir))>0:
            shutil.rmtree(save_model_dir)
            time.sleep(2)
            os.mkdir(save_model_dir)
        model_save_path = os.path.join(save_model_dir, 'model_%d'%(iter))
        torch.save(state, model_save_path)
        return model_save_path 

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()
    

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        min_val_loss = np.inf
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % config.print_interval == 0:
                tf.logging.info('steps %d, seconds for %d batch: %.2f , loss: %f, min_val_loss: %f' % (iter, config.print_interval,
                                                                           time.time() - start, loss, min_val_loss))
                start = time.time()
            if iter % config.model_save_iters == 0:
                self.summary_writer.flush()
                model_save_path = self.save_model(running_avg_loss, iter, mode='train')
                tf.logging.info('Evaluate the model %s at validation set....' % model_save_path)
                evl_model = Evaluate(model_save_path)
                val_avg_loss = evl_model.run_eval()
                if val_avg_loss < min_val_loss:
                    min_val_loss = val_avg_loss
                    best_model_save_path = self.save_model(running_avg_loss, iter, mode='eval')
                    tf.logging.info('Save best model at %s' % best_model_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
