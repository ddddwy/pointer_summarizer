import os

root_dir = os.path.expanduser("~")
#root_dir = os.path.join(root_dir, "Desktop/NLG")

print_interval = 100
save_model_iter = 1000

train_data_path = os.path.join(root_dir, "cnn-dailymail/quora/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "cnn-dailymail/quora/finished_files/chunked/val_*")
decode_data_path = os.path.join(root_dir, "cnn-dailymail/quora/finished_files/chunked/test_*")
vocab_path = os.path.join(root_dir, "cnn-dailymail/quora/finished_files/vocab")
log_root = os.path.join(root_dir, "pointer_summarizer/log_quora_l2")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps= 20
max_dec_steps= 20
beam_size= 8
min_dec_steps= 5
vocab_size= 5000

lr=0.15
adagrad_init_acc=0.1
L2_loss = 0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 5000000

use_gpu=True

lr_coverage=0.15
