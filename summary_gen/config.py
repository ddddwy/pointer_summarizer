import os

root_dir = os.path.expanduser("~")
#root_dir = os.path.join(root_dir, "Desktop/NLG")

print_interval = 100
model_save_iters = 1000
eval_teacher_forcing = False

train_data_path = os.path.join(root_dir, "cnn-dailymail/cnn/finished_lines/chunked/train_*")
eval_data_path = os.path.join(root_dir, "cnn-dailymail/cnn/finished_lines/chunked/val_*")
decode_data_path = os.path.join(root_dir, "cnn-dailymail/cnn/finished_lines/chunked/test_*")
vocab_path = os.path.join(root_dir, "cnn-dailymail/cnn/finished_lines/vocab")
log_root = os.path.join(root_dir, "pointer_summarizer/log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps=400
max_dec_steps=200
beam_size=4
min_dec_steps=5
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 1000000

use_gpu=True

lr_coverage=0.15
