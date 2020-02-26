import numpy as np
import matplotlib.pyplot as plt

#plt.rc('text', usetex=True)
plt.rc('font', size=15.0, family='serif')
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# Fixing random state for reproducibility
np.random.seed(19680801)

averaged=False
if averaged:
    default = np.load(r'Default/default_max_errors.npy')
    mlp_cond = np.load(r'PreConditional_LSTMs/max_errors.npy')
    pre_cond = np.load(r'PreConditional_LSTMs/max_errors.npy')
    inter_cond = np.load(r'InterConditional_LSTMs/max_errors.npy')
    post_cond = np.load(r'restricted_(Post)Conditional_RNN_models/mean_max_list/mean_max_list_rnn_simpliefied.npy')
    post_cond_na = np.load(r'optimized_(Post)Conditional_RNN_models/mean_max_list/mean_max_list_rnn1_optimized.npy')
    enc_dec = np.load(r'Encoder-Decoder-Models/max_list_not_abs/mean_max_list_encoder_decoder4.npy')
else:
    default = np.load(r'Default/default_max_errors.npy')
    default = np.stack([default for _ in range(5)])
    default = default.reshape((-1))

    mlp_cond = []
    mlp_cond.append(np.load(r'DL_PROJECT_FREIBURG/error_signed_model_0.npy'))
    mlp_cond.append(np.load(r'DL_PROJECT_FREIBURG/error_signed_model_1.npy'))
    mlp_cond.append(np.load(r'DL_PROJECT_FREIBURG/error_signed_model_2.npy'))
    mlp_cond.append(np.load(r'DL_PROJECT_FREIBURG/error_signed_model_3.npy'))
    mlp_cond.append(np.load(r'DL_PROJECT_FREIBURG/error_signed_model_4.npy'))
    mlp_cond = np.array(mlp_cond).reshape((-1))

    pre_cond = np.load(r'PreConditional_LSTMs/all_signed_errors.npy').reshape((-1))
    inter_cond = np.load(r'InterConditional_LSTMs/all_signed_errors.npy').reshape((-1))

    post_cond = []
    post_cond.append(np.load(r'restricted_(Post)Conditional_RNN_models/max_list_not_abs/max_list_rnn0.npy'))
    post_cond.append(np.load(r'restricted_(Post)Conditional_RNN_models/max_list_not_abs/max_list_rnn1.npy'))
    post_cond.append(np.load(r'restricted_(Post)Conditional_RNN_models/max_list_not_abs/max_list_rnn2.npy'))
    post_cond.append(np.load(r'restricted_(Post)Conditional_RNN_models/max_list_not_abs/max_list_rnn3.npy'))
    post_cond.append(np.load(r'restricted_(Post)Conditional_RNN_models/max_list_not_abs/max_list_rnn4.npy'))
    post_cond = np.array(post_cond).reshape((-1))

    post_cond_na = []
    post_cond_na.append(np.load(r'optimized_(Post)Conditional_RNN_models/max_list_not_abs/max_list_not_abs0.npy'))
    post_cond_na.append(np.load(r'optimized_(Post)Conditional_RNN_models/max_list_not_abs/max_list_not_abs1.npy'))
    post_cond_na.append(np.load(r'optimized_(Post)Conditional_RNN_models/max_list_not_abs/max_list_not_abs2.npy'))
    post_cond_na.append(np.load(r'optimized_(Post)Conditional_RNN_models/max_list_not_abs/max_list_not_abs3.npy'))
    post_cond_na.append(np.load(r'optimized_(Post)Conditional_RNN_models/max_list_not_abs/max_list_not_abs4.npy'))
    post_cond_na = np.array(post_cond_na).reshape((-1))

    enc_dec = []
    enc_dec.append(np.load(r'Encoder-Decoder-Models/max_list_not_abs/max_list_encoder_decoder_notabs0.npy'))
    enc_dec.append(np.load(r'Encoder-Decoder-Models/max_list_not_abs/max_list_encoder_decoder_notabs1.npy'))
    enc_dec.append(np.load(r'Encoder-Decoder-Models/max_list_not_abs/max_list_encoder_decoder_notabs2.npy'))
    enc_dec.append(np.load(r'Encoder-Decoder-Models/max_list_not_abs/max_list_encoder_decoder_notabs3.npy'))
    enc_dec.append(np.load(r'Encoder-Decoder-Models/max_list_not_abs/max_list_encoder_decoder_notabs4.npy'))
    enc_dec = np.array(enc_dec).reshape((-1))

# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
data = [default.squeeze(), mlp_cond.squeeze(),pre_cond.squeeze(),inter_cond.squeeze(),
        post_cond.squeeze(),post_cond_na.squeeze(),enc_dec.squeeze()]

# Multiple box plots on one Axes
fig, ax = plt.subplots()
ax.boxplot(data)
#ax.set_xlabel('Model')
ax.set_ylabel('Signed Difference to True Value')
ax.set_xticklabels(np.repeat(['Default','MLP','Pre-Cond. RNN','Inter-Cond. RNN',
                              'Post-Cond. RNN','Post-Cond. RNN (NA opt.)'
                                 ,'Pre-Cond. Enc.-Dec.'], 1),
                    rotation=30, fontsize=13)
ax.set_axisbelow(True)
ax.hlines(0,0,8,linestyles='--')


fig.text(0.75, 0.94,r'Std. dev. of models', color='black', weight='roman',fontsize=12)
fig.text(0.75, 0.91,('Default\t\t\t\t\t'+str(round(np.std(default),2))).expandtabs(), color='black', weight='roman',size='x-small')
fig.text(0.75, 0.88,('MLP\t\t\t\t\t'+str(round(np.std(mlp_cond),2))).expandtabs(), color='black', weight='roman',size='x-small')
fig.text(0.75, 0.85,('Pre-Cond. RNN\t\t\t '+str(round(np.std(pre_cond),2))).expandtabs(),color='black', weight='roman', size='x-small')
fig.text(0.75, 0.82,('Inter-Cond. RNN\t\t\t '+str(round(np.std(inter_cond),2))).expandtabs(), color='black', weight='roman', size='x-small')
fig.text(0.75, 0.79,('Post-Cond. RNN\t\t\t '+str(round(np.std(post_cond),2))).expandtabs(), color='black', weight='roman',size='x-small')
fig.text(0.75, 0.76,('Post-Cond. RNN (NA opt.)   '+str(round(np.std(post_cond_na),2))).expandtabs(), color='black', weight='roman',size='x-small')
fig.text(0.75, 0.73,('Pre-Cond. Enc.-Dec.\t\t'+str(round(np.std(enc_dec),2))).expandtabs(), color='black', weight='roman',size='x-small')

plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.show()