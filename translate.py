import util
import model
import visualize
import random
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--sentence", type=str, default="elle a cinq ans de moins que moi", help="French sentence to translate(No more than 10 words)")
parser.add_argument("--visual", type=bool, default=False, help="Show Attention")
parser.add_argument("--n_iters", type=int, default=750, help="Training iters")
parser.add_argument("--plot_every", type=int, default=100, help="Sample interval to display loss")
parser.add_argument("--learning_rate", type=int, default=0.01, help="Learning rate")
opt = parser.parse_args()
print(opt)

##################################
####   Dataset Information    ####
##################################

input_lang, output_lang, pairs = util.readLangs('eng', 'fra', True)
pairs = util.filterPairs(pairs)

for pair in pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])

##################################
####  	Train Model    		  ####
##################################
hidden_size = 256
encoder1 = model.EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = model.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
plot_losses = []
plot_loss_total = 0  

encoder_optimizer = optim.SGD(encoder1.parameters(), lr=opt.learning_rate)
decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=opt.learning_rate)
training_pairs = [util.tensorsFromPair(random.choice(pairs), input_lang,output_lang)
                      for i in range(opt.n_iters)]
criterion = nn.NLLLoss()

for iter in range(1, opt.n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = model.train(input_tensor, target_tensor, encoder1,
                     attn_decoder1, encoder_optimizer, decoder_optimizer, criterion)
    plot_loss_total += loss

    if iter % opt.plot_every == 0:
        plot_loss_avg = plot_loss_total / opt.plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        print("[Iter: %d] [Loss: %f]" % (iter,plot_loss_avg))

visualize.showPlot(plot_losses)
model.evaluateRandomly(encoder1, attn_decoder1,pairs,input_lang,output_lang)

output_words, attentions = model.evaluate(encoder1, attn_decoder1,opt.sentence,input_lang,output_lang )

print(output_words)
if opt.visual:
    plt.matshow(attentions.numpy())
