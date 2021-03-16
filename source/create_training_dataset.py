import os
import numpy as np
import time, random
from sampling import *
import argparse
from utils import Option, sen2mat
import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from copy import deepcopy
import time, random
from itertools import product
import numpy as np
import tensorflow as tf
from models import *
import pickle as pkl
from utils import *
import logging


from multiprocessing import Pool


ACTION_REPLACE  = 0
ACTION_INSERT   = 1
ACTION_DELETE   = 2
VALID_ACTIONS_DEFAULT = [ACTION_REPLACE, ACTION_INSERT, ACTION_DELETE]


def run_epoch(sess, model, input, sequence_length, target=None, mode='train'):
    #Runs the model on the given data.
    if mode=='train':
        #train language model
        _,cost = sess.run([model._train_op, model._cost], feed_dict={model._input: input, model._target:target, model._sequence_length:sequence_length})
        return cost
    elif mode=='test':
        #test language model
        cost = sess.run(model._cost, feed_dict={model._input: input, model._target:target, model._sequence_length:sequence_length})
        return cost
    else:
        #use the language model to calculate sentence probability
        output_prob = sess.run(model._output_prob, feed_dict={model._input: input, model._sequence_length:sequence_length})
        return output_prob


def run_epoch_mp(config, input, sequence_length):
    with tf.name_scope("forward_train"):
        with tf.variable_scope("forward", reuse=None):
            m_forward = PTBModel(is_training=True, config = config)
    with tf.name_scope("forward_test"):
        with tf.variable_scope("forward", reuse=True):
            mtest_forward = PTBModel(is_training=False, config = config)
    with tf.name_scope("backward_train"):
        with tf.variable_scope("backward", reuse=None):
            m_backward = PTBModel(is_training=True, config = config)
    with tf.name_scope("backward_test"):
        with tf.variable_scope("backward", reuse=True):
            mtest_backward = PTBModel(is_training=False, config = config)
    
    init = tf.global_variables_initializer()  
    
    # Restore session, prevent GPU from preallocating
    session_config = tf.ConfigProto(use_per_session_threads=True)
    session_config.gpu_options.allow_growth = True
    session = tf.Session(config=session_config)
    session.run(init)

    #use the language model to calculate sentence probability
    output_prob = sess.run(model._output_prob, feed_dict={model._input: input, model._sequence_length:sequence_length})
    return output_prob



def compute_prob_sim_old(session, mtest_forward, input_sentence, input_original, sequence_length, 
        similarity_func, sta_vec, id2sen, emb_word, similaritymodel, option):
    prob_old = run_epoch(session, mtest_forward, input_sentence, sequence_length, mode='use')[0]
    tem = 1
    for j in range(sequence_length[0] - 1):
        tem *= prob_old[j][input_sentence[0][j + 1]]
    tem *= prob_old[j + 1][option.dict_size + 1]
    prob_old_prob = tem
    if similarity_func != None:
        similarity_old = similarity_func(input_sentence, input_original, sta_vec, id2sen, emb_word, option, similaritymodel)[0]
        prob_old_prob *= similarity_old
    else:
        similarity_old = -1
    
    return prob_old_prob, similarity_old


def generate_N_candidates(session, mtest_forward, mtest_backward, input_sentence, sequence_length, ind, option, action, calibrated_set):
    # Separate input into forward/backward portions at the replacement point (ind)
    input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
            cut_from_point(input_sentence, sequence_length, ind, option, mode=action)

    # Use language model to get forward/backward probs for each candidate word in vocab
    prob_forward = run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
    prob_backward = run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
    prob_mul = (prob_forward * prob_backward)

    # Generate the N possible candidate sentences
    input_candidate, sequence_length_candidate = generate_candidate_input_calibrated(input_sentence,\
            sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
        calibrated_set=calibrated_set)

    return input_candidate, sequence_length_candidate


def compute_fluency(session, mtest_forward, input_candidate, sequence_length_candidate, option):
    # Compute fluency scores for each candidate
    prob_candidate_pre = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
    prob_candidate = []

    # For each candidate sentence, compute 
    for i in range(len(input_candidate)):
        tem = 1
        for j in range(sequence_length_candidate[0] - 1):
            tem *= prob_candidate_pre[i][j][input_candidate[i][j + 1]]
        tem *= prob_candidate_pre[i][j + 1][option.dict_size + 1]
        prob_candidate.append(tem)
    prob_candidate = np.array(prob_candidate)

    return prob_candidate


def compute_semantic_preservation_expression_diversity(input_candidate, input_original, sta_vec, id2sen, emb_word, option, similaritymodel, similarity_func):
    # Compute the semantic preservation and expression diversity
    if similarity_func != None:
        return similarity_func(input_candidate, input_original, sta_vec, id2sen, emb_word, option, similaritymodel)
    else:
        return 1

def sample_candidate(prob_candidate):
    # Normalize and sample a candidate from the objective functions of top N candidates
    idx = sample_from_candidate(normalize(prob_candidate))
    return prob_candidate[idx], idx


def acceptance_proposal(candidate_prob, prob_old_prob, sequence_length_candidate, sequence_length_old, temperature):
    V_new = math.log(max(np.power(candidate_prob, 1.0/sequence_length_candidate), 1e-200))
    V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length_old), 1e-200))
    acceptance_prob = min(1, math.exp(min((V_new - V_old)/temperature, 200)))
    return V_new, V_old, acceptance_prob

def is_accepted(acceptance_prob):
    # Return True if the proposal is accepted, False otherwise
    return choose_action([acceptance_prob, 1 - acceptance_prob]) == 0


def word_in_dict(input_candidate_word, option):
    # Check if the candidate word is not a special word in our vocab
    return input_candidate_word < option.dict_size


def add_to_calibrated(word, calibrated_set, option):
    # If candidate word not a special word, add to calibrated set
    if word < option.dict_size:
        calibrated_set.add(word)


def check_to_skip(sequence_length, action, ind, num_steps):
    if action not in VALID_ACTIONS_DEFAULT:
        return True
    # Action is insertion and sequence would be too long
    if action == ACTION_INSERT and sequence_length >= num_steps:
        return True
    # Action is delete and sequence isn't long enough to delete
    elif action == ACTION_DELETE and sequence_length <= 2 or ind == 0:
        return True
    return False


class NodeScorer():

    def __init__(self, input_original, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option):
        self.input_original = input_original
        self.sta_vec = sta_vec
        self.id2sen = id2sen
        self.emb_word = emb_word
        self.session = session
        self.mtest_forward = mtest_forward
        self.mtest_backward = mtest_backward
        self.option = option
        self.similarity_func = similarity_keyword_bleu_tensor if option.mode == 'kw-bleu' else similarity_keyword

    def score(self, input_candidate, sequence_length_candidate):
        similarity_candidate = compute_semantic_preservation_expression_diversity(input_candidate, \
            self.input_original, self.sta_vec, self.id2sen, self.emb_word, self.option, None, self.similarity_func)
        fluency_candidate = compute_fluency(self.session, self.mtest_forward, input_candidate, sequence_length_candidate, self.option)
        sore = similarity_candidate * fluency_candidate
        return sore[0]


class ForwardModel():

    def __init__(self, input_original, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option):
        self.input_original = input_original
        self.sta_vec = sta_vec
        self.id2sen = id2sen
        self.emb_word = emb_word
        self.session = session
        self.mtest_forward = mtest_forward
        self.mtest_backward = mtest_backward
        self.option = option
        self.similarity_func = similarity_keyword_bleu_tensor if option.mode == 'kw-bleu' else similarity_keyword

    # @Each node needs its own calibrated set?
    def forward(self, current_sentence, sequence_length, calibrated_set, action, ind):
        if ind >= sequence_length[0] - 1 or check_to_skip(sequence_length, action, ind, 15):
            return current_sentence, sequence_length, calibrated_set
        # Generate the top N candidates sentences using forward/backward probabilities
        if action == ACTION_REPLACE or action == ACTION_INSERT:  # Insert or add
            input_candidate, sequence_length_candidate = generate_N_candidates(self.session, self.mtest_forward, \
                self.mtest_backward, current_sentence, sequence_length, ind, self.option, action, list(calibrated_set))
        elif action == ACTION_DELETE:   # Delete
            input_candidate, sequence_length_candidate = generate_candidate_input_calibrated(current_sentence,\
                sequence_length, ind, None, self.option.search_size, self.option,\
                mode=action, calibrated_set=list(calibrated_set))

        # Compute fluency scores for each candidate
        fluency_candidates = compute_fluency(self.session, self.mtest_forward, input_candidate, sequence_length_candidate, self.option)
        fluency_candidate = fluency_candidates[0] if action == 2 else fluency_candidates

        # Compute the semantic preservation and expression diversity
        similarity_candidate = compute_semantic_preservation_expression_diversity(input_candidate, \
            self.input_original, self.sta_vec, self.id2sen, self.emb_word, self.option, None, self.similarity_func)

        # Compute scores for candidates
        prob_candidate = fluency_candidates * similarity_candidate

        # Sample candidate from top N candidates
        # If action is delete, we only have one candiate, we just get back the single input candidate
        candidate_prob, candidate_idx = sample_candidate(prob_candidate)
        
        # Otherwise, set current sentence, add removed words to calibrated set and continue
        new_sentence = input_candidate[candidate_idx : candidate_idx+1]
        new_length = deepcopy(sequence_length)
        if action == ACTION_REPLACE and word_in_dict(input_candidate[candidate_idx][ind], self.option):
            add_to_calibrated(current_sentence[0][ind+1], calibrated_set, self.option)
        elif action == ACTION_INSERT and word_in_dict(input_candidate[candidate_idx][ind], self.option):
            new_length += 1
        elif action == ACTION_DELETE:
            add_to_calibrated(current_sentence[0][ind], calibrated_set, self.option)
            new_length -= 1

        return new_sentence, new_length, calibrated_set


def create_dataset(config):
    option = config

    with tf.name_scope("forward_train"):
        with tf.variable_scope("forward", reuse=None):
            m_forward = PTBModel(is_training=True, config = config)
    with tf.name_scope("forward_test"):
        with tf.variable_scope("forward", reuse=True):
            mtest_forward = PTBModel(is_training=False, config = config)
    with tf.name_scope("backward_train"):
        with tf.variable_scope("backward", reuse=None):
            m_backward = PTBModel(is_training=True, config = config)
    with tf.name_scope("backward_test"):
        with tf.variable_scope("backward", reuse=True):
            mtest_backward = PTBModel(is_training=False, config = config)

    init = tf.global_variables_initializer()
  
    dataclass = data.Data(config)       
    
    # Restore session, prevent GPU from preallocating
    session_config = tf.ConfigProto(use_per_session_threads=True)
    session_config.gpu_options.allow_growth = True
    session = tf.Session(config=session_config)
    session.run(init)

    # Python 3 conversion for loading embeddings
    fileobj = open(option.emb_path,'r')
    emb_word, emb_id = pkl.load(StrToBytes(fileobj), encoding='bytes')
    emb_word = {k.decode("utf-8") : v for k, v in emb_word.items()}
    fileobj.close()

    sim = option.sim

    # Get the dateset and keyword vector representation
    # Dataset is list of word ids
    # keyword vectors are boolean flags indicating whether a word is classified as a keyword
    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    

    # Calculate the range to operate on
    idx_start = option.data_start
    idx_end = option.data_end if option.data_end != -1 else use_data.length
    print('Operating in range of [{}, {})'.format(idx_start, idx_end))
    
    # Loop for each sentence 
    data_values = []
    for sen_id in range(idx_start, idx_end):
        logging.info(sen_id)
        sta_vec = sta_vec_list[sen_id]
        input_original, sequence_length_original, _ = use_data(1, sen_id)

        calibrated_set = set([x for x in input_original[0] if x < option.dict_size])
        
        # List all actions
        valid_actions = [(a, i) for a, i in list(product(VALID_ACTIONS_DEFAULT, range(option.num_steps)))]

        scorer = NodeScorer(input_original[0], sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option)
        fm = ForwardModel(input_original[0], sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option)

        values = []
        for action, idx in valid_actions:
            value = 0.0
            for _ in range(5):
                try:
                    # Sample N times to get value estimate avg
                    current_sentence, sequence_length, _ = fm.forward(input_original, sequence_length_original, calibrated_set, action, idx)
                    value += scorer.score(current_sentence, sequence_length)
                except:
                    print(sequence_length_original[0])
                    print(idx)
                    quit()
            value /= 5
            values.append(value)
        data_values.append(values)
            
    # Save
    np.save('data/quoradata/{}'.format(option.save_path), np.array(data_values), allow_pickle=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)

    # data property
    parser.add_argument('--data_path', default='data/quoradata/test.txt', type=str)
    parser.add_argument('--dict_path', default='data/quoradata/dict.pkl', type=str)
    parser.add_argument('--dict_size', default=30000, type=int)
    parser.add_argument('--vocab_size', default=30003, type=int)
    parser.add_argument('--backward', default=False, action="store_true")
    parser.add_argument('--keyword_pos', default=True, action="store_false")
    # model architecture
    parser.add_argument('--num_steps', default=15, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--emb_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=300, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--model', default=0, type=int)
    # optimization
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--clip_norm', default=0.00, type=float)
    parser.add_argument('--no_cuda', default=False, action="store_true")
    parser.add_argument('--local', default=False, action="store_true")
    parser.add_argument('--threshold', default=0.1, type=float)

    # evaluation
    parser.add_argument('--sim', default='word_max', type=str)
    parser.add_argument('--mode', default='sa', type=str)
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--accumulate_step', default=1, type=int)
    parser.add_argument('--backward_path', default=None, type=str)
    parser.add_argument('--forward_path', default=None, type=str)

    # sampling
    parser.add_argument('--use_data_path', default='data/quoradata/test.txt', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--pos_path', default='POS/english-models', type=str)
    parser.add_argument('--emb_path', default='data/quoradata/emb.pkl', type=str)
    parser.add_argument('--max_key', default=3, type=float)
    parser.add_argument('--max_key_rate', default=0.5, type=float)
    parser.add_argument('--rare_since', default=30000, type=int)
    parser.add_argument('--sample_time', default=100, type=int)
    parser.add_argument('--search_size', default=100, type=int)
    parser.add_argument('--action_prob', default=[0.3,0.3,0.3,0.3], type=list)
    parser.add_argument('--just_acc_rate', default=0.0, type=float)
    parser.add_argument('--sim_mode', default='keyword', type=str)
    parser.add_argument('--save_path', default='temp.txt', type=str)
    parser.add_argument('--forward_save_path', default='data/tfmodel/forward.ckpt', type=str)
    parser.add_argument('--backward_save_path', default='data/tfmodel/backward.ckpt', type=str)
    parser.add_argument('--max_grad_norm', default=5, type=float)
    parser.add_argument('--keep_prob', default=1, type=float)
    parser.add_argument('--N_repeat', default=1, type=int)
    parser.add_argument('--C', default=0.03, type=float)
    parser.add_argument('--M_kw', default=8, type=float)
    parser.add_argument('--M_bleu', default=1, type=float)

    # Samples to work on 
    # This lets us run multiple instances on separate parts of the data 
    # for added parallelism
    parser.add_argument('--data_start', default=0, type=int)
    parser.add_argument('--data_end', default=-1, type=int)
    parser.add_argument('--alg', default="sa", type=str)

    d = vars(parser.parse_args())
    option = Option(d)

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename="logs/{}.log".format(option.save_path.split(".")[0]))
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    random.seed(option.seed)
    np.random.seed(option.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu
    config = option

    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)


    create_dataset(option)