from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from copy import deepcopy
import time, random
from itertools import product
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.client import device_lib
from models import *
import pickle as pkl
from utils import *
import data, RAKE
from data import array_data
from zpar import ZPar

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


def simulatedAnnealing(config):
    option = config
    with tf.name_scope("forward_train"):
        with tf.variable_scope("forward", reuse=None):
            m_forward = PTBModel(is_training=True, config = config)
    with tf.name_scope("forward_test"):
        with tf.variable_scope("forward", reuse=True):
            mtest_forward = PTBModel(is_training=False, config = config)
    var=tf.trainable_variables()
    var_forward=[x for x in var if x.name.startswith('forward')]
    saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
    with tf.name_scope("backward_train"):
        with tf.variable_scope("backward", reuse=None):
            m_backward = PTBModel(is_training=True, config = config)
    with tf.name_scope("backward_test"):
        with tf.variable_scope("backward", reuse=True):
            mtest_backward = PTBModel(is_training=False, config = config)
    var=tf.trainable_variables()
    var_backward=[x for x in var if x.name.startswith('backward')]
    saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
    init = tf.global_variables_initializer()
  
    dataclass = data.Data(config)       
    
    # Restore session, prevent GPU from preallocating
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session = tf.Session(config=session_config)
    session.run(init)
    saver_forward.restore(session, option.forward_save_path)
    saver_backward.restore(session, option.backward_save_path)

    tfflag = True

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
    generateset = []
    
    temperatures = option.C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    option.temperatures = temperatures

    # Calculate the range to operate on
    idx_start = option.data_start
    idx_end = option.data_end if option.data_end != -1 else use_data.length
    print('Operating in range of [{}, {})'.format(idx_start, idx_end))
    
    # Loop for each sentence 
    count_accepted = 0.0
    for sen_id in range(idx_start, idx_end):
        sta_vec = sta_vec_list[sen_id]
        input, sequence_length, _ = use_data(1, sen_id)
        print('----------------')
        print('Item {} of {}'.format(sen_id, use_data.length))
        print(' '.join(id2sen(input[0])))       # Starting sentence before SA
        print(sta_vec)                          # Binary indicies if word is keyword
        maxV = -30
        # Repeat running SA for N_repeat times (I guess to find maximal result as SA is random)
        for k in range(option.N_repeat):
            sen, V, count = sa(input, sequence_length, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option)
            count_accepted += count
            if maxV<V:
                sampledsen = sen
                maxV = V
            appendtext(sampledsen, os.path.join(option.this_expsdir, option.save_path))
    
    appendtext(str(count_accepted / (idx_end - idx_start)), os.path.join(option.this_expsdir, option.save_path))


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
        if ind >= sequence_length:
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


class TreeNode():
    def __init__(self, sequence_length, max_len, parent=None, action_seq=[]):
        self.sequence_length = sequence_length
        self.valid_actions = [
            (a, i) for a, i in list(product(VALID_ACTIONS_DEFAULT, range(sequence_length))) if not check_to_skip(sequence_length, a, i, max_len)
        ]
        random.shuffle(self.valid_actions)
        self.valid_actions = self.valid_actions[:10]
        self.action_seq = action_seq
        self.parent = parent
        self.isTerminal = False
        self.isFullyExpanded = self.isTerminal
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


EXPLORATION_CONSTANT = np.sqrt(2)
def UCT(node):
    best_value = -1
    best_nodes = []
    denom = np.log(node.numVisits)
    for child in node.children.values():
        node_value = (child.totalReward / child.numVisits) + EXPLORATION_CONSTANT * np.sqrt(denom / child.numVisits)
        if node_value > best_value:
            best_value = node_value
            best_nodes = [child]
        elif node_value == best_value:
            best_nodes.append(child)
    return random.choice(best_nodes)


def rollout_random(valid_actions, fm, current_sentence, sequence_length, calibrated_set):
    action, idx = random.choice(valid_actions)
    if not check_to_skip(sequence_length, action, idx, fm.option.num_steps):
        current_sentence, sequence_length, calibrated_set = fm.forward(current_sentence, sequence_length, calibrated_set, action, idx)
    return current_sentence, sequence_length, calibrated_set


def select_most_visited(node):
    best_count = -1
    best_nodes = []
    for child in node.children.values():
        visit_count = child.numVisits()
        if visit_count > best_count:
            best_count = visit_count
            best_nodes = [child]
        elif visit_count == best_count:
            best_nodes.append(child)
    return random.choice(best_nodes)


def select_best(node):
    best_value = -1
    best_nodes = []
    for child_action, child_node in node.children.items():
        node_value = child_node.totalReward / child_node.numVisits
        if node_value > best_value:
            best_value = node_value
            best_nodes = [child_node]
        elif node_value == best_value:
            best_nodes.append(child_node)
    return random.choice(best_nodes)


class MCTS():

    def __init__(self, selection_policy, rollout_policy, scorer, forward_model, iteration_limit=10, rollout_expected=10, rollout_depth=3):
        self._selection_policy = selection_policy
        self._rollout_policy = rollout_policy
        self._scorer = scorer
        self._fm = forward_model
        self._iteration_limit = iteration_limit
        self._rollout_expected = rollout_expected
        self._rollout_depth = rollout_depth

    def search(self, current_sentence, sequence_length, calibrated_set, root=None):
        self.current_sentence = current_sentence
        self.sequence_length = sequence_length
        self.calibrated_set = calibrated_set
        self.max_depth = 0
        if root is None:
            self.root = TreeNode(sequence_length[0], self._fm.option.num_steps, parent=None, action_seq=[])
        else:
            self.root = root

        # Run MCTS for N interations
        for i in range(self._iteration_limit):
            self.singleRound()

        # Select best child
        best_child = select_best(self.root)
        action, idx = best_child.action_seq[-1]
        print('max_depth {}'.format(self.max_depth))
        return action, idx, best_child

    def singleRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node)
        self.backpropogate(node, reward)


    def selectNode(self, node):
        count = 0
        while not node.isTerminal:
            count += 1
            if count > self.max_depth:
                self.max_depth = count
            if node.isFullyExpanded:
                node = self._selection_policy(node)
            else:
                return self.expand(node)
        return node


    def rollout(self, node):
        total_reward = 0
        for i in range(self._rollout_expected):
            current_sentence = self.current_sentence
            sequence_length = self.sequence_length
            calibrated_set = deepcopy(self.calibrated_set)
            # Bring the state to current position
            for action, idx in node.action_seq:
                current_sentence, sequence_length, calibrated_set = self._fm.forward(current_sentence, sequence_length, calibrated_set, action, idx)
            # Perform random rollout
            for _ in range(self._rollout_depth):
                current_sentence, sequence_length, calibrated_set = self._rollout_policy(node.valid_actions, self._fm, current_sentence, sequence_length, calibrated_set)
            total_reward += self._scorer.score(current_sentence, sequence_length)
        return total_reward / self._rollout_depth

    def expand(self, node):
        for action, idx in node.valid_actions:
            if (action, idx) not in node.children:
                # Apply action to current sentence
                child_node = TreeNode(self.sequence_length[0], self._fm.option.num_steps, node, [(action, idx)])
                node.children[(action, idx)] = child_node
                if len(node.valid_actions) == len(node.children):
                    node.isFullyExpanded = True
                return child_node

        raise Exception("No valid children, shouldn't get here.")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent


def runMCTS(config):
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
    generateset = []
    
    temperatures = option.C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    option.temperatures = temperatures

    # Calculate the range to operate on
    idx_start = option.data_start
    idx_end = option.data_end if option.data_end != -1 else use_data.length
    print('Operating in range of [{}, {})'.format(idx_start, idx_end))
    
    # Loop for each sentence 
    for sen_id in range(idx_start, idx_end):
        sta_vec = sta_vec_list[sen_id]
        input_original, sequence_length, _ = use_data(1, sen_id)
        print('----------------')
        print('Item {} of {}'.format(sen_id, use_data.length))
        print(' '.join(id2sen(input_original[0])))       # Starting sentence before SA
        print(sta_vec)                          # Binary indicies if word is keyword

        calibrated_set = set([x for x in input_original[0] if x < option.dict_size])

        scorer = NodeScorer(input_original[0], sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option)
        fm = ForwardModel(input_original[0], sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option)
        ts = MCTS(UCT, rollout_random, scorer, fm, iteration_limit=40, rollout_expected=20, rollout_depth=3)
        root = None

        current_sentence = deepcopy(input_original)
        for i in range(5):
            print('loop {}'.format(i))
            start = time.time()
            action, idx, root = ts.search(current_sentence, sequence_length, calibrated_set, None)
            current_sentence, sequence_length, calibrated_set = fm.forward(current_sentence, sequence_length, calibrated_set, action, idx)
            end = time.time()
            print('time taken: {}'.format(end - start))
            print('action {} idx {}'.format(action, idx))
            print(' '.join(id2sen(current_sentence[0]))) 
        
        appendtext(' '.join(id2sen(current_sentence[0])), os.path.join(option.this_expsdir, option.save_path))



def sa(current_sentence, sequence_length, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option):
    if option.mode == 'kw-bleu':
        similarity_func = similarity_keyword_bleu_tensor
    else:
        similarity_func = similarity_keyword
    
    similaritymodel = None
    input_original = current_sentence[0]
    sta_vec_original = [x for x in sta_vec]
    calibrated_set = set([x for x in current_sentence[0] if x < option.dict_size])

    count_accepted = 0

    for iteration in range(option.sample_time):
        temperature = option.temperatures[iteration]

        # Roll for index/action
        # In the original implementation, indx was sequentially chosen rather than randomly sampled as suggested by the paper
        ind = iteration % (sequence_length[0] - 1)
        action = choose_action(option.action_prob)

        # Check if chosen action is valid. 
        # If not, then continue
        if check_to_skip(sequence_length, action, ind, option.num_steps):
            continue
        
        # For some reason, the default implementation has 4 actions?
        # Maybe this is so that we have some chance at NOOP?
        assert action in VALID_ACTIONS_DEFAULT

        # Compute the similarity scores of the current sentence to original
        prob_old_prob, similarity_old = compute_prob_sim_old(session, mtest_forward, current_sentence, input_original, \
            sequence_length, similarity_func, sta_vec, id2sen, emb_word, similaritymodel, option)

        # Generate the top N candidates sentences using forward/backward probabilities
        if action == ACTION_REPLACE or action == ACTION_INSERT:  # Insert or add
            input_candidate, sequence_length_candidate = generate_N_candidates(session, mtest_forward, \
                mtest_backward, current_sentence, sequence_length, ind, option, action, list(calibrated_set))
        elif action == ACTION_DELETE:   # Delete
            input_candidate, sequence_length_candidate = generate_candidate_input_calibrated(current_sentence,\
                sequence_length, ind, None, option.search_size, option,\
                mode=action, calibrated_set=list(calibrated_set))

        # Compute fluency scores for each candidate
        fluency_candidates = compute_fluency(session, mtest_forward, input_candidate, sequence_length_candidate, option)
        fluency_candidate = fluency_candidates[0] if action == 2 else fluency_candidates

        # Compute the semantic preservation and expression diversity
        similarity_candidate = compute_semantic_preservation_expression_diversity(input_candidate, \
            input_original, sta_vec, id2sen, emb_word, option, similaritymodel, similarity_func)

        # Compute scores for candidates
        prob_candidate = fluency_candidates * similarity_candidate

        # Sample candidate from top N candidates
        # If action is delete, we only have one candiate, we just get back the single input candidate
        candidate_prob, candidate_idx = sample_candidate(prob_candidate)

        # Find acceptance probability
        V_new, V_old, acceptance_prob = acceptance_proposal(candidate_prob, prob_old_prob, sequence_length_candidate[0], sequence_length, temperature)

        # If we don't accept, then move onto next trial
        if not is_accepted(acceptance_prob):
            continue

        count_accepted += 1

        # Otherwise, set current sentence, add removed words to calibrated set and continue
        if action == ACTION_REPLACE and word_in_dict(input_candidate[candidate_idx][ind], option):
            add_to_calibrated(current_sentence[0][ind+1], calibrated_set, option)
            current_sentence = input_candidate[candidate_idx:candidate_idx+1]
        elif action == ACTION_INSERT and word_in_dict(input_candidate[candidate_idx][ind], option):
            current_sentence = input_candidate[candidate_idx : candidate_idx+1]
            sequence_length += 1
        elif action == ACTION_DELETE:
            add_to_calibrated(current_sentence[0][ind], calibrated_set, option)
            current_sentence = input_candidate[candidate_idx : candidate_idx+1]
            sequence_length -= 1

    return ' '.join(id2sen(current_sentence[0])), V_old, count_accepted
