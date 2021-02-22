from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, math
from copy import copy
import time, random
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import argparse
from tensorflow.python.client import device_lib
import pickle as pkl
from utils import *
from models import *
import data, RAKE
from data import array_data
from zpar import ZPar

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

# def simulatedAnnealing_std(config):
#     option = config
#     with tf.name_scope("forward_train"):
#         with tf.variable_scope("forward", reuse=None):
#             m_forward = PTBModel(is_training=True, config = config)
#     with tf.name_scope("forward_test"):
#         with tf.variable_scope("forward", reuse=True):
#             mtest_forward = PTBModel(is_training=False, config = config)
#     var=tf.trainable_variables()
#     var_forward=[x for x in var if x.name.startswith('forward')]
#     saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
#     with tf.name_scope("backward_train"):
#         with tf.variable_scope("backward", reuse=None):
#             m_backward = PTBModel(is_training=True, config = config)
#     with tf.name_scope("backward_test"):
#         with tf.variable_scope("backward", reuse=True):
#             mtest_backward = PTBModel(is_training=False, config = config)
#     var=tf.trainable_variables()
#     var_backward=[x for x in var if x.name.startswith('backward')]
#     saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
#     init = tf.global_variables_initializer()
  
#     dataclass = data.Data(config)       

#     session = tf.Session()
#     session.run(init)
#     saver_forward.restore(session, option.forward_save_path)
#     saver_backward.restore(session, option.backward_save_path)

#     if option.mode == 'kw-bleu':
#         similarity = similarity_keyword_bleu
#     else:
#         similarity = similarity_keyword
#     similaritymodel = None

#     tfflag = True

#     fileobj = open(option.emb_path,'r')
#     # Python 3 conversion
#     # --------------------------
#     #emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='latin1')
#     emb_word,emb_id=pkl.load(StrToBytes(fileobj), encoding='bytes')
#     emb_word = {k.decode("utf-8") : v for k, v in emb_word.items()}
#     # --------------------------
#     fileobj.close()
#     sim=option.sim
#     sta_vec=list(np.zeros([option.num_steps-1]))

#     use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
#     id2sen = dataclass.id2sen
#     generateset = []
#     C = 0.05
#     temperatures =  C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
#     print(temperatures)
    
#     for sen_id in range(use_data.length):
#         sta_vec=sta_vec_list[sen_id]
#         input, sequence_length, _=use_data(1, sen_id)
#         input_original=input[0]
#         sta_vec_original = [x for x in sta_vec]
#         # for i in range(1,option.num_steps):
#         #   if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
#         #     sta_vec[i-1]=1
#         pos=0
#         print('----------------')
#         print(' '.join(id2sen(input[0])))
#         print(sta_vec)
#         calibrated_set = [x for x in input[0]]
#         for iter in range(option.sample_time):
#             temperature = temperatures[iter]
#             ind=pos%(sequence_length[0]-1)
#             action=choose_action(option.action_prob)
#             calibrated_set = list(set(calibrated_set))
#             if action==0: # word replacement (action: 0)
#                 if tfflag:
#                     prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
#                             mode='use')[0]
#                 else:
#                     prob_old= output_p(input, forwardmodel) #15,K
#                 tem=1
#                 for j in range(sequence_length[0]-1):
#                     tem*=prob_old[j][input[0][j+1]]
#                 tem*=prob_old[j+1][option.dict_size+1]
#                 prob_old_prob=tem
#                 if sim!=None:
#                     similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
#                           option, similaritymodel)[0]
#                     prob_old_prob*=similarity_old
#                 else:
#                     similarity_old=-1

#                 input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
#                         cut_from_point(input, sequence_length, ind, option, mode=action)
#                 if tfflag:
#                     prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
#                     prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
#                 else:
#                     prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
#                     prob_backward = output_p(input_backward,backwardmodel)[
#                         sequence_length[0]-1-ind%(sequence_length[0]-1),:]
#                 prob_mul=(prob_forward*prob_backward)

#                 input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
#                         sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
#                          calibrated_set=calibrated_set)

#                 prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
#                             sequence_length_candidate,mode='use')
#                 prob_candidate=[]
#                 for i in range(len(input_candidate)):
#                   tem=1
#                   for j in range(sequence_length[0]-1):
#                     tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
#                   tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
#                   prob_candidate.append(tem)
          
#                 prob_candidate=np.array(prob_candidate)
#                 if sim!=None:
#                     similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
#                             id2sen, emb_word, option, similaritymodel)
#                     prob_candidate=prob_candidate*similarity_candidate

#                 prob_candidate_norm=normalize(prob_candidate)
#                 prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                
#                 prob_candidate_prob=prob_candidate[prob_candidate_ind]
                
#                 V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length),1e-200))
#                 V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
#                 alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
                
#                 if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
#                     input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
#                     if np.sum(input1[0])==np.sum(input[0]):
#                         pass
#                     else:
#                         calibrated_set.append(input[0][ind+1])
#                         input= input1
#                         print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
#                                         V_new,alphat,similarity_old,similarity_candidate[prob_candidate_ind])
#                         print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

#             elif action==1: # word insert
#                 if sequence_length[0]>=option.num_steps:
#                     pos += 1
#                     continue
#                     # break

#                 input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
#                         cut_from_point(input, sequence_length, ind, option, mode=action)

#                 if tfflag:
#                     prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
#                     prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
#                 else:
#                     prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
#                     prob_backward = output_p(input_backward,backwardmodel)[
#                         sequence_length[0]-1-ind%(sequence_length[0]-1),:]

#                 prob_mul=(prob_forward*prob_backward)

#                 input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
#                         sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
#                     calibrated_set=calibrated_set)

#                 if tfflag:
#                     prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate,\
#                             sequence_length_candidate,mode='use')
#                 else:
#                     prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
#                 prob_candidate=[]
#                 #for i in range(option.search_size):
#                 for i in range(len(input_candidate)):
#                     tem=1
#                     for j in range(sequence_length_candidate[0]-1):
#                         tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
#                     tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
#                     prob_candidate.append(tem)
#                 prob_candidate=np.array(prob_candidate)
#                 if sim!=None:
#                     similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
#                             id2sen, emb_word, option, similaritymodel)
#                     prob_candidate=prob_candidate*similarity_candidate
#                 prob_candidate_norm=normalize(prob_candidate)
#                 prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
#                 prob_candidate_prob=prob_candidate[prob_candidate_ind]
#                 similarity_new = similarity_candidate[prob_candidate_ind]

#                 if tfflag:
#                     prob_old=run_epoch(session, mtest_forward, input,\
#                             sequence_length,mode='use')[0]
#                 else:
#                     prob_old = output_p(input, forwardmodel) # 100,15,300003

#                 tem=1
#                 for j in range(sequence_length[0]-1):
#                     tem*=prob_old[j][input[0][j+1]]
#                 tem*=prob_old[j+1][option.dict_size+1]
#                 prob_old_prob=tem
#                 if sim!=None:
#                     similarity_old=similarity(input, input_original,sta_vec,\
#                             id2sen, emb_word, option, similaritymodel)[0]
#                     prob_old_prob=prob_old_prob*similarity_old
#                 else:
#                     similarity_old=-1
#                 V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
#                 V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

#                 alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
#                 if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
#                     input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
#                     sequence_length+=1

#                     pos+=1
#                     # sta_vec.insert(ind, 0.0)
#                     # del(sta_vec[-1])
#                     print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action,prob_old_prob,V_old,\
#                             V_new,alphat,similarity_old,similarity_new)

#                     print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


#             elif action==2: # word delete
#                 if sequence_length[0]<=2 or ind==0:
#                     pos += 1
#                     continue
#                 if tfflag:
#                     prob_old=run_epoch(session, mtest_forward, input, sequence_length,\
#                             mode='use')[0]
#                 else:
#                     prob_old= output_p(input, forwardmodel) #15,K
#                 tem=1
#                 for j in range(sequence_length[0]-1):
#                     tem*=prob_old[j][input[0][j+1]]
#                 tem*=prob_old[j+1][option.dict_size+1]
#                 prob_old_prob=tem
#                 if sim!=None:
#                     similarity_old=similarity(input, input_original,sta_vec,\
#                             id2sen, emb_word, option, similaritymodel)[0]
#                     prob_old_prob=prob_old_prob*similarity_old
#                 else:
#                     similarity_old=-1

#                 input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(input,\
#                         sequence_length, ind, None, option.search_size, option,\
#                         mode=action,calibrated_set=calibrated_set)

#                 # delete sentence
#                 if tfflag:
#                     prob_new=run_epoch(session, mtest_forward, input_candidate,\
#                             sequence_length_candidate,mode='use')[0]
#                 else:
#                     prob_new = output_p(input_candidate, forwardmodel)


#                 tem=1
#                 for j in range(sequence_length_candidate[0]-1):
#                     tem*=prob_new[j][input_candidate[0][j+1]]
#                 tem*=prob_new[j+1][option.dict_size+1]
#                 prob_new_prob=tem
#                 if sim!=None:
#                     similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
#                             id2sen, emb_word, option, similaritymodel)[0]
#                     prob_new_prob=prob_new_prob*similarity_candidate
                
#                 #alpha is acceptance ratio of current proposal
#                 if input[0] in input_candidate:
#                     for candidate_ind in range(len(input_candidate)):
#                         if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
#                             break
#                         pass
#                     V_new = math.log(max(np.power(prob_new_prob,1.0/sequence_length_candidate[0]),1e-200))
#                     V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))

#                     alphat = min(1,math.exp((V_new-V_old)/temperature))
#                 else:
#                     alphat=0
             
#                 if choose_action([alphat, 1-alphat])==0:
#                     calibrated_set.append(input[0][ind])
#                     input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
#                     sequence_length-=1
#                     # del(sta_vec[ind])
#                     # sta_vec.append(0)
#                     pos -= 1

#                     print('ind, action,oldprob,vold, vnew, alpha,simold, simnew',ind, action,prob_old_prob,V_old,\
#                                 V_new,alphat,similarity_old,similarity_candidate)
#                     print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))


#             pos += 1
#         generateset.append(id2sen(input[0]))
#         appendtext(id2sen(input[0]), option.save_path)

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

    # # --- 
    # input, sequence_length, _ = use_data(4, 1)
    # for i in input:
    #     print(i)
    # for i in sequence_length:
    #     print(i)
    # quit()
    # # ----

    id2sen = dataclass.id2sen
    generateset = []
    
    temperatures = option.C*(1.0/100)*np.array(list(range(option.sample_time+1,1,-1)))
    print(temperatures)
    option.temperatures = temperatures
    
    # Loop for each sentence 
    for sen_id in range(use_data.length):
        sta_vec = sta_vec_list[sen_id]
        input, sequence_length, _ = use_data(1, sen_id)
        print('----------------')
        print('Item {} of {}'.format(sen_id, use_data.length))
        print(' '.join(id2sen(input[0])))       # @note Starting sentence before SA
        print(sta_vec)      # @note Binary 
        maxV = -30
        # @note Repeat running SA for N_repeat times (I guess to find maximal result as SA is random)
        for k in range(option.N_repeat):
            sen, V = sa(input, sequence_length, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option)
            # print(sen,V)
            if maxV<V:
                sampledsen = sen
                maxV = V
            appendtext(sampledsen, os.path.join(option.this_expsdir, option.save_path+'top-{}'.format(k)))


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


def sa(current_sentence, sequence_length, sta_vec, id2sen, emb_word, session, mtest_forward, mtest_backward, option):
    if option.mode == 'kw-bleu':
        similarity_func = similarity_keyword_bleu_tensor
    else:
        similarity_func = similarity_keyword
    
    similaritymodel = None
    input_original = current_sentence[0]
    sta_vec_original = [x for x in sta_vec]
    calibrated_set = set([x for x in current_sentence[0] if x < option.dict_size])

    for iteration in range(option.sample_time):
        temperature = option.temperatures[iteration]
        ind = iteration % (sequence_length[0] - 1)
        action = choose_action(option.action_prob)

        if action == 0: # word replacement (action: 0)

            # ---------------------------------------------------
            # prob_old=run_epoch(session, mtest_forward, input, sequence_length, mode='use')[0]
            # tem=1
            # for j in range(sequence_length[0]-1):
            #     tem*=prob_old[j][input[0][j+1]]
            # tem*=prob_old[j+1][option.dict_size+1]
            # prob_old_prob=tem
            # if sim!=None:
            #     similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word, option, similaritymodel)[0]
            #     prob_old_prob*=similarity_old
            # else:
            #     similarity_old=-1
            # ---------------------------------------------------
            # Compute the similarity scores of the current sentence to original
            prob_old_prob, similarity_old = compute_prob_sim_old(session, mtest_forward, current_sentence, input_original, \
                sequence_length, similarity_func, sta_vec, id2sen, emb_word, similaritymodel, option)


            # ---------------------------------------------------
            # # Separate input into forward/backward portions at the replacement point (ind)
            # input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
            #         cut_from_point(input, sequence_length, ind, option, mode=action)

            # # Use language model to get forward/backward probs for each candidate word in vocab
            # prob_forward = run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
            # prob_backward = run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
            # prob_mul = (prob_forward * prob_backward)

            # # Generate the N possible candidate sentences
            # input_candidate, sequence_length_candidate = generate_candidate_input_calibrated(input,\
            #         sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
            #     calibrated_set=calibrated_set)
            # ---------------------------------------------------
            # Generate the top N candidates sentences using forward/backward probabilities
            input_candidate, sequence_length_candidate = generate_N_candidates(session, mtest_forward, \
                mtest_backward, current_sentence, sequence_length, ind, option, action, list(calibrated_set))
            
            # ---------------------------------------------------
            # prob_candidate_pre = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
            # prob_candidate = []
            # for i in range(len(input_candidate)):
            #     tem = 1
            #     for j in range(sequence_length[0] - 1):
            #         tem *= prob_candidate_pre[i][j][input_candidate[i][j+1]]
            #     tem *= prob_candidate_pre[i][j+1][option.dict_size+1]
            #     prob_candidate.append(tem)
            # prob_candidate = np.array(prob_candidate)
            # ---------------------------------------------------
            # Compute fluency scores for each candidate
            fluency_candidates = compute_fluency(session, mtest_forward, input_candidate, sequence_length_candidate, option)
            
            # Compute the semantic preservation and expression diversity
            similarity_candidate = compute_semantic_preservation_expression_diversity(input_candidate, \
                input_original, sta_vec, id2sen, emb_word, option, similaritymodel, similarity_func)
            
            # Compute scores for candidates
            prob_candidate = fluency_candidates * similarity_candidate

            # Sample candidate from top N candidates
            candidate_prob, candidate_idx = sample_candidate(prob_candidate)
            
            # ------------------------------------------
            # V_new = math.log(max(np.power(candidate_prob,1.0/sequence_length),1e-200))
            # V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
            # alphat = min(1,math.exp(min((V_new-V_old)/temperature,100)))
            # ------------------------------------------
            # Find acceptance probability
            V_new, V_old, acceptance_prob = acceptance_proposal(candidate_prob, prob_old_prob, sequence_length, sequence_length, temperature)
            
            # If accepted, we add word to calibrated set and set the sentence to candidate sentence
            if is_accepted(acceptance_prob) and word_in_dict(input_candidate[candidate_idx][ind], option):
                add_to_calibrated(current_sentence[0][ind+1], calibrated_set, option)
                current_sentence = input_candidate[candidate_idx:candidate_idx+1]
                # print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action, prob_old_prob, V_old,\
                #     V_new, acceptance_prob, similarity_old, similarity_candidate[candidate_idx])
                # print('Temperature:{:3.3f}:   '.format(temperature) + ' '.join(id2sen(current_sentence[0])), sequence_length)


        elif action == 1: # word insert
            if sequence_length[0] >= option.num_steps:
                continue
            
            # ---------------------------------------------------
            # prob_old = run_epoch(session, mtest_forward, input, sequence_length, mode='use')[0]
            # tem=1
            # for j in range(sequence_length[0]-1):
            #     tem*=prob_old[j][input[0][j+1]]
            # tem*=prob_old[j+1][option.dict_size+1]
            # prob_old_prob=tem
            # if sim!=None:
            #     similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word, option, similaritymodel)[0]
            #     prob_old_prob=prob_old_prob*similarity_old
            # else:
            #     similarity_old=-1
            # ---------------------------------------------------
            # Compute the similarity scores of the current sentence to original
            prob_old_prob, similarity_old = compute_prob_sim_old(session, mtest_forward, current_sentence, input_original, \
                sequence_length, similarity_func, sta_vec, id2sen, emb_word, similaritymodel, option)

            # ---------------------------------------------------
            # Separate input into forward/backward portions at the insertion point (ind)
            # input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
            #         cut_from_point(input, sequence_length, ind, option, mode=action)

            # # Use language model to get forward/backward probs for each candidate word in vocab
            # prob_forward = run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
            # prob_backward = run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
            # prob_mul = (prob_forward * prob_backward)

            # # Generate the N possible candidate sentences
            # input_candidate, sequence_length_candidate = generate_candidate_input_calibrated(input,\
            #         sequence_length, ind, prob_mul, option.search_size, option, mode=action,\
            #     calibrated_set=calibrated_set)
            # ---------------------------------------------------
            # Generate the top N candidates sentences using forward/backward probabilities
            input_candidate, sequence_length_candidate = generate_N_candidates(session, mtest_forward, \
                mtest_backward, current_sentence, sequence_length, ind, option, action, list(calibrated_set))

            
            # ---------------------------------------------------
            # prob_candidate_pre = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
            # prob_candidate = []
            # for i in range(len(input_candidate)):
            #     tem = 1
            #     for j in range(sequence_length[0] - 1):
            #         tem *= prob_candidate_pre[i][j][input_candidate[i][j+1]]
            #     tem *= prob_candidate_pre[i][j+1][option.dict_size+1]
            #     prob_candidate.append(tem)
            # prob_candidate = np.array(prob_candidate)
            # ---------------------------------------------------
            # Compute fluency scores for each candidate
            fluency_candidates = compute_fluency(session, mtest_forward, input_candidate, sequence_length_candidate, option)


            # Compute the semantic preservation and expression diversity
            similarity_candidate = compute_semantic_preservation_expression_diversity(input_candidate, \
                input_original, sta_vec, id2sen, emb_word, option, similaritymodel, similarity_func)
            
            # Compute scores for candidates
            prob_candidate = fluency_candidates * similarity_candidate

            # Sample candidate from top N candidates
            candidate_prob, candidate_idx = sample_candidate(prob_candidate)


            # Just being used in print @REMOVE
            similarity_new = similarity_candidate[candidate_idx]

           
            # ------------------------------------------
            # Compute scores 
            # V_new = math.log(max(np.power(candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
            # V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
            # alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
            # ------------------------------------------
            # Find acceptance probability
            V_new, V_old, acceptance_prob = acceptance_proposal(candidate_prob, prob_old_prob, sequence_length_candidate[0], sequence_length, temperature)

            # If accepted, we add word to calibrated set and set the sentence to candidate sentence
            if is_accepted(acceptance_prob) and word_in_dict(input_candidate[candidate_idx][ind], option):
                current_sentence = input_candidate[candidate_idx : candidate_idx+1]
                sequence_length += 1
                # pos += 1
                # print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action, prob_old_prob, V_old,\
                #     V_new, acceptance_prob, similarity_old, similarity_new)
                # print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(current_sentence[0])), sequence_length)


        elif action==2: # word delete
            # Check if sentence is too short to delete a word
            if sequence_length[0] <= 2 or ind == 0:
                continue


            # ---------------------------------------------------
            # prob_old = run_epoch(session, mtest_forward, input, sequence_length, mode='use')[0]
            # tem = 1
            # for j in range(sequence_length[0] - 1):
            #     tem *= prob_old[j][input[0][j + 1]]
            # tem *= prob_old[j + 1][option.dict_size + 1]
            # prob_old_prob = tem
            # if sim != None:
            #     similarity_old = similarity(input, input_original, sta_vec, id2sen, emb_word, option, similaritymodel)[0]
            #     prob_old_prob = prob_old_prob * similarity_old
            # else:
            #     similarity_old = -1
            # ---------------------------------------------------
            # Compute the similarity scores of the current sentence to original
            prob_old_prob, similarity_old = compute_prob_sim_old(session, mtest_forward, current_sentence, input_original, \
                sequence_length, similarity_func, sta_vec, id2sen, emb_word, similaritymodel, option)

            # Generate the top N candidates sentences using forward/backward probabilities
            input_candidate, sequence_length_candidate=generate_candidate_input_calibrated(current_sentence,\
                sequence_length, ind, None, option.search_size, option,\
                mode=action, calibrated_set=list(calibrated_set))


            # ---------------------------------------------------
            # delete sentence
            # prob_new = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')[0]

            # tem=1
            # for j in range(sequence_length_candidate[0]-1):
            #     tem*=prob_new[j][input_candidate[0][j+1]]
            # tem*=prob_new[j+1][option.dict_size+1]
            # prob_new_prob=tem
            # ---------------------------------------------------
            # Compute fluency scores for each candidate
            fluency_candidates = compute_fluency(session, mtest_forward, input_candidate, sequence_length_candidate, option)
            fluency_candidate = fluency_candidates[0]

            # Compute the semantic preservation and expression diversity
            similarity_candidate = compute_semantic_preservation_expression_diversity(input_candidate, \
                input_original, sta_vec, id2sen, emb_word, option, similaritymodel, similarity_func)
            
            # Compute scores for candidates
            prob_candidate = fluency_candidates * similarity_candidate
            
            # Sample candidate from top N candidates
            # Since we only have one candiate, we just get back the single input candidate
            candidate_prob, candidate_idx = sample_candidate(prob_candidate)

            # ------------------------------------------
            #alpha is acceptance ratio of current proposal
            # V_new = math.log(max(np.power(prob_candidate_prob,1.0/sequence_length_candidate[0]),1e-200))
            # V_old = math.log(max(np.power(prob_old_prob, 1.0/sequence_length),1e-200))
            # alphat = min(1, math.exp((V_new-V_old)/temperature))
            # ------------------------------------------
            # Find acceptance probability
            V_new, V_old, acceptance_prob = acceptance_proposal(candidate_prob, prob_old_prob, sequence_length_candidate[0], sequence_length, temperature)

            # If accepted, we add word to calibrated set and set the sentence to candidate sentence
            if is_accepted(acceptance_prob):
                add_to_calibrated(current_sentence[0][ind], calibrated_set, option)
                current_sentence = input_candidate[candidate_idx : candidate_idx+1]
                sequence_length -= 1
                # pos -= 1
                # print('ind, action,oldprob,vold, vnew, alpha,simold, simnew', ind, action, prob_old_prob, V_old,\
                #     V_new, acceptance_prob, similarity_old, similarity_candidate)
                # print('Temperature:{:3.3f}:   '.format(temperature) + ' '.join(id2sen(current_sentence[0])), sequence_length)

    return ' '.join(id2sen(current_sentence[0])), V_old
